"""HTTP 接口测试：/health、/chat（SSE 与非流式）、向量库维护接口、SQL Agent 接口。"""

from __future__ import annotations

import json
import uuid


def _parse_sse(raw: str) -> list[dict]:
    """把 SSE 文本解析成事件字典列表。"""
    events: list[dict] = []
    for frame in raw.split("\n\n"):
        line = frame.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def test_health(client):
    """健康检查返回各组件状态。"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    for key in ("llm", "embedding", "vectorstore", "database", "search"):
        assert key in data
    assert "users" in data["database"]


def test_chat_stream(client):
    """SSE 流式对话：新用户返回欢迎语，事件以 [DONE] 结束。"""
    phone = "13900002001"
    response = client.post("/chat", json={"phone_number": phone, "query": "你好", "stream": True})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["Content-Type"]
    raw = response.get_data(as_text=True)
    assert raw.strip().endswith("data: [DONE]")

    events = _parse_sse(raw)
    types = [event["type"] for event in events]
    assert types[0] == "start"
    assert "delta" in types
    assert "done" in types
    done = [event for event in events if event["type"] == "done"][-1]
    assert done["mode"] == "welcome"
    assert done["reply"].strip()


def test_chat_non_stream(client):
    """非流式对话：返回 JSON，包含意图、回答与工具调用。"""
    phone = "13900002002"
    client.post("/chat", json={"phone_number": phone, "query": "你好", "stream": False})

    response = client.post("/chat", json={"phone_number": phone, "query": "Python 课程多少钱？", "stream": False})
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert data["intent_code"] == 2
    assert data["reply"].strip()
    assert isinstance(data["tool_calls"], list)
    assert data["user_info"]["phone_number"] == phone


def test_chat_missing_params(client):
    """缺少手机号 / 问题时返回 400 与中文错误信息。"""
    response = client.post("/chat", json={"query": "你好", "stream": False})
    assert response.status_code == 400
    assert response.get_json()["ok"] is False

    response = client.post("/chat", json={"phone_number": "13900002003", "stream": False})
    assert response.status_code == 400


def test_vectorstore_list(client):
    """向量库列表包含 market / course 两个集合，且已有示例文档。"""
    response = client.get("/vectorstore/list")
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    names = {item["name"]: item["count"] for item in data["collections"]}
    assert "market" in names and "course" in names
    assert names["market"] > 0 and names["course"] > 0


def test_vectorstore_crud_and_search(client):
    """向量库新增 → 检索命中 → 更新生效 → 删除生效（对外开放的维护接口）。"""
    collection = "course"
    doc_id = f"api-test-{uuid.uuid4().hex[:8]}"
    original = "接口测试专用：量子计算入门课程，价格 9999 元，周期 8 周。"
    updated = "接口测试专用：脑机接口进阶课程，价格 18888 元，周期 10 周。"

    try:
        response = client.post(
            "/vectorstore/add",
            json={"collection": collection, "documents": [original], "ids": [doc_id], "metadatas": [{"source": "api-test"}]},
        )
        assert response.status_code == 200, response.get_data(as_text=True)
        assert response.get_json()["ok"] is True

        response = client.post("/vectorstore/search", json={"collection": collection, "query": "量子计算课程价格", "top_k": 3})
        results = response.get_json()["results"]
        assert any(item["id"] == doc_id for item in results)

        response = client.post(
            "/vectorstore/update",
            json={"collection": collection, "ids": [doc_id], "documents": [updated]},
        )
        assert response.status_code == 200
        response = client.post("/vectorstore/search", json={"collection": collection, "query": "脑机接口进阶课程", "top_k": 3})
        results = response.get_json()["results"]
        hit = [item for item in results if item["id"] == doc_id]
        assert hit and "脑机接口" in hit[0]["document"]

        response = client.post("/vectorstore/delete", json={"collection": collection, "ids": [doc_id]})
        assert response.status_code == 200
        response = client.post("/vectorstore/search", json={"collection": collection, "query": "量子计算课程价格", "top_k": 3})
        results = response.get_json()["results"]
        assert all(item["id"] != doc_id for item in results)
    finally:
        # 清理测试数据，避免影响其它用例
        client.post("/vectorstore/delete", json={"collection": collection, "ids": [doc_id]})


def test_vectorstore_invalid_collection(client):
    """非法集合名返回 400。"""
    response = client.post("/vectorstore/add", json={"collection": "unknown_collection", "documents": ["测试"]})
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_database_query_api(client):
    """SQL Agent 接口：自然语言查询课程。"""
    response = client.post("/database/query", json={"question": "数据库里有哪些课程？"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert data["rows"]
    assert data["mode"] in ("llm", "rule")


def test_database_schema_api(client):
    """表结构接口返回 4 张表。"""
    response = client.get("/database/schema")
    assert response.status_code == 200
    data = response.get_json()
    assert set(data["tables"]) >= {"users", "chat_history", "courses", "career_info"}


def test_404_returns_json(client):
    """未知接口返回 JSON 而不是 HTML。"""
    response = client.get("/not-exists")
    assert response.status_code == 404
    assert response.get_json()["ok"] is False


def test_index_html_inline_javascript_not_double_escaped(client):
    """回归测试：首页内联 JS 中的反斜杠不能被二次转义。

    历史 bug：模板写成 ``/^1[3-9]\\\\d{9}$/``（字面双反斜杠），JS 里 ``\\\\d``
    只匹配「反斜杠 + d」，导致合法 11 位手机号被前端拦截；``split("\\\\n\\\\n")``
    也拆不开 SSE 帧。这里直接校验浏览器实际拿到的 HTML。
    """
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # 手机号正则：必须单反斜杠；双反斜杠形态视为回归
    assert r"/^1[3-9]\d{9}$/.test(phone)" in html
    assert r"\\d{9}" not in html

    # SSE 分帧：必须按真实换行 \n\n 分割，而非字面反斜杠
    assert html.count('split("\\n\\n")') == 1
    assert 'split("\\\\n\\\\n")' not in html

    # 工具列表换行 join 同理
    assert '.join("\\n")' in html
    assert '.join("\\\\n")' not in html

    # 顶部技术栈徽标已按使用反馈移除（那行字在深色背景下看不清）
    assert 'class="badge"' not in html
    assert "DeepSeek-Flash · ChromaDB · SQLite" not in html
    # 工具调用明细必须作为独立块渲染，不能塞进 .meta 的 flex 标签行
    assert '<details class="tools">' in html
    assert ".meta details" not in html
