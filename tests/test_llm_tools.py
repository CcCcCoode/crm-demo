"""AI 能力层与工具层单元测试（无 API Key、无网络环境下必须全部通过）。

覆盖范围（对应 docs/CONTRACT.md §5 / §6）：
1. ``keyword_intent`` 三个验收用例 + 加权/tie 规则；
2. ``extract_json`` 的纯 JSON / Markdown 代码块 / 前后夹文字 / 非法输入；
3. 无 DEEPSEEK_API_KEY 时 client 必须抛 LLMUnavailableError，状态接口正确；
4. ``get_tool_names`` / ``get_openai_tools`` / ``TOOL_SCHEMAS`` 结构；
5. ``execute_tool`` 六个工具的真实执行（database + vectorstore 由 conftest 初始化）；
6. 关闭联网搜索时 ``web_search`` 返回本地兜底或 ``[]`` 且绝不抛异常。

运行：``python -m pytest tests/test_llm_tools.py -q``
"""

from __future__ import annotations

import pytest

from config import config
from database import db
from llm import (
    INTENT_CAREER,
    INTENT_COURSE,
    INTENT_OTHER,
    LLMUnavailableError,
    chat_completion,
    detect_intent_code,
    detect_intent_with_reason,
    extract_json,
    get_llm_status,
    get_search_status,
    is_llm_available,
    is_search_available,
    keyword_intent,
    reset_client,
    reset_search_status,
    run_streaming_tool_loop,
    run_tool_calling_loop,
    stream_completion,
    web_search,
)
from tools import TOOL_SCHEMAS, execute_tool, get_openai_tools, get_tool_names

EXPECTED_TOOL_NAMES = [
    "detect_intent",
    "web_search",
    "search_market_docs",
    "search_course_docs",
    "query_database",
    "save_chat_record",
]

# 测试专用手机号（chat_history 以 phone_number 关联，不依赖 users 表）
TEST_PHONE = "13900000001"


# ---------------------------------------------------------------------------
# 1. keyword_intent 验收用例
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("AI 职业规划", INTENT_CAREER),
        ("Python 课程多少钱", INTENT_COURSE),
        ("今天天气", INTENT_OTHER),
    ],
)
def test_keyword_intent_acceptance(query: str, expected: int) -> None:
    """契约要求的三个验收用例：职业→1、课程→2、其他→0。"""
    assert keyword_intent(query) == expected


def test_keyword_intent_weighted_rules() -> None:
    """加权打分细节：课程词权重 2、职业词权重 1；同分视为无法判断返回 0。"""
    assert keyword_intent("我该转行做 AI 吗") == INTENT_CAREER  # 转行
    assert keyword_intent("课程价格是多少") == INTENT_COURSE  # 课程(+2)+价格(+1)
    assert keyword_intent("报名试听") == INTENT_COURSE
    assert keyword_intent("岗位和课程怎么选") == INTENT_COURSE  # 课程 2 分 > 岗位 1 分
    assert keyword_intent("职业规划 和 课程 哪个好") == INTENT_OTHER  # 2:2 同分 → 无法判断
    assert keyword_intent("") == INTENT_OTHER


def test_detect_intent_fallback_offline() -> None:
    """无 Key 时 detect_intent_with_reason 必须降级为规则模式且不抛异常。"""
    result = detect_intent_with_reason("Python 课程多少钱")
    assert result["intent_code"] == INTENT_COURSE
    assert result["mode"] == "rule"
    assert result["reason"]
    assert detect_intent_code("AI 职业规划") == INTENT_CAREER
    # 空输入不应崩溃
    assert detect_intent_with_reason("")["intent_code"] == INTENT_OTHER


# ---------------------------------------------------------------------------
# 2. extract_json
# ---------------------------------------------------------------------------
def test_extract_json_plain_object() -> None:
    assert extract_json('{"intent_code": 1, "reason": "职业"}') == {
        "intent_code": 1,
        "reason": "职业",
    }


def test_extract_json_markdown_code_block() -> None:
    text = '模型输出如下：\n```json\n{"name": "张三", "budget": 12000}\n```\n请查收。'
    assert extract_json(text) == {"name": "张三", "budget": 12000}


def test_extract_json_surrounded_by_text() -> None:
    text = '好的，结果是 {"sql": "SELECT id FROM courses LIMIT 3"} 希望有帮助'
    assert extract_json(text) == {"sql": "SELECT id FROM courses LIMIT 3"}


def test_extract_json_invalid_returns_none() -> None:
    assert extract_json("这里没有任何 JSON") is None
    assert extract_json("") is None
    assert extract_json("{不是合法 JSON}") is None
    assert extract_json(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. LLM client 无 Key 行为
# ---------------------------------------------------------------------------
def test_llm_client_without_api_key_raises() -> None:
    """无 Key：客户端不可用、状态准确、所有模型调用抛 LLMUnavailableError。"""
    reset_client()
    assert config.DEEPSEEK_API_KEY == ""
    assert is_llm_available() is False

    status = get_llm_status()
    assert status["available"] is False
    assert status["model"]
    assert status["base_url"]

    messages = [{"role": "user", "content": "你好"}]
    with pytest.raises(LLMUnavailableError):
        chat_completion(messages)
    with pytest.raises(LLMUnavailableError):
        list(stream_completion(messages))
    with pytest.raises(LLMUnavailableError):
        run_tool_calling_loop(messages, TOOL_SCHEMAS, lambda name, arguments: {})
    with pytest.raises(LLMUnavailableError):
        list(
            run_streaming_tool_loop(messages, TOOL_SCHEMAS, lambda name, arguments: {})
        )


# ---------------------------------------------------------------------------
# 4. 工具 schema 结构
# ---------------------------------------------------------------------------
def test_tool_schemas_and_names() -> None:
    """6 个工具 schema 必须是合法 OpenAI tools 结构，名称唯一且顺序固定。"""
    assert get_tool_names() == EXPECTED_TOOL_NAMES
    assert len(TOOL_SCHEMAS) == 6
    assert len({schema["function"]["name"] for schema in TOOL_SCHEMAS}) == 6

    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] in EXPECTED_TOOL_NAMES
        assert function["description"].strip()
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert isinstance(parameters["properties"], dict)
        assert parameters["properties"]
        assert isinstance(parameters["required"], list)


def test_get_openai_tools_filter_and_copy() -> None:
    """get_openai_tools 支持按名称过滤，且返回深拷贝不污染 TOOL_SCHEMAS。"""
    all_tools = get_openai_tools()
    assert len(all_tools) == 6
    assert [tool["function"]["name"] for tool in all_tools] == EXPECTED_TOOL_NAMES

    subset = get_openai_tools(["web_search", "query_database"])
    assert [tool["function"]["name"] for tool in subset] == [
        "web_search",
        "query_database",
    ]

    subset[0]["function"]["description"] = "被修改"
    again = get_openai_tools(["web_search"])
    assert again[0]["function"]["description"] != "被修改"

    # 未知名称应被忽略，不应抛异常
    assert get_openai_tools(["not_exists"]) == []


# ---------------------------------------------------------------------------
# 5. execute_tool：统一返回结构与六个工具
# ---------------------------------------------------------------------------
def test_execute_tool_unknown_and_bad_arguments() -> None:
    """未知工具 / 非法参数必须返回 ok=False，而不是抛异常。"""
    result = execute_tool("not_a_tool")
    assert result["ok"] is False
    assert result["tool"] == "not_a_tool"
    assert result["result"] is None
    assert result["error"]

    for arguments in ("{不是合法 JSON}", 12345):
        bad = execute_tool("web_search", arguments)  # type: ignore[arg-type]
        assert bad["ok"] is False
        assert bad["error"]


def test_execute_tool_detect_intent() -> None:
    """detect_intent：校验 intent_code；只给 query 时自动识别（离线路由）。"""
    ok = execute_tool(
        "detect_intent", {"intent_code": INTENT_COURSE, "reason": "用户询问课程价格"}
    )
    assert ok["ok"] is True, ok["error"]
    assert ok["result"] == {"intent_code": INTENT_COURSE, "reason": "用户询问课程价格"}

    bad = execute_tool("detect_intent", {"intent_code": 9})
    assert bad["ok"] is False
    assert "intent_code" in bad["error"]

    by_query = execute_tool("detect_intent", {"query": "AI 职业规划"})
    assert by_query["ok"] is True, by_query["error"]
    assert by_query["result"]["intent_code"] == INTENT_CAREER
    assert by_query["result"]["reason"]


def test_execute_tool_web_search_offline_fallback() -> None:
    """关闭联网搜索时走本地职业知识库兜底：结构完整、不抛异常、状态可读。"""
    reset_search_status()
    assert is_search_available() is False

    result = execute_tool(
        "web_search", {"query": "AI 应用开发工程师 薪资", "top_k": 3}
    )
    assert result["ok"] is True, result["error"]
    payload = result["result"]
    assert payload["query"] == "AI 应用开发工程师 薪资"
    assert payload["count"] == len(payload["results"])
    assert payload["count"] >= 1  # career_info 已由 conftest 灌入示例数据
    assert payload["count"] <= 3
    for item in payload["results"]:
        assert set(item) == {"title", "snippet", "url", "source", "provider"}
        assert item["title"]
        assert item["source"] == "本地知识库兜底"
        assert item["provider"] == "local"

    status = get_search_status()
    assert set(status) == {
        "enabled",
        "provider",
        "last_error",
        "fallback_used",
        "calls",
        "cooldown_remaining",
    }
    assert status["enabled"] is False
    assert status["calls"] >= 1
    assert status["fallback_used"] is True
    assert status["cooldown_remaining"] == 0.0


def test_web_search_never_raises() -> None:
    """web_search 是兜底入口：空词/正常词都必须返回 list，绝不抛异常。"""
    assert web_search("") == []
    assert isinstance(web_search("AI 行业前景", top_k=2), list)
    assert isinstance(web_search("Python 课程", top_k=999), list)


def test_execute_tool_search_documents() -> None:
    """两个知识库检索工具：返回集合名、结果列表与条数，且能命中示例文档。"""
    market = execute_tool(
        "search_market_docs", {"query": "AI 应用开发 技能要求", "top_k": 3}
    )
    assert market["ok"] is True, market["error"]
    market_payload = market["result"]
    assert market_payload["collection"] == config.COLLECTION_MARKET
    assert market_payload["count"] == len(market_payload["results"])
    assert market_payload["count"] >= 1
    assert "document" in market_payload["results"][0]

    course = execute_tool(
        "search_course_docs", {"query": "零基础 Python 课程 价格", "top_k": 3}
    )
    assert course["ok"] is True, course["error"]
    course_payload = course["result"]
    assert course_payload["collection"] == config.COLLECTION_COURSE
    assert course_payload["count"] == len(course_payload["results"])
    assert course_payload["count"] >= 1
    assert "document" in course_payload["results"][0]

    missing = execute_tool("search_course_docs", {})
    assert missing["ok"] is False


def test_execute_tool_query_database() -> None:
    """query_database：执行只读 SELECT 返回行；写操作必须被拒绝但不抛异常。"""
    result = execute_tool(
        "query_database",
        {"sql": "SELECT name, price FROM courses ORDER BY price DESC LIMIT 3"},
    )
    assert result["ok"] is True, result["error"]
    payload = result["result"]
    assert payload["sql"].upper().startswith("SELECT")
    assert payload["row_count"] == len(payload["rows"]) == 3
    assert payload["rows"][0]["name"]
    assert payload["rows"][0]["price"] > 0

    write_attempt = execute_tool("query_database", {"sql": "DELETE FROM courses"})
    assert write_attempt["ok"] is False
    assert write_attempt["error"]

    missing = execute_tool("query_database", {})
    assert missing["ok"] is False
    # 拒绝写操作后数据仍在
    assert db.execute_sql("SELECT COUNT(*) AS n FROM courses")[0]["n"] >= 6


def test_execute_tool_save_chat_record() -> None:
    """save_chat_record：优先用 context 中的手机号，写入后可查回。"""
    content = "我想了解一下 AI 职业规划，目前是后端开发"
    result = execute_tool(
        "save_chat_record",
        {"role": "user", "content": content},
        context={"phone_number": TEST_PHONE, "intent_code": INTENT_CAREER},
    )
    assert result["ok"] is True, result["error"]
    assert result["result"]["saved"] is True
    assert isinstance(result["result"]["id"], int)
    assert result["result"]["id"] > 0

    history = db.fetch_chat_history(TEST_PHONE)
    assert any(message["content"] == content for message in history)
    saved = next(message for message in history if message["content"] == content)
    assert saved["role"] == "user"
    assert saved["intent_code"] == INTENT_CAREER

    # 参数里显式给 phone_number 也应可用
    explicit = execute_tool(
        "save_chat_record",
        {"phone_number": TEST_PHONE, "role": "assistant", "content": "好的，我来帮你规划"},
    )
    assert explicit["ok"] is True, explicit["error"]

    # 缺少手机号 / 非法角色 → ok=False 且不抛异常
    no_phone = execute_tool("save_chat_record", {"role": "user", "content": "x"})
    assert no_phone["ok"] is False
    assert "phone_number" in no_phone["error"]

    bad_role = execute_tool(
        "save_chat_record",
        {"phone_number": TEST_PHONE, "role": "system", "content": "x"},
    )
    assert bad_role["ok"] is False


def test_parse_bing_search_results() -> None:
    """Bing 搜索页解析：提取标题/摘要/链接、清洗 HTML 标签与实体、滤掉非 http 链接。"""
    from llm.web_search import _parse_bing

    html = (
        '<ol id="b_results">'
        '<li class="b_algo"><h2><a href="https://example.com/a">AI 工程师薪资 <strong>2025</strong></a></h2>'
        '<div><p>月薪 15K-28K，&#x4E00;&#x7EBF;&#x57CE;&#x5E02;。</p></div></li>'
        '<li class="b_algo"><h2><a href="/relative/link">相对链接应被过滤</a></h2><p>x</p></li>'
        '<li class="b_algo"><h2><a href="https://example.com/b">无摘要结果</a></h2></li>'
        "</ol>"
    )
    parsed = _parse_bing(html)
    assert len(parsed) == 2
    assert parsed[0]["title"] == "AI 工程师薪资 2025"
    assert parsed[0]["url"] == "https://example.com/a"
    assert "15K-28K" in parsed[0]["snippet"]
    assert parsed[1]["title"] == "无摘要结果"
    assert parsed[1]["snippet"] == ""
    # 空内容/无法解析时返回空列表而不是抛异常
    assert _parse_bing("") == []
    assert _parse_bing("<html>nothing</html>") == []
