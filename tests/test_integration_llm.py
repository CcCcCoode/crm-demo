"""真实 Function Calling / 流式路径集成测试（使用本地 mock OpenAI 兼容服务）。

验收环境没有 DeepSeek API Key，因此这里把 ``DEEPSEEK_BASE_URL`` 指向本机
``tests/mock_llm_server.py``，验证：
- ``run_tool_calling_loop`` 完整执行工具调用并拿到最终回答；
- ``run_streaming_tool_loop`` 逐块输出 delta 与 final；
- 意图识别走 LLM 工具调用；
- ``/chat`` 在「有 Key」模式下真的走「意图工具 → 检索工具 → 流式回答」链路。
"""

from __future__ import annotations

import pytest

import llm.client as llm_client
from config import config
from llm.intent import detect_intent_with_reason
from tests.mock_llm_server import REQUESTS, mock_llm_server, reset_requests
from tools.functions import execute_tool, get_openai_tools


@pytest.fixture(scope="module")
def mock_llm():
    """启动 mock 服务并临时把 LLM 配置指向它。"""
    with mock_llm_server() as base_url:
        old_key = config.DEEPSEEK_API_KEY
        old_base = config.DEEPSEEK_BASE_URL
        config.DEEPSEEK_API_KEY = "test-key-for-mock"
        config.DEEPSEEK_BASE_URL = base_url
        llm_client.reset_client()
        reset_requests()
        try:
            yield base_url
        finally:
            config.DEEPSEEK_API_KEY = old_key
            config.DEEPSEEK_BASE_URL = old_base
            llm_client.reset_client()


def _executor(calls: list[str]):
    """构造一个记录调用并返回真实工具结果的执行器。"""

    def executor(name: str, arguments: dict):
        calls.append(name)
        return execute_tool(name, arguments or {}, context={"phone_number": "13800138000"})

    return executor


def test_run_tool_calling_loop(mock_llm):
    """非流式工具循环：模型调用检索工具 → 执行器执行 → 模型给出最终回答。"""
    calls: list[str] = []
    messages = [{"role": "user", "content": "Python 课程多少钱？"}]
    result = llm_client.run_tool_calling_loop(
        messages=messages,
        tools=get_openai_tools(),
        tool_executor=_executor(calls),
    )

    assert result["mode"] == "llm"
    assert calls, "应至少执行一次工具调用"
    assert result["tool_calls"]
    assert "6980" in result["content"]
    assert any(request["tools"] for request in REQUESTS)


def test_run_streaming_tool_loop(mock_llm):
    """流式工具循环：tool_call → delta → final，且文本连续。"""
    calls: list[str] = []
    messages = [{"role": "user", "content": "Python 课程多少钱？"}]
    events = list(
        llm_client.run_streaming_tool_loop(
            messages=messages,
            tools=get_openai_tools(),
            tool_executor=_executor(calls),
        )
    )

    types = [event["type"] for event in events]
    assert "tool_call" in types
    assert "delta" in types
    assert types[-1] == "final"

    final = events[-1]
    streamed = "".join(event["content"] for event in events if event["type"] == "delta")
    assert streamed.strip() == final["content"].strip()
    assert "6980" in final["content"]


def test_detect_intent_via_llm(mock_llm):
    """意图识别走 LLM 工具调用（mock 强制 detect_intent）。"""
    result = detect_intent_with_reason("Python 课程多少钱？")
    assert result["intent_code"] == 2
    assert result["mode"] == "llm"
    assert result["reason"]


def test_chat_end_to_end_with_llm(client, mock_llm):
    """端到端：有 Key 模式下 /chat 走真实 function calling + 流式回答。"""
    phone = "13900003001"
    first = client.post("/chat", json={"phone_number": phone, "query": "你好", "stream": False})
    assert first.status_code == 200
    assert first.get_json()["mode"] == "welcome"

    response = client.post("/chat", json={"phone_number": phone, "query": "Python 课程多少钱？", "stream": False})
    assert response.status_code == 200
    data = response.get_json()
    assert data["mode"] == "llm"
    assert "6980" in data["reply"]
    tool_names = [item["name"] for item in data["tool_calls"]]
    assert "search_course_docs" in tool_names

    # 流式接口同样可用（SSE 中能看到 delta 与 done）
    response = client.post("/chat", json={"phone_number": phone, "query": "再推荐一门课程", "stream": True})
    raw = response.get_data(as_text=True)
    assert "data: [DONE]" in raw
    assert '"type": "delta"' in raw or '"type":"delta"' in raw
