"""本地 Mock LLM 服务（OpenAI 兼容协议）。

项目验收环境没有可用的 DeepSeek API Key，为了验证「真实 function calling / 流式输出」
代码路径，这里用 Flask 起一个本地 OpenAI 兼容服务，供集成测试把
``config.DEEPSEEK_BASE_URL`` 指向它：

- ``POST /v1/chat/completions``：支持非流式与流式（SSE）两种响应；
- 只要请求里带 ``tools`` 且强制了 ``detect_intent``，就返回 detect_intent 工具调用；
- 主对话循环第一轮返回 ``search_course_docs`` / ``search_market_docs`` 工具调用，
  收到 tool 结果后的第二轮返回最终中文回答（流式逐块输出）。

这样可以在完全离线（本机 127.0.0.1）的情况下，完整走通
「意图工具调用 → 检索工具调用 → 最终流式回答」的链路。
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from typing import Any, Iterator

from flask import Flask, Response, jsonify, request
from werkzeug.serving import make_server

# 记录收到的请求，便于测试断言（进程内共享，测试单线程跑足够）
REQUESTS: list[dict[str, Any]] = []

FINAL_ANSWER = (
    "根据检索结果：Python 全栈实战课程价格 6980 元、周期 12 周，适合零基础学员；"
    "AI 应用开发课程价格 12800 元、周期 16 周。\n\n"
    "方便告诉我您目前的基础和每周可投入的学习时间吗？我帮您匹配更合适的班型。"
)


def reset_requests() -> None:
    """清空请求记录。"""
    REQUESTS.clear()


def _tool_result_messages(messages: list[dict[str, Any]]) -> bool:
    """判断消息列表中是否已经包含工具执行结果。"""
    return any(str(item.get("role") or "") == "tool" for item in messages)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """取最后一条 user 消息的文本（用于 mock 意图判断）。"""
    for item in reversed(messages or []):
        if str(item.get("role") or "") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return ""


def _decide(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, tool_choice: Any) -> dict[str, Any]:
    """根据请求内容决定 mock 返回「工具调用」还是「文本回答」。"""
    names = [str((tool.get("function") or {}).get("name") or "") for tool in (tools or [])]
    forced_name = ""
    if isinstance(tool_choice, dict):
        forced_name = str((tool_choice.get("function") or {}).get("name") or "")
    elif isinstance(tool_choice, str):
        forced_name = tool_choice

    # 1) 意图识别：强制工具调用，按用户问题给出版本号
    if forced_name == "detect_intent" or ("detect_intent" in names and forced_name):
        query = _last_user_text(messages)
        intent_code = 1 if any(word in query for word in ("职业", "岗位", "薪资", "就业", "转行")) else 2
        if any(word in query for word in ("天气", "闲聊", "你好")):
            intent_code = 0
        return {
            "type": "tool",
            "tool_calls": [
                {
                    "id": "call_intent_1",
                    "type": "function",
                    "function": {
                        "name": "detect_intent",
                        "arguments": json.dumps({"intent_code": intent_code, "reason": f"mock 判定意图为 {intent_code}"}, ensure_ascii=False),
                    },
                }
            ],
        }

    # 2) 已经看过工具结果 → 返回最终回答
    if _tool_result_messages(messages):
        return {"type": "text", "content": FINAL_ANSWER}

    # 3) 主对话第一轮：调用检索工具
    if "search_course_docs" in names:
        return {
            "type": "tool",
            "tool_calls": [
                {
                    "id": "call_search_1",
                    "type": "function",
                    "function": {
                        "name": "search_course_docs",
                        "arguments": json.dumps({"query": "Python 课程价格", "top_k": 2}, ensure_ascii=False),
                    },
                }
            ],
        }
    if "search_market_docs" in names:
        return {
            "type": "tool",
            "tool_calls": [
                {
                    "id": "call_search_2",
                    "type": "function",
                    "function": {
                        "name": "search_market_docs",
                        "arguments": json.dumps({"query": "AI 职业发展前景", "top_k": 2}, ensure_ascii=False),
                    },
                }
            ],
        }
    return {"type": "text", "content": "（mock）这是测试环境下的模型回答。"}


def _completion_payload(model: str, decision: dict[str, Any], stream: bool) -> dict[str, Any]:
    """构造非流式 chat.completion 响应体。"""
    if decision["type"] == "tool":
        message = {"role": "assistant", "content": None, "tool_calls": decision["tool_calls"]}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": decision["content"]}
        finish_reason = "stop"
    return {
        "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def _sse_frame(payload: dict[str, Any]) -> str:
    """序列化为一条 SSE 数据帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_chunks(model: str, decision: dict[str, Any]) -> Iterator[str]:
    """把 mock 决策拆成 OpenAI 风格的流式 chunk。"""
    chunk_id = f"chatcmpl-mock-{int(time.time() * 1000)}"
    created = int(time.time())

    def base(delta: dict[str, Any], finish_reason: Any = None) -> dict[str, Any]:
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    if decision["type"] == "tool":
        for index, tool_call in enumerate(decision["tool_calls"]):
            arguments = tool_call["function"]["arguments"]
            # 先发工具名与 id，再分片发 arguments，模拟真实流式工具调用
            yield _sse_frame(
                base(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": index,
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {"name": tool_call["function"]["name"], "arguments": ""},
                            }
                        ],
                    }
                )
            )
            for start in range(0, len(arguments), 8):
                yield _sse_frame(
                    base(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "function": {"arguments": arguments[start : start + 8]},
                                }
                            ]
                        }
                    )
                )
        yield _sse_frame(base({}, finish_reason="tool_calls"))
        yield "data: [DONE]\n\n"
        return

    content = decision["content"]
    yield _sse_frame(base({"role": "assistant", "content": ""}))
    for start in range(0, len(content), 6):
        yield _sse_frame(base({"content": content[start : start + 6]}))
    yield _sse_frame(base({}, finish_reason="stop"))
    yield "data: [DONE]\n\n"


def create_mock_llm_app() -> Flask:
    """创建 mock OpenAI 服务 Flask 应用。"""
    app = Flask(__name__)

    @app.post("/v1/chat/completions")
    def chat_completions():  # noqa: ANN202 - Flask 视图
        payload = request.get_json(silent=True) or {}
        messages = list(payload.get("messages") or [])
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")
        stream = bool(payload.get("stream"))
        model = str(payload.get("model") or "mock-deepseek")

        REQUESTS.append(
            {
                "model": model,
                "messages": messages,
                "tools": [str((tool.get("function") or {}).get("name") or "") for tool in (tools or [])],
                "tool_choice": tool_choice,
                "stream": stream,
            }
        )

        decision = _decide(messages, tools, tool_choice)
        if stream:
            return Response(_stream_chunks(model, decision), mimetype="text/event-stream")
        return jsonify(_completion_payload(model, decision, stream))

    @app.post("/chat/completions")
    def chat_completions_compat():  # noqa: ANN202 - 兼容 base_url 不带 /v1 的写法
        return chat_completions()

    return app


@contextlib.contextmanager
def mock_llm_server() -> Iterator[str]:
    """启动 mock 服务，返回可写入 ``DEEPSEEK_BASE_URL`` 的 base_url。

    Yields:
        形如 ``http://127.0.0.1:端口/v1`` 的地址。
    """
    app = create_mock_llm_app()
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)
