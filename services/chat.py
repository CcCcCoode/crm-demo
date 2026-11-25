"""主对话流程服务。

完整链路（对应开发提示词的主流程）：

1. 校验手机号与问题文本；
2. 读取用户画像并创建用户（新用户）；
3. 读取聊天历史：**没有历史 → 直接返回欢迎语**（不进入意图识别）；
4. 有历史 → 意图识别（1 = AI 职业咨询 / 2 = 课程咨询 / 0 = 其他）；
5. 按意图用 Function Calling 工具检索网络与向量库资料；
6. 把「用户画像 + 对话历史 + 检索资料」拼进系统提示词，调用 DeepSeek-Flash 流式回答；
   （无 API Key / 模型失败时自动降级为离线模板回答，保证系统可运行）
7. 写回聊天记录，并根据本轮对话更新用户画像；
8. 通过 ``stream_chat`` 产出逐事件流，``app.py`` 再转成 SSE 响应。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterator

from config import config, get_logger
from database import db
from llm import prompts
from llm.client import (
    LLMResponseError,
    LLMUnavailableError,
    is_llm_available,
    run_streaming_tool_loop,
)
from llm.intent import INTENT_OTHER, detect_intent_with_reason
from services import rag, user_info as user_info_service
from tools.functions import execute_tool, get_openai_tools

logger = get_logger("crm.services.chat")

# 欢迎语分支的意图编号（欢迎语不进入意图分类，仅用于响应结构统一）
WELCOME_INTENT = 0


def validate_query(query: str) -> str:
    """校验并规范化用户问题。

    Args:
        query: 用户原始输入。

    Returns:
        去掉首尾空白后的问题文本。

    Raises:
        ValueError: 问题为空或超长，附带中文提示。
    """
    text = str(query or "").strip()
    if not text:
        raise ValueError("请输入您想咨询的问题")
    if len(text) > config.MAX_QUERY_LENGTH:
        raise ValueError(f"问题太长了，请控制在 {config.MAX_QUERY_LENGTH} 字以内")
    return text


def build_llm_messages(
    user_info: dict | None,
    intent_code: int,
    history: list[dict[str, Any]],
    query: str,
    context_block: str,
) -> list[dict[str, Any]]:
    """构造主回答的 OpenAI messages（系统提示词 + 历史 + 本轮问题）。

    Args:
        user_info: 用户画像。
        intent_code: 意图编号。
        history: SQLite 聊天历史（时间正序）。
        query: 用户本轮问题。
        context_block: 预检索资料文本。

    Returns:
        OpenAI 格式的 messages 列表。
    """
    system_prompt = prompts.build_system_prompt(
        user_info=user_info,
        intent_code=intent_code,
        context_block=context_block or "（暂无，需调用工具检索）",
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(prompts.history_to_messages(history, limit=config.CHAT_HISTORY_LIMIT))
    messages.append({"role": "user", "content": query})
    return messages


def build_fallback_reply(
    intent_code: int,
    query: str,
    user_info: dict | None,
    evidence: dict[str, Any] | None = None,
) -> str:
    """按意图生成离线降级回答（无 API Key / 模型异常时使用）。"""
    evidence = evidence or {}
    return prompts.render_fallback_answer(
        intent_code=intent_code,
        query=query,
        user_info=user_info,
        web_results=evidence.get("web_results"),
        market_docs=evidence.get("market_docs"),
        course_docs=evidence.get("course_docs"),
    )


def _make_streaming_executor(
    phone_number: str,
    tool_calls: list[dict[str, Any]],
) -> Callable[[str, dict[str, Any]], Any]:
    """构造流式 function calling 循环使用的工具执行器。

    每个工具都会：记录日志、把调用信息加入 ``tool_calls``、返回可 JSON 序列化结果。

    Args:
        phone_number: 当前用户手机号（供 save_chat_record 等工具使用）。
        tool_calls: 工具调用记录列表（原地追加）。

    Returns:
        ``executor(name, arguments)`` 函数。
    """

    def executor(name: str, arguments: dict[str, Any] | None = None) -> Any:
        args = arguments or {}
        logger.info("[tool_call] 模型请求调用工具：%s 参数=%s", name, json.dumps(args, ensure_ascii=False))
        try:
            outcome = execute_tool(name, args, context={"phone_number": phone_number})
        except Exception as exc:  # noqa: BLE001 - 工具异常不能打断对话
            logger.error("[tool_call] 工具 %s 执行异常：%s", name, exc)
            outcome = {"ok": False, "tool": name, "result": None, "error": str(exc)}
        tool_calls.append(
            {
                "name": name,
                "arguments": args,
                "ok": bool(outcome.get("ok")),
                "error": outcome.get("error"),
            }
        )
        logger.info(
            "[tool_call] 工具 %s 执行%s",
            name,
            "成功" if outcome.get("ok") else f"失败：{outcome.get('error')}",
        )
        return outcome

    return executor


def save_conversation_turn(
    phone_number: str,
    query: str,
    reply: str,
    intent_code: int | None,
) -> None:
    """把一轮对话（用户问题 + 客服回答）写入 SQLite，失败只记日志。"""
    try:
        db.update_chat_history(phone_number, query, reply, intent_code)
        logger.info("对话记录已保存：phone=%s，intent=%s，回复长度=%d", phone_number, intent_code, len(reply))
    except Exception as exc:  # noqa: BLE001 - 落库失败不阻断响应
        logger.error("对话记录保存失败：%s，错误：%s", phone_number, exc)


def _public_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """裁剪工具调用记录，去掉大字段，便于放进 HTTP 响应。"""
    result = []
    for item in tool_calls:
        result.append(
            {
                "name": item.get("name"),
                "arguments": item.get("arguments") or {},
                "ok": bool(item.get("ok")),
                "error": item.get("error"),
            }
        )
    return result


def stream_chat(phone_number: str, query: str) -> Iterator[dict[str, Any]]:
    """主对话流程的流式版本：yield 事件字典，由 ``app.py`` 转成 SSE。

    事件类型：
    - ``{"type": "error", "message": str}``：参数或系统错误；
    - ``{"type": "start", ...}``：会话开始，包含手机号 / 意图 / 模式；
    - ``{"type": "intent", "intent_code", "intent_name", "reason", "mode"}``：意图识别结果；
    - ``{"type": "tool_call", "name", "arguments", "ok", "error"}``：工具调用；
    - ``{"type": "delta", "content": str}``：回答文本增量；
    - ``{"type": "done", "reply", "intent_code", "mode", "tool_calls", "user_info", ...}``：结束。

    Args:
        phone_number: 用户手机号。
        query: 用户问题。

    Yields:
        上述事件字典。
    """
    started_at = time.time()

    # ---------- 1. 参数校验 ----------
    try:
        phone = user_info_service.normalize_phone(phone_number)
        text = validate_query(query)
    except ValueError as exc:
        yield {"type": "error", "message": str(exc)}
        return

    logger.info("收到咨询：phone=%s，query=%s", phone, text[:100])

    # ---------- 2. 读取 / 创建用户 + 聊天历史 ----------
    try:
        user = user_info_service.get_or_create_user(phone)
    except Exception as exc:  # noqa: BLE001 - 数据库异常返回友好错误
        logger.exception("用户信息读取失败：%s", exc)
        yield {"type": "error", "message": "用户信息读取失败，请稍后重试"}
        return

    history: list[dict[str, Any]] = []
    try:
        history = db.fetch_chat_history(phone, limit=config.CHAT_HISTORY_LIMIT)
    except Exception as exc:  # noqa: BLE001
        logger.error("聊天历史读取失败，将按新会话处理：%s", exc)

    tool_calls: list[dict[str, Any]] = []

    # ---------- 3. 无历史：返回欢迎语并建立用户 ----------
    if not history:
        reply = prompts.render_welcome_message(user.get("name"))
        logger.info("新会话，返回欢迎语：phone=%s", phone)
        yield {
            "type": "start",
            "phone_number": phone,
            "is_new": True,
            "mode": "welcome",
            "intent_code": WELCOME_INTENT,
            "intent_name": prompts.INTENT_NAMES.get(WELCOME_INTENT, "其他/常规咨询"),
        }
        yield {"type": "delta", "content": reply}
        save_conversation_turn(phone, text, reply, None)
        latest_user = user_info_service.update_user_profile(phone, text, reply, user_info=user)
        yield {
            "type": "done",
            "reply": reply,
            "phone_number": phone,
            "is_new": True,
            "intent_code": WELCOME_INTENT,
            "intent_name": prompts.INTENT_NAMES.get(WELCOME_INTENT, "其他/常规咨询"),
            "mode": "welcome",
            "tool_calls": [],
            "user_info": latest_user,
            "elapsed_ms": int((time.time() - started_at) * 1000),
        }
        return

    # ---------- 4. 意图识别 ----------
    try:
        intent = detect_intent_with_reason(text, history=history)
    except Exception as exc:  # noqa: BLE001 - 意图识别异常按「其他」处理
        logger.error("意图识别异常，按其他意图处理：%s", exc)
        intent = {"intent_code": INTENT_OTHER, "reason": "意图识别异常，按其他咨询处理", "mode": "rule"}

    intent_code = int(intent.get("intent_code", INTENT_OTHER))
    intent_name = prompts.INTENT_NAMES.get(intent_code, "其他/常规咨询")
    logger.info("意图识别结果：phone=%s，intent=%s，mode=%s", phone, intent_code, intent.get("mode"))
    yield {
        "type": "start",
        "phone_number": phone,
        "is_new": False,
        "mode": intent.get("mode", "rule"),
        "intent_code": intent_code,
        "intent_name": intent_name,
    }
    yield {
        "type": "intent",
        "intent_code": intent_code,
        "intent_name": intent_name,
        "reason": intent.get("reason", ""),
        "mode": intent.get("mode", "rule"),
    }

    # ---------- 5. 按意图检索资料（Function Calling 工具） ----------
    evidence = rag.retrieve_evidence(
        intent_code=intent_code,
        query=text,
        context={"phone_number": phone},
        tool_calls=tool_calls,
    )
    for record in evidence.get("tool_calls") or []:
        yield {
            "type": "tool_call",
            "name": record.get("name"),
            "arguments": record.get("arguments") or {},
            "ok": bool(record.get("ok")),
            "error": record.get("error"),
        }

    context_block = rag.build_context_block(evidence)
    reply = ""
    mode = "rule"
    rounds = 0

    # ---------- 6. 生成回答：优先真实 LLM + function calling，失败降级模板 ----------
    if is_llm_available():
        try:
            messages = build_llm_messages(user, intent_code, history, text, context_block)
            executor = _make_streaming_executor(phone, tool_calls)
            reply_parts: list[str] = []
            for event in run_streaming_tool_loop(
                messages=messages,
                tools=get_openai_tools(),
                tool_executor=executor,
                max_rounds=config.LLM_MAX_TOOL_ROUNDS,
                temperature=config.LLM_TEMPERATURE,
            ):
                event_type = str(event.get("type") or "")
                if event_type == "tool_call":
                    yield {
                        "type": "tool_call",
                        "name": event.get("name"),
                        "arguments": event.get("arguments") or {},
                        "ok": True,
                        "error": None,
                    }
                elif event_type == "delta":
                    chunk = str(event.get("content") or "")
                    if chunk:
                        reply_parts.append(chunk)
                        yield {"type": "delta", "content": chunk}
                elif event_type == "final":
                    rounds = int(event.get("rounds") or 0)
                    if not reply_parts and event.get("content"):
                        reply_parts.append(str(event["content"]))
            reply = "".join(reply_parts).strip()
            if reply:
                mode = "llm"
            else:
                logger.warning("模型未返回有效文本，降级为离线模板回答")
        except (LLMUnavailableError, LLMResponseError) as exc:
            logger.warning("LLM 流式回答失败，降级为离线模板回答：%s", exc)
        except Exception as exc:  # noqa: BLE001 - 任何模型异常都降级
            logger.exception("LLM 流式回答异常，降级为离线模板回答：%s", exc)

    if not reply:
        mode = "rule"
        reply = build_fallback_reply(intent_code, text, user, evidence)
        yield {"type": "delta", "content": reply}

    # ---------- 7. 写回聊天记录 + 更新用户画像 ----------
    save_conversation_turn(phone, text, reply, intent_code)
    latest_user = user_info_service.update_user_profile(phone, text, reply, user_info=user)

    logger.info(
        "本轮结束：phone=%s，intent=%s，mode=%s，工具调用=%d 次，耗时=%dms",
        phone,
        intent_code,
        mode,
        len(tool_calls),
        int((time.time() - started_at) * 1000),
    )
    yield {
        "type": "done",
        "reply": reply,
        "phone_number": phone,
        "is_new": False,
        "intent_code": intent_code,
        "intent_name": intent_name,
        "mode": mode,
        "tool_calls": _public_tool_calls(tool_calls),
        "user_info": latest_user,
        "rounds": rounds,
        "elapsed_ms": int((time.time() - started_at) * 1000),
    }


def process_chat(phone_number: str, query: str) -> dict[str, Any]:
    """主对话流程的非流式版本（供 ``/chat`` 的 ``stream=false`` 与测试使用）。

    Args:
        phone_number: 用户手机号。
        query: 用户问题。

    Returns:
        ``{"ok": True, "reply", "intent_code", "intent_name", "mode",
        "tool_calls", "user_info", "is_new", "rounds", "elapsed_ms"}``；
        参数错误时返回 ``{"ok": False, "error": str}``。
    """
    reply = ""
    result: dict[str, Any] = {
        "ok": True,
        "phone_number": str(phone_number or ""),
        "reply": "",
        "intent_code": INTENT_OTHER,
        "intent_name": "其他/常规咨询",
        "mode": "rule",
        "tool_calls": [],
        "user_info": {},
        "is_new": False,
        "rounds": 0,
        "elapsed_ms": 0,
    }

    for event in stream_chat(phone_number, query):
        event_type = event.get("type")
        if event_type == "error":
            return {"ok": False, "error": event.get("message") or "处理失败"}
        if event_type == "start":
            result["is_new"] = bool(event.get("is_new"))
            result["mode"] = event.get("mode", result["mode"])
            result["intent_code"] = int(event.get("intent_code", INTENT_OTHER))
            result["intent_name"] = event.get("intent_name", result["intent_name"])
        elif event_type == "intent":
            result["intent_code"] = int(event.get("intent_code", INTENT_OTHER))
            result["intent_name"] = event.get("intent_name", result["intent_name"])
        elif event_type == "delta":
            reply += str(event.get("content") or "")
        elif event_type == "done":
            result.update(
                {
                    "reply": event.get("reply") or reply,
                    "intent_code": int(event.get("intent_code", result["intent_code"])),
                    "intent_name": event.get("intent_name", result["intent_name"]),
                    "mode": event.get("mode", result["mode"]),
                    "tool_calls": event.get("tool_calls") or [],
                    "user_info": event.get("user_info") or {},
                    "rounds": int(event.get("rounds") or 0),
                    "elapsed_ms": int(event.get("elapsed_ms") or 0),
                }
            )

    result["reply"] = reply or result["reply"]
    result["phone_number"] = str(phone_number or "")
    return result
