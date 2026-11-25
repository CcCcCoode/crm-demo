"""RAG 检索服务。

按意图调用 **Function Calling 工具** 获取回答素材：

- 意图 1（AI 职业咨询）：``web_search`` 网络搜索 + ``search_market_docs`` 市场向量库检索；
- 意图 2（课程咨询）：``search_course_docs`` 课程向量库检索；
- 意图 0（其他）：不检索。

之所以统一走 ``tools.functions.execute_tool``，是为了让「工具调用」在有无 API Key 的
两种模式下都真实发生，并把每次调用记录到 ``tool_calls``，便于日志与 ``/chat`` 响应展示。
"""

from __future__ import annotations

import json
from typing import Any

from config import config, get_logger
from llm import prompts
from tools.functions import execute_tool

logger = get_logger("crm.services.rag")

INTENT_CAREER = 1
INTENT_COURSE = 2
INTENT_OTHER = 0


def _run_tool(
    name: str,
    arguments: dict[str, Any],
    context: dict[str, Any] | None,
    tool_calls: list[dict[str, Any]],
    errors: list[str],
) -> Any:
    """执行一次工具调用并记录结果。

    Args:
        name: 工具名。
        arguments: 工具参数。
        context: 执行上下文（如当前手机号）。
        tool_calls: 调用记录列表（原地追加）。
        errors: 错误信息列表（原地追加）。

    Returns:
        工具执行结果对象；失败返回 None。
    """
    try:
        outcome = execute_tool(name, arguments, context=context)
    except Exception as exc:  # noqa: BLE001 - execute_tool 本身不抛异常，这里再兜一层
        logger.error("工具 %s 执行异常：%s", name, exc)
        errors.append(f"{name}: {exc}")
        return None

    ok = bool(outcome.get("ok"))
    record = {
        "name": name,
        "arguments": arguments,
        "ok": ok,
        "error": outcome.get("error"),
    }
    # 统一用 [tool_call] 前缀记录每一次 Function Calling（离线规则模式同样记录），
    # 便于验收时直接在日志里检索「工具调用是否真的发生」。
    try:
        arguments_text = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover - 参数理论上都可 JSON 序列化
        arguments_text = repr(arguments)
    if not ok:
        logger.warning("[tool_call] %s arguments=%s ok=False error=%s", name, arguments_text, outcome.get("error"))
        errors.append(f"{name}: {outcome.get('error')}")
    else:
        logger.info("[tool_call] %s arguments=%s ok=True", name, arguments_text)
    tool_calls.append(record)
    return outcome.get("result") if ok else None


def retrieve_evidence(
    intent_code: int,
    query: str,
    context: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """按意图检索回答素材。

    Args:
        intent_code: 意图编号 0/1/2。
        query: 用户提问。
        context: 工具执行上下文（例如 ``{"phone_number": "..."}``）。
        tool_calls: 外部传入的调用记录列表，便于与 LLM 工具调用合并。
        errors: 外部传入的错误列表。

    Returns:
        ``{"intent_code", "query", "web_results", "market_docs", "course_docs",
        "tool_calls", "errors"}``。
    """
    context = context or {}
    tool_calls = tool_calls if tool_calls is not None else []
    errors = errors if errors is not None else []

    web_results: list[dict[str, Any]] = []
    market_docs: list[dict[str, Any]] = []
    course_docs: list[dict[str, Any]] = []

    try:
        if int(intent_code) == INTENT_CAREER:
            web_payload = _run_tool(
                "web_search",
                {"query": query, "top_k": config.WEB_SEARCH_TOP_K},
                context,
                tool_calls,
                errors,
            )
            if isinstance(web_payload, dict):
                web_results = list(web_payload.get("results") or [])

            market_payload = _run_tool(
                "search_market_docs",
                {"query": query, "top_k": config.RAG_TOP_K},
                context,
                tool_calls,
                errors,
            )
            if isinstance(market_payload, dict):
                market_docs = list(market_payload.get("results") or [])
        elif int(intent_code) == INTENT_COURSE:
            course_payload = _run_tool(
                "search_course_docs",
                {"query": query, "top_k": config.RAG_TOP_K},
                context,
                tool_calls,
                errors,
            )
            if isinstance(course_payload, dict):
                course_docs = list(course_payload.get("results") or [])
    except Exception as exc:  # noqa: BLE001 - 检索失败降级为空资料，不阻断主流程
        logger.error("检索资料失败，将使用空资料回答：%s", exc)
        errors.append(f"retrieve: {exc}")

    logger.info(
        "检索完成：intent=%s，web=%d，market=%d，course=%d",
        intent_code,
        len(web_results),
        len(market_docs),
        len(course_docs),
    )
    return {
        "intent_code": int(intent_code),
        "query": query,
        "web_results": web_results,
        "market_docs": market_docs,
        "course_docs": course_docs,
        "tool_calls": tool_calls,
        "errors": errors,
    }


def build_context_block(evidence: dict[str, Any] | None) -> str:
    """把检索结果拼装成注入系统提示词的参考资料块。

    Args:
        evidence: :func:`retrieve_evidence` 的返回值。

    Returns:
        参考资料文本；无资料时返回空串（提示词会使用占位说明）。
    """
    if not evidence:
        return ""
    try:
        return prompts.build_evidence_block(
            web_results=evidence.get("web_results"),
            market_docs=evidence.get("market_docs"),
            course_docs=evidence.get("course_docs"),
        )
    except Exception as exc:  # noqa: BLE001 - 格式化失败不影响主流程
        logger.error("拼装参考资料失败：%s", exc)
        return ""
