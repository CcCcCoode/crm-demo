"""LLM 客户端封装：OpenAI 兼容协议（DeepSeek）+ Function Calling 循环。

职责：
1. 懒加载并复用 ``openai.OpenAI`` 客户端（无 Key 时返回 None，绝不伪造模型输出）；
2. 提供非流式 / 流式两种补全接口；
3. 实现完整的 function calling 循环（对应开发提示词里的 5 步）：
   请求模型 → 解析 tool_calls → 执行本地工具 → 追加 role="tool" 消息 → 再次请求，
   直到模型给出最终文本；超过最大轮数时强制不带工具收尾；
4. 提供容错的 ``extract_json``，供画像抽取、SQL 规划等场景解析模型输出。

设计原因：
- 当前环境没有 DEEPSEEK_API_KEY，必须保证「无 Key 时抛 LLMUnavailableError」而不是
  返回假答案，让上层（意图 / SQL Agent / 画像抽取）能识别并降级到规则引擎；
- DeepSeek 兼容 OpenAI 协议，因此统一使用官方 ``openai`` SDK，模型名 / 温度 / 超时
  全部来自 ``config``，代码里不写死；
- 工具循环只负责「调度」不负责「业务」，具体工具由上层通过 ``tool_executor`` 注入，
  避免 llm 层反向依赖 tools 层导致循环导入。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterator

import openai

from config import config, get_logger

logger = get_logger("crm.llm.client")


# ---------------------------------------------------------------------------
# 异常定义
# ---------------------------------------------------------------------------
class LLMUnavailableError(RuntimeError):
    """LLM 不可用：未配置 API Key、客户端初始化失败等。

    上层捕获该异常后应降级到规则引擎（mode="rule"），而不是把错误直接抛给用户。
    """


class LLMResponseError(RuntimeError):
    """LLM 调用失败或返回结构非法（网络错误、空 choices、流中断等）。"""


# ---------------------------------------------------------------------------
# 客户端单例（懒加载）
# ---------------------------------------------------------------------------
_client: openai.OpenAI | None = None
_client_error: str | None = None


def _build_client() -> openai.OpenAI:
    """真正构造 OpenAI 客户端。

    单独抽成函数便于测试替换，同时集中处理异常转换。
    """
    return openai.OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        timeout=config.LLM_TIMEOUT,
    )


def get_client() -> openai.OpenAI | None:
    """获取（懒加载的）OpenAI 客户端单例。

    Returns:
        已初始化的 ``openai.OpenAI``；未配置 API Key 或初始化失败时返回 None。
    """
    global _client, _client_error
    if _client is not None:
        return _client
    if not config.llm_configured:
        # 只告警一次，避免每次请求刷屏；上层会走规则降级
        if _client_error is None:
            _client_error = "未配置 DEEPSEEK_API_KEY"
            logger.warning("未配置 DEEPSEEK_API_KEY，LLM 客户端不可用，将走规则降级")
        return None
    try:
        _client = _build_client()
        _client_error = None
        logger.info(
            "LLM 客户端初始化成功：model=%s base_url=%s timeout=%ss",
            config.DEEPSEEK_MODEL,
            config.DEEPSEEK_BASE_URL,
            config.LLM_TIMEOUT,
        )
        return _client
    except Exception as exc:  # 构造失败（Key 非法 / 网络代理配置异常等）
        _client_error = f"LLM 客户端初始化失败：{exc}"
        logger.warning(_client_error)
        return None


def reset_client() -> None:
    """重置客户端单例（测试用：模拟 Key 变化或重新初始化）。"""
    global _client, _client_error
    _client = None
    _client_error = None


def is_llm_available() -> bool:
    """LLM 是否可用（已配置 Key 且客户端能成功初始化）。"""
    return get_client() is not None


def get_llm_status() -> dict:
    """返回 LLM 状态摘要，供 /health、/stats 与前端展示。

    Returns:
        ``{"available": bool, "model": str, "base_url": str}``
    """
    return {
        "available": is_llm_available(),
        "model": config.DEEPSEEK_MODEL,
        "base_url": config.DEEPSEEK_BASE_URL,
    }


def _require_client() -> openai.OpenAI:
    """获取客户端；不可用时抛 ``LLMUnavailableError``。

    这是「不伪造模型输出」的统一出口：所有真正发起模型调用的函数都先经过这里。
    """
    client = get_client()
    if client is None:
        raise LLMUnavailableError(
            _client_error or "LLM 不可用：未配置 DEEPSEEK_API_KEY 或客户端初始化失败"
        )
    return client


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """同时兼容 dict 与 SDK 对象（pydantic 模型）的字段读取。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_arguments(raw: Any) -> dict:
    """把模型返回的 function.arguments 解析成 dict（容错 JSON 非法的情况）。

    Args:
        raw: 模型返回的 arguments，通常是 JSON 字符串；mock 服务也可能直接给 dict。

    Returns:
        解析后的参数字典；无法解析时返回空 dict（并记 warning，调用方继续执行工具）。
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        logger.warning("工具参数类型异常（%s），按空参数处理", type(raw).__name__)
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("工具参数 JSON 解析失败（将按空参数执行）：%s；原始参数=%r", exc, text[:300])
        return {}
    if isinstance(parsed, dict):
        return parsed
    logger.warning("工具参数不是 JSON 对象（%s），按空参数处理", type(parsed).__name__)
    return {}


def _json_dumps(value: Any) -> str:
    """安全序列化：不可序列化对象（如 sqlite3.Row）用 str 兜底。"""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _normalize_tool_calls(message: Any) -> list[dict]:
    """把 SDK 返回的 tool_calls 规范化成 ``[{"id","name","arguments"}]``。

    Args:
        message: ``chat.completions.create`` 返回的 message 对象。

    Returns:
        规范化后的工具调用列表；无工具调用返回 ``[]``。
    """
    raw_calls = _get(message, "tool_calls") or []
    normalized: list[dict] = []
    for index, call in enumerate(raw_calls):
        function = _get(call, "function") or {}
        name = _get(function, "name") or ""
        if not name:
            logger.warning("忽略缺少函数名的 tool_call：%r", call)
            continue
        call_id = _get(call, "id") or f"call_{index}"
        arguments = _parse_arguments(_get(function, "arguments"))
        normalized.append({"id": str(call_id), "name": str(name), "arguments": arguments})
    return normalized


def _assistant_message(content: str, tool_calls: list[dict]) -> dict:
    """把规范化后的工具调用还原成 OpenAI messages 中的 assistant 消息。"""
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": _json_dumps(call.get("arguments") or {}),
                },
            }
            for call in tool_calls
        ],
    }


def _execute_tool_call(
    tool_executor: Callable[[str, dict], Any],
    call: dict,
) -> Any:
    """执行一个工具调用，异常统一转成 ``{"ok": False, "error": ...}``。

    设计原因：工具内部可能访问数据库 / 网络，绝不能因为单个工具失败中断整轮对话，
    模型需要拿到错误信息后自行决定下一步（例如换一种查法或直接回答）。
    """
    name = call.get("name") or ""
    arguments = call.get("arguments") or {}
    try:
        result = tool_executor(name, arguments)
    except Exception as exc:  # 工具执行失败不致命，交给模型处理
        logger.warning("工具 %s 执行异常：%s", name, exc)
        result = {"ok": False, "error": f"工具执行失败：{exc}"}
    logger.info("工具调用完成：name=%s arguments=%s", name, _json_dumps(arguments)[:300])
    return result


def _safe_on_event(on_event: Callable[[dict], None] | None, event: dict) -> None:
    """安全触发事件回调：回调异常只记日志，不能影响主流程。"""
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception as exc:  # 回调由上层提供，异常不应破坏工具循环
        logger.warning("on_event 回调异常（已忽略）：%s", exc)


def _tool_result_message(call_id: str, name: str, result: Any) -> dict:
    """构造 role="tool" 的消息（content 必须是字符串）。"""
    content = result if isinstance(result, str) else _json_dumps(result)
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


# ---------------------------------------------------------------------------
# extract_json：容错解析模型输出的 JSON
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _iter_json_object_candidates(text: str) -> Iterator[str]:
    """按「花括号配对」从文本中切出可能的 JSON 对象子串（跳过字符串内的花括号）。"""
    length = len(text)
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, length):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def extract_json(text: str) -> dict | None:
    """从模型输出文本中提取第一个合法 JSON 对象。

    兼容三种常见形态：
    1. 纯 JSON 文本；2. Markdown 代码块（```json ... ```）；3. JSON 前后夹带解释文字。

    Args:
        text: 模型原始输出。

    Returns:
        解析出的 dict；无法解析或顶层不是对象时返回 None。
    """
    if not isinstance(text, str):
        return None
    cleaned = text.strip().lstrip("\ufeff")
    if not cleaned:
        return None

    # 候选 1：代码块内容；候选 2：整段文本
    candidates: list[str] = [match.group(1).strip() for match in _FENCE_RE.finditer(cleaned)]
    candidates.append(cleaned)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        # 整段解析失败时，再尝试从文本里切出花括号配对的对象
        for object_text in _iter_json_object_candidates(candidate):
            try:
                parsed = json.loads(object_text)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


# ---------------------------------------------------------------------------
# 非流式补全
# ---------------------------------------------------------------------------
def chat_completion(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """非流式调用对话模型（支持 function calling）。

    Args:
        messages: OpenAI 格式消息列表（含 system/user/assistant/tool）。
        tools: 工具 schema 列表；为空表示普通对话。
        tool_choice: 工具选择策略，如 ``{"type": "function", "function": {"name": "detect_intent"}}``。
        temperature: 采样温度，默认取 ``config.LLM_TEMPERATURE``。
        max_tokens: 最大生成 token，默认取 ``config.LLM_MAX_TOKENS``。

    Returns:
        ``{"role": "assistant", "content": str, "tool_calls": [{"id","name","arguments"}]}``

    Raises:
        LLMUnavailableError: 无 API Key / 客户端不可用。
        LLMResponseError: 调用失败或返回结构非法。
    """
    client = _require_client()
    kwargs: dict[str, Any] = {
        "model": config.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        # tool_choice 只有在提供 tools 时才能传，否则部分兼容服务会报错
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # 网络 / 鉴权 / 服务端错误统一转换
        raise LLMResponseError(f"模型调用失败：{exc}") from exc

    choices = _get(response, "choices") or []
    if not choices:
        raise LLMResponseError("模型返回为空（choices 为空）")
    message = _get(choices[0], "message")
    if message is None:
        raise LLMResponseError("模型返回缺少 message 字段")
    content = _get(message, "content") or ""
    tool_calls = _normalize_tool_calls(message)
    logger.debug(
        "chat_completion：content=%r tool_calls=%s", content[:100], [c["name"] for c in tool_calls]
    )
    return {"role": "assistant", "content": str(content), "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# 非流式 Function Calling 循环（提示词里的 5 步）
# ---------------------------------------------------------------------------
def run_tool_calling_loop(
    messages: list[dict],
    tools: list[dict],
    tool_executor: Callable[[str, dict], Any],
    max_rounds: int | None = None,
    temperature: float | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """完整的 function calling 循环（非流式）。

    流程：
    1. 携带工具 schema 请求模型；
    2. 模型返回 tool_calls → 逐个执行 ``tool_executor(name, arguments)``；
    3. 把工具结果以 ``role="tool"`` + ``tool_call_id`` 消息追加回对话；
    4. 重新请求模型，直到模型不再调用工具、给出最终文本；
    5. 超过 ``max_rounds`` 时强制一次「不带 tools」的请求收尾，避免死循环。

    Args:
        messages: 初始消息（system + 历史 + 用户本轮输入）。
        tools: 工具 schema 列表。
        tool_executor: 工具执行器，签名 ``(name, arguments) -> Any``；
            返回结果必须可 JSON 序列化（不可序列化时用 str 兜底）。
        max_rounds: 最大工具轮数，默认 ``config.LLM_MAX_TOOL_ROUNDS``。
        temperature: 采样温度。
        on_event: 可选事件回调，收到 ``{"type": "round"/"tool_call"/"final", ...}``。

    Returns:
        ``{"content": str, "tool_calls": [{"id","name","arguments","result"}],
        "rounds": int, "mode": "llm"}``

    Raises:
        LLMUnavailableError: 无 API Key / 客户端不可用（上层据此降级）。
        LLMResponseError: 模型调用失败。
    """
    client = _require_client()
    if not isinstance(messages, list) or not messages:
        raise LLMResponseError("messages 不能为空")
    if not callable(tool_executor):
        raise LLMResponseError("tool_executor 必须可调用")

    rounds_limit = config.LLM_MAX_TOOL_ROUNDS if max_rounds is None else max(1, int(max_rounds))
    working: list[dict] = [dict(message) for message in messages]
    executed: list[dict] = []
    rounds_used = 0

    for round_index in range(1, rounds_limit + 1):
        rounds_used = round_index
        _safe_on_event(on_event, {"type": "round", "round": round_index})
        response = chat_completion(
            working, tools=tools, temperature=temperature
        )
        tool_calls = response["tool_calls"]
        content = response["content"]

        if not tool_calls:
            # 模型给出最终回答，循环结束
            logger.info("function calling 结束：rounds=%s content_len=%s", rounds_used, len(content))
            result = {
                "content": content,
                "tool_calls": executed,
                "rounds": rounds_used,
                "mode": "llm",
            }
            _safe_on_event(on_event, {"type": "final", **result})
            return result

        # 追加 assistant 的 tool_calls 消息，再逐个执行并追加 tool 消息
        working.append(_assistant_message(content, tool_calls))
        for call in tool_calls:
            tool_result = _execute_tool_call(tool_executor, call)
            record = {
                "id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"],
                "result": tool_result,
            }
            executed.append(record)
            working.append(_tool_result_message(call["id"], call["name"], tool_result))
            _safe_on_event(on_event, {"type": "tool_call", **record})

    # 超过最大轮数：强制一次不带 tools 的请求收尾，保证一定有自然语言回答
    logger.warning("function calling 达到最大轮数 %s，强制生成最终回答", rounds_limit)
    rounds_used += 1
    _safe_on_event(on_event, {"type": "round", "round": rounds_used, "forced": True})
    final_response = chat_completion(working, tools=None, temperature=temperature)
    result = {
        "content": final_response["content"],
        "tool_calls": executed,
        "rounds": rounds_used,
        "mode": "llm",
    }
    _safe_on_event(on_event, {"type": "final", **result})
    return result


# ---------------------------------------------------------------------------
# 流式补全
# ---------------------------------------------------------------------------
def _open_stream(
    client: openai.OpenAI,
    messages: list[dict],
    tools: list[dict] | None,
    tool_choice: Any,
    temperature: float | None,
    max_tokens: int | None,
) -> Any:
    """发起一次 ``stream=True`` 的补全请求，返回可迭代的 chunk 流。"""
    kwargs: dict[str, Any] = {
        "model": config.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise LLMResponseError(f"流式模型调用失败：{exc}") from exc


def _close_stream(stream: Any) -> None:
    """关闭流（部分兼容服务不实现 close，忽略异常）。"""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - 关闭失败无关紧要
            logger.debug("关闭流式响应失败", exc_info=True)


def _stream_text(
    client: openai.OpenAI,
    messages: list[dict],
    temperature: float | None,
    max_tokens: int | None,
) -> Iterator[str]:
    """流式文本增量生成器（无工具场景）。"""
    stream = _open_stream(client, messages, None, None, temperature, max_tokens)
    try:
        for chunk in stream:
            choices = _get(chunk, "choices") or []
            if not choices:
                continue
            delta = _get(choices[0], "delta")
            text = _get(delta, "content") if delta is not None else None
            if text:
                yield str(text)
    except LLMResponseError:
        raise
    except Exception as exc:  # 流中断 / 网络抖动
        raise LLMResponseError(f"流式响应中断：{exc}") from exc
    finally:
        _close_stream(stream)


def stream_completion(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """流式调用对话模型，逐块返回文本增量。

    Args:
        messages: OpenAI 格式消息列表。
        temperature: 采样温度，默认取配置。
        max_tokens: 最大生成 token，默认取配置。

    Returns:
        文本增量迭代器（每次 yield 一个片段）。

    Raises:
        LLMUnavailableError: 无 API Key / 客户端不可用（**调用时立即抛出**）。
        LLMResponseError: 请求失败或流中断（迭代过程中抛出）。
    """
    client = _require_client()  # 立即校验，保证「开始阶段」就把不可用暴露给上层
    return _stream_text(client, messages, temperature, max_tokens)


def _finalize_stream_tool_calls(tool_calls: dict[int, dict]) -> list[dict]:
    """把按 index 拼接的 tool_calls 转成规范化列表（丢弃无名调用，解析参数 JSON）。"""
    normalized: list[dict] = []
    for index in sorted(tool_calls):
        entry = tool_calls[index]
        name = entry.get("name") or ""
        if not name:
            logger.warning("流式响应中出现缺少函数名的 tool_call（index=%s），已忽略", index)
            continue
        normalized.append(
            {
                "id": entry.get("id") or f"call_{index}",
                "name": name,
                "arguments": _parse_arguments(entry.get("arguments")),
            }
        )
    return normalized


def run_streaming_tool_loop(
    messages: list[dict],
    tools: list[dict] | None,
    tool_executor: Callable[[str, dict], Any],
    max_rounds: int | None = None,
    temperature: float | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> Iterator[dict]:
    """流式版 function calling 循环。

    依次 yield：
    - ``{"type": "tool_call", "name": str, "arguments": dict, "result": Any}``：每执行一个工具；
    - ``{"type": "delta", "content": str}``：最终回答的文本增量；
    - ``{"type": "final", "content": str, "tool_calls": list[dict], "rounds": int}``：结束事件。

    Args:
        messages: 初始消息列表。
        tools: 工具 schema 列表（可为 None，表示纯对话）。
        tool_executor: 工具执行器 ``(name, arguments) -> Any``。
        max_rounds: 最大工具轮数，默认配置值；超限后强制不带工具收尾。
        temperature: 采样温度。
        on_event: 可选事件回调（与 yield 的事件同构）。

    Returns:
        事件字典迭代器。

    Raises:
        LLMUnavailableError: 无 API Key / 客户端不可用（**调用时立即抛出**）。
        LLMResponseError: 请求失败或流中断。
    """
    client = _require_client()  # 立即校验：无 Key 时在开始阶段就抛给上层
    if not isinstance(messages, list) or not messages:
        raise LLMResponseError("messages 不能为空")
    if not callable(tool_executor):
        raise LLMResponseError("tool_executor 必须可调用")
    return _stream_tool_loop_impl(
        client, messages, tools, tool_executor, max_rounds, temperature, on_event
    )


def _stream_tool_loop_impl(
    client: openai.OpenAI,
    messages: list[dict],
    tools: list[dict] | None,
    tool_executor: Callable[[str, dict], Any],
    max_rounds: int | None,
    temperature: float | None,
    on_event: Callable[[dict], None] | None,
) -> Iterator[dict]:
    """流式工具循环的实际实现（纯生成器，便于外层立即抛出不可用异常）。"""
    rounds_limit = config.LLM_MAX_TOOL_ROUNDS if max_rounds is None else max(1, int(max_rounds))
    working: list[dict] = [dict(message) for message in messages]
    executed: list[dict] = []
    rounds_used = 0
    all_content: list[str] = []

    def emit(event: dict) -> dict:
        """触发回调并返回事件本身，便于「回调 + yield」复用。"""
        _safe_on_event(on_event, event)
        return event

    answer_mode = False  # 是否已进入「最终回答」阶段（不再带工具请求）
    while True:
        if rounds_used >= rounds_limit:
            # 达到最大轮数：强制一次不带 tools 的请求收尾
            logger.warning("流式 function calling 达到最大轮数 %s，强制生成最终回答", rounds_limit)
            answer_mode = True

        rounds_used += 1
        current_tools = None if answer_mode else tools
        stream = _open_stream(client, working, current_tools, None, temperature, None)
        pending: dict[int, dict] = {}  # 本轮 tool_calls 增量累加器：{index: {...}}
        try:
            for chunk in stream:
                choices = _get(chunk, "choices") or []
                if not choices:
                    continue
                delta = _get(choices[0], "delta")
                if delta is None:
                    continue
                text = _get(delta, "content")
                if text:
                    # 文本增量边收边 yield，保证真正的流式体验
                    all_content.append(str(text))
                    yield emit({"type": "delta", "content": str(text)})
                tool_delta = _get(delta, "tool_calls")
                if tool_delta:
                    _merge_tool_delta(pending, tool_delta)
        except LLMResponseError:
            raise
        except Exception as exc:
            raise LLMResponseError(f"流式响应中断：{exc}") from exc
        finally:
            _close_stream(stream)

        tool_calls = _finalize_stream_tool_calls(pending)

        if answer_mode or not tool_calls:
            # 没有工具调用（或已强制收尾）→ 这就是最终回答
            content = "".join(all_content)
            result = {
                "content": content,
                "tool_calls": executed,
                "rounds": rounds_used,
            }
            yield emit({"type": "final", **result})
            return

        # 有工具调用：补充 assistant 消息、执行工具、追加 tool 消息后继续下一轮
        working.append(_assistant_message("", tool_calls))
        for call in tool_calls:
            tool_result = _execute_tool_call(tool_executor, call)
            record = {
                "id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"],
                "result": tool_result,
            }
            executed.append(record)
            working.append(_tool_result_message(call["id"], call["name"], tool_result))
            yield emit({"type": "tool_call", **record})


def _merge_tool_delta(pending: dict[int, dict], tool_delta: list[Any]) -> None:
    """把单个 chunk 里的 tool_calls 增量按 index 合并进 pending。

    Args:
        pending: ``{index: {"id","name","arguments","type"}}`` 累加器。
        tool_delta: 当前 chunk 的 ``delta.tool_calls`` 列表。
    """
    for raw_call in tool_delta:
        index = _get(raw_call, "index", 0) or 0
        entry = pending.setdefault(
            int(index), {"id": "", "name": "", "arguments": "", "type": "function"}
        )
        call_id = _get(raw_call, "id")
        if call_id:
            entry["id"] = str(call_id)
        function = _get(raw_call, "function")
        if function is None:
            continue
        name = _get(function, "name")
        if name:
            entry["name"] += str(name)
        arguments = _get(function, "arguments")
        if arguments:
            entry["arguments"] += str(arguments)


# ---------------------------------------------------------------------------
# 简易对话封装
# ---------------------------------------------------------------------------
def simple_chat(messages: list[dict], **kwargs: Any) -> str:
    """单轮对话，直接返回文本内容（供画像抽取、SQL 生成等场景使用）。

    Args:
        messages: OpenAI 格式消息列表。
        **kwargs: 透传给 ``chat_completion``（temperature / max_tokens / tools / tool_choice）。

    Returns:
        模型回答文本（可能为空字符串）。

    Raises:
        LLMUnavailableError / LLMResponseError: 同 ``chat_completion``。
    """
    response = chat_completion(messages, **kwargs)
    return response.get("content", "")
