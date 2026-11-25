"""AI 工具层：把「意图识别 / 联网搜索 / 知识库检索 / SQL 查询 / 记录保存」暴露给大模型。

上层（services / database.sql_agent）通过 OpenAI function calling 把模型请求的
``name + arguments`` 交给 :func:`execute_tool`，由本模块统一调度、容错并返回
``{"ok": bool, "tool": str, "result": Any, "error": str | None}``。

依赖方向（避免循环导入）：
- ``tools`` 依赖 ``llm``（意图 / 搜索）与 ``database``、``vectorstore``；
- ``llm`` 的各子模块**不得**反向 import ``tools``；
- ``query_database`` 的 schema 直接复用 ``database.sql_agent.SQL_TOOL_SCHEMA``，
  ``detect_intent`` 的 schema 直接复用 ``llm.intent.DETECT_INTENT_TOOL_SCHEMA``，
  保证「模型看到的工具定义」和「实际执行逻辑」永远一致。

``execute_tool`` 永不抛异常：任何参数错误 / 依赖异常都会转成 ``ok=False`` 的结果，
让 function calling 循环可以继续，而不是把整个对话打断。
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any, Callable

from config import config, get_logger
from database import db
from database.sql_agent import SQL_TOOL_SCHEMA
from llm.intent import (
    DETECT_INTENT_TOOL_SCHEMA,
    VALID_INTENT_CODES,
    detect_intent_with_reason,
)
from llm.web_search import web_search as run_web_search

logger = get_logger("crm.tools.functions")


# ---------------------------------------------------------------------------
# 工具 schema（OpenAI tools 格式）
# ---------------------------------------------------------------------------

# web_search：实时联网信息
WEB_SEARCH_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "联网搜索互联网上的实时信息，适合回答行业最新趋势、岗位薪资行情、招聘动态等"
            "本地知识库可能过时的问题。query 传简洁的搜索关键词，不要传整段口语化长句。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，例如「AI 应用开发工程师 薪资 前景」",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果条数，默认 5，取值范围 1-10",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

# search_market_docs：职业市场知识库（向量检索）
SEARCH_MARKET_DOCS_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_market_docs",
        "description": (
            "检索本地「AI 职业市场」知识库，内容包括各 AI 岗位的职责、薪资区间、技能要求、"
            "行业趋势与转行路径。适合回答职业规划、岗位前景、就业方向、技能要求等问题。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索问题或关键词，例如「AI 应用开发需要什么技能」",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回文档条数，默认取配置 RAG_TOP_K（3），取值范围 1-10",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

# search_course_docs：课程知识库（向量检索）
SEARCH_COURSE_DOCS_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_course_docs",
        "description": (
            "检索本地「课程」知识库，内容包括课程内容、价格、时长、适合人群、学习路线与证书。"
            "适合回答学什么课程、课程多少钱、课程多久、是否适合零基础等问题。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索问题或关键词，例如「零基础学 Python 的课程」",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回文档条数，默认取配置 RAG_TOP_K（3），取值范围 1-10",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

# save_chat_record：保存聊天记录到 SQLite
SAVE_CHAT_RECORD_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "save_chat_record",
        "description": (
            "把一轮对话中的一条消息（用户提问或助手回答）保存到聊天记录表，"
            "便于后续查看历史与做用户画像。phone_number 不传时从对话上下文读取。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "用户手机号；不传则从 context 中的 phone_number 获取",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant"],
                    "description": "消息角色：user=用户，assistant=助手",
                },
                "content": {
                    "type": "string",
                    "description": "要保存的消息内容",
                },
                "intent_code": {
                    "type": "integer",
                    "enum": [1, 2, 0],
                    "description": "本轮意图编号（1=职业咨询，2=课程咨询，0=其他），可选",
                },
            },
            "required": ["role", "content"],
        },
    },
}

# 6 个工具的完整 schema 列表（顺序即 get_tool_names 的返回顺序）
TOOL_SCHEMAS: list[dict] = [
    DETECT_INTENT_TOOL_SCHEMA,
    WEB_SEARCH_TOOL_SCHEMA,
    SEARCH_MARKET_DOCS_TOOL_SCHEMA,
    SEARCH_COURSE_DOCS_TOOL_SCHEMA,
    SQL_TOOL_SCHEMA,
    SAVE_CHAT_RECORD_TOOL_SCHEMA,
]

# 工具名 → schema 的索引，供 get_openai_tools 过滤使用
_SCHEMA_BY_NAME: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}


# ---------------------------------------------------------------------------
# 参数与上下文小工具
# ---------------------------------------------------------------------------
def _get_text(arguments: Mapping[str, Any], context: Mapping[str, Any], *keys: str) -> str:
    """按顺序从 arguments / context 中取第一个非空字符串。

    Args:
        arguments: 模型传入的工具参数。
        context: 上层注入的对话上下文。
        *keys: 候选键名，按优先级排列。

    Returns:
        去空白后的文本；全部缺失时返回空串。
    """
    for source in (arguments, context):
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _coerce_top_k(arguments: Mapping[str, Any], default: int) -> int:
    """把 top_k 参数转换成 1~10 的整数，非法值回退默认值。"""
    raw = arguments.get("top_k")
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("非法 top_k=%r，使用默认值 %s", raw, default)
        return default
    return max(1, min(value, 10))


def _coerce_intent_code(value: Any) -> int | None:
    """把传入的 intent_code 转成合法编号（bool 不算），非法返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        code = value
    elif isinstance(value, float) and value.is_integer():
        code = int(value)
    elif isinstance(value, str):
        try:
            code = int(value.strip())
        except ValueError:
            return None
    else:
        return None
    return code if code in VALID_INTENT_CODES else None


def _normalize_arguments(arguments: dict | None) -> dict:
    """统一处理 arguments：None→{}、JSON 字符串→dict，非法时抛 ValueError。"""
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具参数不是合法 JSON：{exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        return parsed
    raise ValueError(f"工具参数类型非法：{type(arguments).__name__}")


# ---------------------------------------------------------------------------
# 各工具执行函数（内部使用：参数错误直接抛 ValueError，由 execute_tool 兜底）
# ---------------------------------------------------------------------------
def _tool_detect_intent(arguments: dict, context: dict) -> dict:
    """执行意图识别工具。

    两种用法：
    1. 模型已在 arguments 里给出 ``intent_code``（function calling 场景）→ 校验后原样采用；
    2. 只给了 ``query``（代码显式调用场景）→ 调用 detect_intent_with_reason 重新识别。

    Returns:
        ``{"intent_code": int, "reason": str}``
    """
    raw_code = arguments.get("intent_code")
    reason = str(arguments.get("reason") or "").strip()

    # 用法 1：模型/调用方直接给出编号
    if raw_code is not None:
        code = _coerce_intent_code(raw_code)
        if code is None:
            raise ValueError(
                f"非法 intent_code={raw_code!r}，只允许 1（职业咨询）/2（课程咨询）/0（其他）"
            )
        if not reason:
            reason = "由调用方给出的意图编号"
        logger.info("工具 detect_intent：采用外部传入编号 %s", code)
        return {"intent_code": code, "reason": reason}

    # 用法 2：只给 query → 走检测逻辑（LLM 优先，无 Key 自动降级关键词规则）
    query = _get_text(arguments, context, "query", "message", "question")
    if not query:
        raise ValueError("detect_intent 需要 intent_code，或提供 query 用于自动识别")
    result = detect_intent_with_reason(query, context.get("history"))
    return {
        "intent_code": int(result["intent_code"]),
        "reason": str(result["reason"]),
    }


def _tool_web_search(arguments: dict, context: dict) -> dict:
    """执行联网搜索工具（失败会降级本地职业知识库，不抛异常）。

    Returns:
        ``{"query": str, "results": list[dict], "count": int}``
    """
    query = _get_text(arguments, context, "query", "keyword", "q")
    if not query:
        raise ValueError("web_search 需要非空 query 参数")
    top_k = _coerce_top_k(arguments, int(config.WEB_SEARCH_TOP_K))
    results = run_web_search(query, top_k=top_k)
    results = results or []
    logger.info("工具 web_search：query=%r 返回 %s 条", query[:50], len(results))
    return {"query": query, "results": results, "count": len(results)}


def _search_documents(collection: str, arguments: dict, context: dict) -> dict:
    """检索指定向量集合的公共实现。

    Args:
        collection: 集合名（config.COLLECTION_MARKET / COLLECTION_COURSE）。
        arguments: 工具参数（query/top_k）。
        context: 对话上下文（可提供 query）。

    Returns:
        ``{"query": str, "collection": str, "results": list[dict], "count": int}``
    """
    query = _get_text(arguments, context, "query", "question", "keyword")
    if not query:
        raise ValueError(f"检索 {collection} 知识库需要非空 query 参数")
    top_k = _coerce_top_k(arguments, int(config.RAG_TOP_K))
    # 延迟导入：向量库依赖较重（chromadb / embedding 模型），且便于其它模块在
    # vectorstore 未初始化时也能 import tools；真正检索时才要求它可用。
    from vectorstore import store  # noqa: PLC0415 - 延迟导入避免导入期重依赖

    results = store.search(collection, query, top_k=top_k)
    results = results or []
    logger.info("工具检索知识库：collection=%s query=%r 返回 %s 条", collection, query[:50], len(results))
    return {"query": query, "collection": collection, "results": results, "count": len(results)}


def _tool_search_market_docs(arguments: dict, context: dict) -> dict:
    """执行「职业市场」知识库检索工具。"""
    return _search_documents(config.COLLECTION_MARKET, arguments, context)


def _tool_search_course_docs(arguments: dict, context: dict) -> dict:
    """执行「课程」知识库检索工具。"""
    return _search_documents(config.COLLECTION_COURSE, arguments, context)


def _tool_query_database(arguments: dict, context: dict) -> dict:
    """执行只读 SQL 查询工具（SQL 安全校验由 database.db.execute_sql 负责）。

    Returns:
        ``{"sql": str, "rows": list[dict], "row_count": int}``
    """
    sql = _get_text(arguments, context, "sql", "query", "statement")
    if not sql:
        raise ValueError("query_database 需要非空 sql 参数")
    rows = db.execute_sql(sql)
    rows = rows or []
    logger.info("工具 query_database：执行 SQL 返回 %s 行", len(rows))
    return {"sql": sql, "rows": rows, "row_count": len(rows)}


def _tool_save_chat_record(arguments: dict, context: dict) -> dict:
    """执行保存聊天记录工具。

    phone_number 优先取 arguments，其次 context（前端会话上下文中带手机号）；
    role 只允许 user / assistant（与 database.db.insert_message 的校验一致）。

    Returns:
        ``{"id": int, "saved": True}``
    """
    phone_number = _get_text(arguments, context, "phone_number", "phone", "user_phone")
    if not phone_number:
        raise ValueError("save_chat_record 缺少 phone_number（参数与 context 中都没有）")
    role = _get_text(arguments, context, "role").lower() or "assistant"
    if role not in ("user", "assistant"):
        raise ValueError(f"非法消息角色：{role!r}，只允许 user 或 assistant")
    content = _get_text(arguments, context, "content", "message", "text")
    if not content:
        raise ValueError("save_chat_record 缺少非空 content")

    intent_code = _coerce_intent_code(arguments.get("intent_code"))
    if intent_code is None:
        intent_code = _coerce_intent_code(context.get("intent_code"))

    message_id = db.insert_message(phone_number, role, content, intent_code=intent_code)
    logger.info("工具 save_chat_record：id=%s phone=%s role=%s", message_id, phone_number, role)
    return {"id": int(message_id), "saved": True}


# 工具名 → 执行函数映射（execute_tool 的唯一调度表）
_TOOL_HANDLERS: dict[str, Callable[[dict, dict], dict]] = {
    "detect_intent": _tool_detect_intent,
    "web_search": _tool_web_search,
    "search_market_docs": _tool_search_market_docs,
    "search_course_docs": _tool_search_course_docs,
    "query_database": _tool_query_database,
    "save_chat_record": _tool_save_chat_record,
}


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------
def get_tool_names() -> list[str]:
    """返回全部工具名（顺序与 TOOL_SCHEMAS 一致）。

    Returns:
        6 个工具名列表。
    """
    return [schema["function"]["name"] for schema in TOOL_SCHEMAS]


def get_openai_tools(names: list[str] | None = None) -> list[dict]:
    """返回可直接传给 ``client.chat.completions.create(tools=...)`` 的 schema 列表。

    Args:
        names: 需要保留的工具名；None / 空列表表示返回全部 6 个工具。

    Returns:
        深拷贝后的 schema 列表，调用方修改不会污染 TOOL_SCHEMAS。
    """
    if not names:
        return copy.deepcopy(TOOL_SCHEMAS)
    wanted = {str(name) for name in names if str(name)}
    unknown = wanted - set(_SCHEMA_BY_NAME)
    if unknown:
        logger.warning("get_openai_tools 收到未知工具名，已忽略：%s", sorted(unknown))
    selected = [
        copy.deepcopy(schema)
        for schema in TOOL_SCHEMAS
        if schema["function"]["name"] in wanted
    ]
    return selected


def execute_tool(
    name: str,
    arguments: dict | None = None,
    context: dict | None = None,
) -> dict:
    """执行一个本地工具，永不抛异常。

    Args:
        name: 工具名（detect_intent / web_search / search_market_docs /
            search_course_docs / query_database / save_chat_record）。
        arguments: 工具参数，dict 或 JSON 字符串；None 视为空参数。
        context: 对话上下文，可带 phone_number / query / history / intent_code 等。

    Returns:
        ``{"ok": bool, "tool": str, "result": Any, "error": str | None}``；
        参数非法、未知工具、依赖异常都会返回 ``ok=False`` 并附带 error 文本。
    """
    tool_name = str(name or "").strip()
    if not tool_name:
        return {"ok": False, "tool": "", "result": None, "error": "工具名不能为空"}

    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        logger.warning("execute_tool 收到未知工具：%r", tool_name)
        return {
            "ok": False,
            "tool": tool_name,
            "result": None,
            "error": f"未知工具：{tool_name}（可用：{', '.join(get_tool_names())}）",
        }

    try:
        normalized_arguments = _normalize_arguments(arguments)
        normalized_context = dict(context) if isinstance(context, Mapping) else {}
        result = handler(normalized_arguments, normalized_context)
        return {"ok": True, "tool": tool_name, "result": result, "error": None}
    except ValueError as exc:
        # 参数类错误：属于可预期的调用问题，记录 warning 即可
        logger.warning("工具 %s 参数错误：%s", tool_name, exc)
        return {"ok": False, "tool": tool_name, "result": None, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - 工具执行失败必须转成结果，不能打断对话
        logger.error("工具 %s 执行异常：%s", tool_name, exc, exc_info=True)
        return {
            "ok": False,
            "tool": tool_name,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
