"""SQL Agent：自然语言 → SQL → 执行 → 中文总结（database/sql_agent.py）。

职责：
- 提供 OpenAI function calling 工具 schema ``SQL_TOOL_SCHEMA``（供 LLM 主流程与 query_database 工具复用）；
- ``answer_with_sql_agent``：优先走 LLM tool calling，LLM 不可用/异常时自动降级为规则引擎；
- ``rule_based_answer``：无 API Key 也能覆盖课程列表、课程价格、薪资阈值、职业列表、用户统计等验收问题；
- ``generate_sql``：让模型直接输出一条只读 SQL（非 tool 调用路径）。

设计原因：
- 当前环境没有 ``DEEPSEEK_API_KEY``，因此规则兜底是**可验收的主路径**，而不是可选优化；
- LLM 模块由其它 agent 并行开发，本模块对 ``llm.client`` / ``llm.prompts`` 采用**函数内延迟导入**，
  即使它们尚未就绪也不会导致 ``import database`` 失败；
- 任何路径都不向 Flask 抛异常，统一返回
  ``{"ok", "answer", "sql", "rows", "mode", "error"}`` 六字段结构。
"""

from __future__ import annotations

import json
import re
from typing import Any

from config import config, get_logger
from database.db import (
    FORBIDDEN_SQL_MESSAGE,
    execute_sql,
    get_schema_description,
    is_readonly_sql,
)

logger = get_logger("crm.sql_agent")

# ---------------------------------------------------------------------------
# OpenAI function calling 工具 schema（契约冻结，禁止改动结构）
# ---------------------------------------------------------------------------
SQL_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": (
            "在业务 SQLite 数据库中执行一条只读 SELECT 查询，"
            "用于查询课程列表/价格、职业信息/薪资区间、用户画像与聊天记录统计等结构化数据。"
            "只允许单条 SELECT 或 WITH 语句，禁止任何写操作。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "只读 SELECT 语句（SQLite 语法，建议带 LIMIT 限制行数）",
                }
            },
            "required": ["sql"],
        },
    },
}

# ---------------------------------------------------------------------------
# 规则兜底关键词
# ---------------------------------------------------------------------------
_CAREER_KEYWORDS: tuple[str, ...] = (
    "职业",
    "岗位",
    "就业",
    "薪资",
    "工资",
    "月薪",
    "年薪",
    "薪酬",
    "待遇",
    "招聘",
    "行业",
    "前景",
    "转行",
    "工作",
)
_COURSE_KEYWORDS: tuple[str, ...] = (
    "课程",
    "培训",
    "学费",
    "上课",
    "学习",
    "报名",
    "试听",
    "证书",
)
_PRICE_KEYWORDS: tuple[str, ...] = (
    "多少钱",
    "价格",
    "费用",
    "学费",
    "收费",
    "报价",
    "贵",
    "便宜",
)
_COUNT_KEYWORDS: tuple[str, ...] = ("多少", "几个", "几门", "几类", "数量", "统计", "总数", "一共", "共有")
_USER_KEYWORDS: tuple[str, ...] = ("用户", "学员", "客户", "会员")
_MESSAGE_KEYWORDS: tuple[str, ...] = ("消息", "聊天记录", "对话")

# 课程主题 → SQL WHERE 条件（关键词均来自规则表，不直接拼接用户原文，避免注入）
_COURSE_TOPIC_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("python", "全栈"),
        "name LIKE '%Python%' OR description LIKE '%Python%'",
    ),
    (
        ("数据分析",),
        "name LIKE '%数据分析%' OR category LIKE '%数据%'",
    ),
    (
        ("ai", "人工智能", "大模型", "机器学习", "算法"),
        "name LIKE '%AI%' OR name LIKE '%大模型%' OR name LIKE '%算法%' "
        "OR description LIKE '%AI%' OR description LIKE '%大模型%'",
    ),
    (
        ("前端", "vue"),
        "name LIKE '%前端%' OR name LIKE '%Vue%'",
    ),
    (
        ("办公", "excel", "自动化"),
        "name LIKE '%办公%' OR description LIKE '%Excel%'",
    ),
)

_CN_NUMBERS: dict[str, float] = {
    "一": 1.0,
    "二": 2.0,
    "两": 2.0,
    "三": 3.0,
    "四": 4.0,
    "五": 5.0,
    "六": 6.0,
    "七": 7.0,
    "八": 8.0,
    "九": 9.0,
    "十": 10.0,
}

# llm.prompts 不可用时使用的内置 SQL Agent 系统提示词（含表结构与安全规则）
_FALLBACK_SQL_AGENT_SYSTEM_PROMPT: str = """你是专业的 SQLite 只读查询助手，负责把用户的自然语言问题转换成 SQL 查询。

## 数据库表结构
{schema}

## 工作规则
1. 只允许 SELECT 查询，禁止 INSERT / UPDATE / DELETE / DROP / ALTER 等任何写操作；
2. 表名、字段名必须与上面的结构完全一致；
3. 生成 SQL 后必须调用 query_database 工具执行，不要凭空编造查询结果；
4. 拿到查询结果后，用简洁的中文总结（必要时用列表），并说明数据来源表；
5. 用户问题与数据表无关时，直接说明「数据库中暂无相关数据」，不要调用工具；
6. SQL 使用 SQLite 语法；需要限制行数时使用 LIMIT {max_rows}。
"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def _result(
    ok: bool,
    answer: str,
    sql: list[str] | None = None,
    rows: list[dict] | None = None,
    mode: str = "rule",
    error: str | None = None,
) -> dict:
    """构造统一的 SQL Agent 返回结构（六个字段，顺序固定）。

    参数：
        ok: 是否成功得到可用答案。
        answer: 给用户看的中文回答。
        sql: 实际执行的 SQL 列表。
        rows: 查询结果行列表。
        mode: 回答来源，``llm`` / ``rule`` / ``error``。
        error: 错误说明，成功时为 ``None``。

    返回值：
        契约要求的六字段 dict。
    """
    return {
        "ok": bool(ok),
        "answer": str(answer or ""),
        "sql": list(sql or []),
        "rows": list(rows or []),
        "mode": mode,
        "error": error,
    }


def _max_rows() -> int:
    """读取配置中的 SQL 最大返回行数，非法配置回退为 100。

    参数：
        无。

    返回值：
        大于 0 的 int。
    """
    try:
        value = int(config.SQL_MAX_ROWS)
    except (TypeError, ValueError):
        value = 100
    return max(1, value)


def _safe_execute(sql: str) -> list[dict]:
    """执行固定规则 SQL，异常时记录日志并返回空列表。

    参数：
        sql: 规则引擎生成的只读 SQL。

    返回值：
        查询结果行列表；任何异常都返回 ``[]``（规则兜底不允许抛异常）。
    """
    try:
        return execute_sql(sql)
    except ValueError as exc:
        logger.error("规则 SQL 未通过只读校验（不应该发生）：%s | %s", exc, sql[:200])
        return []
    except Exception as exc:  # pragma: no cover - 兜底
        logger.warning("规则 SQL 执行失败：%s | %s", exc, sql[:200])
        return []


def _history_to_messages(history: list[dict] | None, limit: int = 4) -> list[dict]:
    """把数据库聊天历史转换成 OpenAI messages（只保留最近若干条）。

    参数：
        history: ``database.db.fetch_chat_history`` 的返回值。
        limit: 最多保留的条数。

    返回值：
        ``[{"role": "user"/"assistant", "content": ...}]``；非法输入返回空列表。
    """
    if not history:
        return []
    messages: list[dict] = []
    for item in list(history)[-limit:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def _format_course_rows(rows: list[dict]) -> list[str]:
    """把课程查询结果格式化为中文列表行。

    参数：
        rows: 含 name/category/price/duration 的行列表。

    返回值：
        形如 ``1. 课程名（分类）：12800 元，24 周`` 的列表。
    """
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        name = row.get("name") or "未命名课程"
        category = row.get("category") or "未分类"
        price = row.get("price")
        if isinstance(price, (int, float)):
            price_text = f"{float(price):.0f} 元"
        else:
            price_text = f"{price} 元"
        duration = row.get("duration") or "周期待定"
        lines.append(f"{index}. {name}（{category}）：{price_text}，{duration}")
    return lines


def _format_career_rows(rows: list[dict]) -> list[str]:
    """把职业查询结果格式化为中文列表行。

    参数：
        rows: 含 title/industry/salary_min/salary_max 的行列表。

    返回值：
        形如 ``1. AI 应用开发工程师（互联网/人工智能）：月薪 15000-28000 元`` 的列表。
    """
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        title = row.get("title") or "未命名岗位"
        industry = row.get("industry") or "行业未知"
        salary_min = row.get("salary_min")
        salary_max = row.get("salary_max")
        lines.append(f"{index}. {title}（{industry}）：月薪 {salary_min}-{salary_max} 元")
    return lines


def _summarize_rows(rows: list[dict], limit: int = 10) -> str:
    """把查询结果行压缩成简洁中文摘要（LLM 未给出总结时兜底使用）。

    参数：
        rows: 查询结果行列表。
        limit: 最多展示的行数。

    返回值：
        中文摘要文本。
    """
    if not rows:
        return "查询已执行，但数据库中暂无匹配的数据。"
    lines: list[str] = []
    for index, row in enumerate(rows[:limit], start=1):
        if isinstance(row, dict):
            parts = [f"{key}={value}" for key, value in row.items() if value is not None]
            lines.append(f"{index}. " + "；".join(parts))
        else:
            lines.append(f"{index}. {row}")
    suffix = "" if len(rows) <= limit else f"\n（共 {len(rows)} 条，仅展示前 {limit} 条）"
    return "查询结果：\n" + "\n".join(lines) + suffix


# ---------------------------------------------------------------------------
# 规则兜底：问题解析
# ---------------------------------------------------------------------------
def _extract_salary_threshold(question: str) -> int | None:
    """从问题中解析薪资阈值（单位：元/月）。

    参数：
        question: 用户问题，例如「薪资 2 万以上的职业有哪些」。

    返回值：
        解析出的整数阈值（``2万`` → 20000，``8k`` → 8000）；无法解析返回 ``None``。

    说明：
        只有问题里出现薪资类关键词时才解析，避免把课程时长等数字误判为薪资。
    """
    if not any(keyword in question for keyword in _CAREER_KEYWORDS):
        return None

    # 1) 中文数字 + 万（"两万"、"三万"）
    for match in re.finditer(r"([一两二三四五六七八九十])\s*万", question):
        return int(_CN_NUMBERS[match.group(1)] * 10000)

    # 2) 阿拉伯数字 + 单位（万 / w / k / 千 / 元）
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(万|w|W|k|K|千|元)", question):
        number = float(match.group(1))
        unit = match.group(2)
        if unit in {"万", "w", "W"}:
            return int(number * 10000)
        if unit in {"k", "K", "千"}:
            return int(number * 1000)
        return int(number)

    # 3) 裸数字：只有 4 位及以上才视为金额（如 "薪资 20000"）
    for match in re.finditer(r"(?<![\d.])(\d{4,})(?![\d.])", question):
        return int(match.group(1))

    # 4) "月薪过万 / 薪资上万" 这类口语表达
    if "过万" in question or "上万" in question:
        return 10000
    return None


def _course_topic_condition(question: str) -> str | None:
    """根据问题里的课程主题关键词返回 SQL WHERE 条件。

    参数：
        question: 用户问题。

    返回值：
        形如 ``name LIKE '%Python%'`` 的条件字符串；未识别出主题返回 ``None``。
    """
    lowered = question.lower()
    for keywords, condition in _COURSE_TOPIC_RULES:
        if any(keyword in lowered for keyword in keywords):
            return condition
    return None


# ---------------------------------------------------------------------------
# 规则兜底：各类型回答
# ---------------------------------------------------------------------------
def _rule_salary_query(threshold: int, max_rows: int) -> dict:
    """按薪资上限阈值查询职业（「薪资 2 万以上的职业有哪些」）。

    参数：
        threshold: 薪资阈值（元/月），按 ``salary_max >= threshold`` 过滤。
        max_rows: 最大返回行数。

    返回值：
        六字段结果 dict，``mode="rule"``。
    """
    sql = (
        "SELECT title, industry, salary_min, salary_max FROM career_info "
        f"WHERE salary_max >= {int(threshold)} ORDER BY salary_max DESC LIMIT {int(max_rows)}"
    )
    rows = _safe_execute(sql)
    if not rows:
        answer = f"数据库中暂未找到薪资上限达到 {threshold} 元的职业记录。"
    else:
        lines = _format_career_rows(rows)
        answer = (
            f"薪资上限达到 {threshold} 元/月的职业共有 {len(rows)} 个：\n"
            + "\n".join(lines)
            + "\n\n以上数据来自 career_info 表（薪资为月薪区间，供职业方向参考）。"
        )
    return _result(True, answer, [sql], rows)


def _rule_course_price(question: str, max_rows: int) -> dict:
    """回答课程价格类问题（课程多少钱 / 最贵的课程 / 最便宜的课程）。

    参数：
        question: 用户问题。
        max_rows: 最大返回行数。

    返回值：
        六字段结果 dict，``mode="rule"``。
    """
    if "最贵" in question or "最高" in question:
        sql = (
            "SELECT name, category, price, duration FROM courses "
            f"ORDER BY price DESC LIMIT {int(max_rows)}"
        )
        rows = _safe_execute(sql)
        if rows:
            top = rows[0]
            answer = (
                f"目前最贵的课程是「{top.get('name')}」，价格 {float(top.get('price') or 0):.0f} 元，"
                f"学习周期 {top.get('duration') or '待定'}。\n课程价格排行（从高到低）：\n"
                + "\n".join(_format_course_rows(rows))
            )
        else:
            answer = "数据库中暂时没有课程价格信息。"
        return _result(True, answer, [sql], rows)

    if "最便宜" in question or "最低" in question or "便宜" in question:
        sql = (
            "SELECT name, category, price, duration FROM courses "
            f"ORDER BY price ASC LIMIT {int(max_rows)}"
        )
        rows = _safe_execute(sql)
        if rows:
            cheap = rows[0]
            answer = (
                f"目前最便宜的课程是「{cheap.get('name')}」，价格 {float(cheap.get('price') or 0):.0f} 元，"
                f"学习周期 {cheap.get('duration') or '待定'}。\n课程价格排行（从低到高）：\n"
                + "\n".join(_format_course_rows(rows))
            )
        else:
            answer = "数据库中暂时没有课程价格信息。"
        return _result(True, answer, [sql], rows)

    # 普通价格询问：若带方向关键词（Python / AI / 数据分析…）则按主题过滤
    topic_condition = _course_topic_condition(question)
    where_clause = f"WHERE ({topic_condition}) " if topic_condition else ""
    sql = (
        "SELECT name, category, price, duration FROM courses "
        f"{where_clause}ORDER BY price ASC LIMIT {int(max_rows)}"
    )
    rows = _safe_execute(sql)
    if rows:
        prefix = "相关课程价格如下" if topic_condition else "课程价格如下"
        answer = f"{prefix}（共 {len(rows)} 门，按价格从低到高）：\n" + "\n".join(
            _format_course_rows(rows)
        )
    else:
        answer = "数据库中暂时没有符合条件的课程价格信息。"
    return _result(True, answer, [sql], rows)


def _rule_course_list(question: str, max_rows: int) -> dict:
    """回答课程列表类问题（有哪些课程 / Python 课程有哪些）。

    参数：
        question: 用户问题。
        max_rows: 最大返回行数。

    返回值：
        六字段结果 dict，``mode="rule"``。
    """
    topic_condition = _course_topic_condition(question)
    where_clause = f"WHERE ({topic_condition}) " if topic_condition else ""
    sql = (
        "SELECT name, category, price, duration FROM courses "
        f"{where_clause}ORDER BY price ASC LIMIT {int(max_rows)}"
    )
    rows = _safe_execute(sql)
    if rows:
        prefix = "找到相关课程" if topic_condition else "当前共有课程"
        answer = (
            f"{prefix} {len(rows)} 门：\n"
            + "\n".join(_format_course_rows(rows))
            + "\n\n如需了解某门课程的详细大纲、适合人群与报名方式，可以继续问我。"
        )
    else:
        answer = "数据库中暂时没有符合条件的课程。"
    return _result(True, answer, [sql], rows)


def _rule_career_list(max_rows: int) -> dict:
    """回答职业列表类问题（有哪些职业 / 哪个岗位薪资高）。

    参数：
        max_rows: 最大返回行数。

    返回值：
        六字段结果 dict，``mode="rule"``。
    """
    sql = (
        "SELECT title, industry, salary_min, salary_max FROM career_info "
        f"ORDER BY salary_max DESC LIMIT {int(max_rows)}"
    )
    rows = _safe_execute(sql)
    if rows:
        answer = (
            f"当前数据库收录了 {len(rows)} 个 AI / Python 相关职业：\n"
            + "\n".join(_format_career_rows(rows))
            + "\n\n以上数据来自 career_info 表（薪资为月薪区间，按薪资上限从高到低排列）。"
        )
    else:
        answer = "数据库中暂时没有职业信息。"
    return _result(True, answer, [sql], rows)


def _rule_user_count() -> dict:
    """回答用户数量统计问题。

    参数：
        无。

    返回值：
        六字段结果 dict，``mode="rule"``。
    """
    sql = "SELECT COUNT(*) AS user_count FROM users"
    rows = _safe_execute(sql)
    total = int(rows[0].get("user_count") or 0) if rows else 0
    answer = f"数据库当前共有 {total} 位用户（学员）画像记录。"
    return _result(True, answer, [sql], rows)


def _rule_message_count() -> dict:
    """回答聊天消息数量统计问题。

    参数：
        无。

    返回值：
        六字段结果 dict，``mode="rule"``。
    """
    sql = "SELECT COUNT(*) AS message_count FROM chat_history"
    rows = _safe_execute(sql)
    total = int(rows[0].get("message_count") or 0) if rows else 0
    answer = f"数据库当前共保存 {total} 条聊天消息。"
    return _result(True, answer, [sql], rows)


def _rule_course_count() -> dict:
    """回答课程数量统计问题。

    参数：
        无。

    返回值：
        六字段结果 dict，``mode="rule"``。
    """
    sql = "SELECT COUNT(*) AS course_count FROM courses"
    rows = _safe_execute(sql)
    total = int(rows[0].get("course_count") or 0) if rows else 0
    answer = f"数据库当前共有 {total} 门课程。"
    return _result(True, answer, [sql], rows)


def _rule_fallback(question: str) -> dict:
    """无法映射到结构化查询时的友好兜底回答。

    参数：
        question: 用户原始问题。

    返回值：
        六字段结果 dict，``ok=False``、``mode="rule"``。
    """
    answer = (
        f"抱歉，我暂时无法把「{question}」转换成数据库查询。\n"
        "我目前可以查询以下结构化数据：\n"
        "1. 课程信息：如「有哪些课程」「Python 课程多少钱」「最贵的课程」；\n"
        "2. 职业与薪资：如「薪资 2 万以上的职业有哪些」「有哪些职业」；\n"
        "3. 数据统计：如「现在有多少用户」「有多少条聊天记录」。\n"
        "您可以换个更具体的问法，或者直接咨询 AI 职业规划与课程顾问。"
    )
    return _result(False, answer, error="无法将问题映射为数据库查询")


def rule_based_answer(question: str) -> dict:
    """离线规则兜底：关键词 → 固定只读 SQL → 中文回答。

    参数：
        question: 用户自然语言问题。

    返回值：
        与 ``answer_with_sql_agent`` 相同的六字段 dict，``mode="rule"``：
        覆盖课程列表/价格/最贵最便宜、职业列表、薪资阈值、用户与聊天统计；
        无法映射时返回 ``ok=False`` 的友好说明（不抛异常）。
    """
    question_text = str(question or "").strip()
    if not question_text:
        return _result(
            False,
            "请先输入您想查询的问题，例如「有哪些课程」「薪资 2 万以上的职业有哪些」。",
            error="问题为空",
        )

    max_rows = _max_rows()
    salary_threshold = _extract_salary_threshold(question_text)
    has_career = any(keyword in question_text for keyword in _CAREER_KEYWORDS)
    has_course = any(keyword in question_text for keyword in _COURSE_KEYWORDS)
    has_price = any(keyword in question_text for keyword in _PRICE_KEYWORDS)
    has_count = any(keyword in question_text for keyword in _COUNT_KEYWORDS)

    # 1) 薪资阈值类：优先级最高（"薪资 2 万以上的职业有哪些"）
    if has_career and salary_threshold is not None:
        return _rule_salary_query(salary_threshold, max_rows)

    # 2) 课程价格类：必须早于「课程数量」判断，避免 "课程多少钱" 被当成数量查询
    if has_course and has_price:
        return _rule_course_price(question_text, max_rows)

    # 3) 用户 / 聊天记录统计
    if has_count and any(keyword in question_text for keyword in _USER_KEYWORDS):
        return _rule_user_count()
    if has_count and any(keyword in question_text for keyword in _MESSAGE_KEYWORDS):
        return _rule_message_count()

    # 4) 课程数量类："有多少门课程"（"课程多少钱" 已在第 2 步处理）
    if has_course and has_count and "课" in question_text:
        return _rule_course_count()

    # 5) 课程列表类（含主题过滤）
    if has_course:
        return _rule_course_list(question_text, max_rows)

    # 6) 职业列表类
    if has_career:
        return _rule_career_list(max_rows)

    # 7) 无法映射
    logger.info("规则兜底无法映射问题：%s", question_text[:100])
    return _rule_fallback(question_text)


# ---------------------------------------------------------------------------
# LLM 路径
# ---------------------------------------------------------------------------
def _build_sql_agent_messages(
    question: str, history: list[dict] | None = None
) -> list[dict]:
    """构造 SQL Agent 的 messages（系统提示词注入表结构 + 最近对话 + 当前问题）。

    参数：
        question: 用户问题。
        history: 最近聊天历史（可选），用于理解上下文省略句。

    返回值：
        OpenAI messages 列表。

    说明：
        优先复用 ``llm.prompts.build_sql_agent_messages``；该模块不可用时使用内置提示词，
        保证本模块在并行开发期间始终可用。
    """
    schema = get_schema_description()
    max_rows = _max_rows()
    history_messages = _history_to_messages(history)

    try:
        from llm.prompts import build_sql_agent_messages, history_to_messages

        messages = build_sql_agent_messages(question, schema, max_rows)
        if history:
            try:
                history_messages = history_to_messages(history, limit=4)
            except Exception as exc:  # pragma: no cover - 兼容不同实现
                logger.debug("llm.prompts.history_to_messages 调用失败，使用内置转换：%s", exc)
        if history_messages:
            # 历史消息放在当前问题之前，帮助模型理解“那它多少钱”之类的省略句
            messages = messages[:-1] + history_messages[-4:] + messages[-1:]
        return messages
    except Exception as exc:
        logger.warning("llm.prompts 暂不可用，使用内置 SQL Agent 提示词：%s", exc)
        messages: list[dict] = [
            {
                "role": "system",
                "content": _FALLBACK_SQL_AGENT_SYSTEM_PROMPT.format(
                    schema=schema, max_rows=max_rows
                ),
            }
        ]
        messages.extend(history_messages[-4:])
        messages.append({"role": "user", "content": question})
        return messages


def _collect_tool_results(tool_calls: Any) -> tuple[list[str], list[dict], list[str]]:
    """从 ``run_tool_calling_loop`` 的 tool_calls 中提取 SQL、结果行与错误信息。

    参数：
        tool_calls: 循环返回的 tool_calls 列表（元素可能含 ``result``）。

    返回值：
        ``(sql_statements, rows, errors)`` 三元组；兼容 result 为 dict 或被 JSON 序列化为 str 的情况。
    """
    sql_statements: list[str] = []
    rows: list[dict] = []
    errors: list[str] = []

    if not isinstance(tool_calls, list):
        return sql_statements, rows, errors

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            sql_text = arguments.get("sql")
            if isinstance(sql_text, str) and sql_text.strip():
                sql_statements.append(sql_text.strip())

        call_result = call.get("result")
        # 某些实现可能把工具结果 JSON 序列化后再放回 messages，这里做兼容解析
        if isinstance(call_result, str):
            try:
                call_result = json.loads(call_result)
            except (TypeError, ValueError):
                call_result = None
        if not isinstance(call_result, dict):
            continue
        if call_result.get("error"):
            errors.append(str(call_result["error"]))
        call_rows = call_result.get("rows")
        if isinstance(call_rows, list):
            rows.extend(row for row in call_rows if isinstance(row, dict))
    return sql_statements, rows, errors


def answer_with_sql_agent(question: str, history: list[dict] | None = None) -> dict:
    """自然语言 → SQL → 执行 → 中文总结（LLM 优先，规则兜底）。

    参数：
        question: 用户自然语言问题。
        history: 最近聊天历史（可选），用于理解上下文。

    返回值：
        ``{"ok": bool, "answer": str, "sql": list[str], "rows": list[dict],
        "mode": "llm" | "rule" | "error", "error": str | None}``；
        任何情况下都不抛异常：LLM 不可用 / 调用异常时自动走 ``rule_based_answer``。
    """
    question_text = str(question or "").strip()
    if not question_text:
        return _result(False, "请先输入您想查询的问题。", error="问题为空")

    # 延迟导入：llm.client 由其它 agent 并行实现，未就绪时不能影响本模块被 import
    try:
        from llm.client import LLMUnavailableError, is_llm_available, run_tool_calling_loop
    except Exception as exc:
        logger.info("llm.client 暂不可用（%s），SQL Agent 使用规则兜底", exc)
        return rule_based_answer(question_text)

    try:
        llm_available = bool(is_llm_available())
    except Exception as exc:  # pragma: no cover - 防御性处理
        logger.warning("检查 LLM 可用性失败，降级为规则兜底：%s", exc)
        llm_available = False

    if not llm_available:
        logger.info("LLM 未配置或不可用，SQL Agent 降级为规则兜底")
        return rule_based_answer(question_text)

    executed_sql: list[str] = []

    def _tool_executor(name: str, arguments: dict | None = None) -> Any:
        """执行 LLM 请求的 query_database 工具（只读 SQL），异常转成错误结果。"""
        if name != "query_database":
            logger.warning("SQL Agent 收到未知工具调用：%s", name)
            return {"ok": False, "error": f"未知工具：{name}"}
        try:
            sql_text = str((arguments or {}).get("sql") or "").strip()
            if not sql_text:
                return {"ok": False, "error": "缺少 sql 参数"}
            query_rows = execute_sql(sql_text)
            executed_sql.append(sql_text)
            logger.info("SQL Agent 执行 SQL 成功，返回 %d 行：%s", len(query_rows), sql_text[:200])
            return {"sql": sql_text, "rows": query_rows, "row_count": len(query_rows)}
        except ValueError as exc:
            logger.warning("SQL Agent 拒绝执行非法 SQL：%s", exc)
            return {"ok": False, "error": str(exc) or FORBIDDEN_SQL_MESSAGE}
        except Exception as exc:  # pragma: no cover - 兜底，工具执行异常不中断循环
            logger.warning("SQL Agent 工具执行失败：%s", exc)
            return {"ok": False, "error": f"SQL 执行失败：{exc}"}

    try:
        messages = _build_sql_agent_messages(question_text, history)
        loop_result = run_tool_calling_loop(
            messages,
            [SQL_TOOL_SCHEMA],
            _tool_executor,
            max_rounds=config.LLM_MAX_TOOL_ROUNDS,
        )
        if not isinstance(loop_result, dict):
            raise TypeError(f"run_tool_calling_loop 返回值类型异常：{type(loop_result)!r}")

        sql_statements, rows, tool_errors = _collect_tool_results(loop_result.get("tool_calls"))
        if not sql_statements:
            sql_statements = executed_sql

        answer = str(loop_result.get("content") or "").strip()
        if not answer:
            # 模型没有给出文字总结时，用查询结果兜底生成摘要
            answer = _summarize_rows(rows)
        logger.info(
            "SQL Agent LLM 路径完成：sql=%d 条，rows=%d 行，tool_errors=%s",
            len(sql_statements),
            len(rows),
            tool_errors[:2],
        )
        return _result(True, answer, sql_statements, rows, mode="llm")
    except LLMUnavailableError as exc:
        logger.warning("LLM 不可用，SQL Agent 降级为规则兜底：%s", exc)
        return rule_based_answer(question_text)
    except Exception as exc:
        logger.warning("SQL Agent LLM 调用失败，降级规则兜底：%s", exc, exc_info=True)
        if config.LLM_OFFLINE_FALLBACK:
            return rule_based_answer(question_text)
        return _result(
            False,
            "抱歉，数据库查询服务暂时不可用，请稍后再试。",
            mode="error",
            error=str(exc),
        )


def _extract_sql_from_text(text: str) -> str | None:
    """从模型输出文本中提取第一条合法的只读 SQL。

    参数：
        text: 模型原始输出（可能带 Markdown 代码围栏或解释文字）。

    返回值：
        合法的单条只读 SQL；无法提取或不是只读查询返回 ``None``。
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:sql)?", " ", str(text), flags=re.IGNORECASE).strip()
    # 从第一个 SELECT / WITH 关键字开始截取
    match = re.search(r"(?is)\b(select|with)\b", cleaned)
    if match is None:
        return None
    candidate = cleaned[match.start():].strip()

    # 依次尝试：整段 → 截断到第一个分号之前（应对 "SELECT ...; 解释文字" 的情况）
    variants = [candidate]
    if ";" in candidate:
        variants.append(candidate.split(";", 1)[0])
    for variant in variants:
        statement = variant.strip()
        while statement.endswith(";"):
            statement = statement[:-1].strip()
        if statement and is_readonly_sql(statement):
            return statement
    return None


def generate_sql(question: str) -> str | None:
    """让 LLM 只输出一条 SQL（非 tool 调用路径）。

    参数：
        question: 用户自然语言问题。

    返回值：
        合法单条只读 SQL；LLM 不可用、调用失败、输出无法解析或不是只读查询时返回 ``None``。
    """
    question_text = str(question or "").strip()
    if not question_text:
        return None

    try:
        from llm.client import chat_completion, is_llm_available
    except Exception as exc:
        logger.info("llm.client 暂不可用，generate_sql 返回 None：%s", exc)
        return None

    try:
        if not is_llm_available():
            logger.info("LLM 未配置或不可用，generate_sql 返回 None")
            return None

        schema = get_schema_description()
        max_rows = _max_rows()
        system_prompt = (
            "你是 SQLite 只读 SQL 生成器。根据下面的数据库表结构，把用户问题转换成一条 SQL。\n"
            "要求：\n"
            "1. 只输出一条 SELECT 或 WITH 查询，不要输出任何解释，不要使用 Markdown 代码块；\n"
            "2. 只允许只读查询，禁止 INSERT / UPDATE / DELETE / DROP / ALTER 等写操作；\n"
            "3. 表名、字段名必须与表结构完全一致；\n"
            f"4. 需要限制行数时使用 LIMIT {max_rows}。\n\n"
            f"数据库表结构：\n{schema}"
        )
        response = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ],
            temperature=0.0,
        )
        if not isinstance(response, dict):
            logger.warning("generate_sql 收到非 dict 响应：%r", type(response))
            return None
        sql = _extract_sql_from_text(str(response.get("content") or ""))
        if sql is None:
            logger.warning("generate_sql 未能从模型输出中解析出合法只读 SQL")
            return None
        logger.info("generate_sql 生成 SQL：%s", sql[:200])
        return sql
    except Exception as exc:
        # 无 Key 的 LLMUnavailableError 与模型调用异常都在这里静默降级为 None
        logger.warning("generate_sql 调用 LLM 失败，返回 None：%s", exc)
        return None
