"""用户画像服务。

职责：
1. 校验手机号，读取或创建用户；
2. 优先用 LLM（Function Calling 之外的普通对话）从「用户提问 + 客服回答」中抽取画像增量，
   无 API Key 或调用失败时自动降级为关键词/正则规则抽取；
3. 把画像更新写回 SQLite，并返回最新画像，供下一轮提示词使用。

设计说明：画像字段只允许 name / intention_direction / budget / stage / notes，
与 ``database.db.ALLOWED_USER_FIELDS`` 保持一致，避免脏字段写库。
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from config import get_logger
from database import db
from llm import prompts
from llm.client import LLMUnavailableError, chat_completion, extract_json, is_llm_available

logger = get_logger("crm.services.user_info")

# 中国大陆手机号
PHONE_PATTERN = re.compile(r"^1[3-9]\d{9}$")
# 合法的咨询阶段
STAGE_VALUES = ("新咨询", "已了解", "已报名")
# 允许写库的画像字段
PROFILE_FIELDS = ("name", "intention_direction", "budget", "stage", "notes")

# 「我叫 X / 我是 X / 我的名字是 X」中的名字候选，需要过滤掉非姓名片段
NAME_STOPWORDS = {
    "来咨询",
    "想咨询",
    "咨询的",
    "学生",
    "新手",
    "小白",
    "零基础",
    "在职",
    "上班族",
    "一名",
    "应届生",
    "程序员",
    "开发者",
}

# 关注方向关键词（按优先级从高到低匹配）
INTENTION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AI 应用开发", ("ai 应用", "ai应用", "人工智能", "大模型", "llm", "算法")),
    ("数据分析", ("数据分析", "数据挖掘", "数据开发")),
    ("Python 后端开发", ("python", "后端", "服务端", "爬虫")),
    ("前端开发", ("前端", "web", "网页")),
)

# 已报名信号
ENROLLED_KEYWORDS = ("已报名", "已经报名", "已缴费", "已交钱", "已付款", "在读", "上课中", "已经学了")
# 已了解信号（问价格/详情/试听说明已经进入了解阶段）
UNDERSTOOD_KEYWORDS = (
    "多少钱",
    "价格",
    "费用",
    "学费",
    "课时",
    "周期",
    "试听",
    "课程详情",
    "课程大纲",
    "方案",
    "优惠",
)

BUDGET_PATTERN_WAN = re.compile(
    r"(?:预算|费用|学费|价格|花费|准备花|能花|最多|大概|大约)[^\d]{0,8}?(\d+(?:\.\d+)?)\s*(?:万|w)(?![a-zA-Z])",
    re.IGNORECASE,
)
BUDGET_PATTERN_YUAN = re.compile(
    r"(?:预算|费用|学费|价格|花费|准备花|能花|最多)[^\d]{0,8}?(\d{3,6})\s*(?:元|块|rmb|人民币)?",
    re.IGNORECASE,
)
# 用户没写「预算」但明确带金额单位时也可识别，例如「8000 元可以吗」
BUDGET_PATTERN_EXPLICIT = re.compile(r"(?<![\d.])(\d{3,6})\s*(?:元|块|rmb|人民币)", re.IGNORECASE)
NAME_PATTERN = re.compile(r"(?:我叫|我是|我的名字(?:是|叫)|称呼我)\s*([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z ]{0,9})")


def normalize_phone(phone_number: str | None) -> str:
    """校验并规范化手机号。

    Args:
        phone_number: 原始手机号，可能包含空格或短横线。

    Returns:
        11 位数字手机号。

    Raises:
        ValueError: 手机号为空或格式非法，附带中文提示语。
    """
    raw = str(phone_number or "").strip()
    cleaned = re.sub(r"[\s-]", "", raw)
    if not cleaned:
        raise ValueError("请提供手机号，格式示例：13800138000")
    if not PHONE_PATTERN.match(cleaned):
        raise ValueError("手机号格式不正确，请提供 11 位中国大陆手机号，例如：13800138000")
    return cleaned


def get_or_create_user(phone_number: str) -> dict:
    """按手机号读取用户画像；不存在则创建一条「新咨询」用户。

    Args:
        phone_number: 已规范化的手机号。

    Returns:
        用户画像 dict（含 created_at / updated_at）。
    """
    user_info = db.get_user_info(phone_number)
    if user_info:
        return user_info
    user_info = db.create_user(phone_number, stage="新咨询")
    logger.info("新用户已创建：%s", phone_number)
    return user_info or {}


def get_user_profile(phone_number: str) -> dict:
    """读取用户画像，不存在时返回空 dict（不触发创建）。"""
    return db.get_user_info(phone_number) or {}


def detect_budget(text: str) -> float | None:
    """从文本中解析预算金额（支持「1 万」「1.5w」「8000 元」等写法）。

    Args:
        text: 用户输入或回答文本。

    Returns:
        金额（float）；无法解析返回 None。
    """
    content = str(text or "")
    for pattern, multiplier in (
        (BUDGET_PATTERN_WAN, 10000.0),
        (BUDGET_PATTERN_YUAN, 1.0),
        (BUDGET_PATTERN_EXPLICIT, 1.0),
    ):
        match = pattern.search(content)
        if not match:
            continue
        try:
            amount = float(match.group(1)) * multiplier
        except (TypeError, ValueError):
            continue
        if 500 <= amount <= 1_000_000:
            return round(amount, 2)
    return None


def detect_name(text: str) -> str | None:
    """从「我叫张三」这类表达中解析姓名，过滤明显的非姓名片段。"""
    match = NAME_PATTERN.search(str(text or ""))
    if not match:
        return None
    name = match.group(1).strip()
    for stopword in NAME_STOPWORDS:
        if name.startswith(stopword):
            return None
    name = name[:10]
    if len(name) < 2:
        return None
    return name


def detect_intention_direction(text: str) -> str | None:
    """从文本中识别用户关注的技术方向（AI / 数据分析 / Python 后端 / 前端）。"""
    content = str(text or "").lower()
    for direction, keywords in INTENTION_KEYWORDS:
        if any(keyword in content for keyword in keywords):
            return direction
    return None


def detect_stage(text: str) -> str | None:
    """从文本中识别咨询阶段（已报名优先，其次已了解）。"""
    content = str(text or "")
    if any(keyword in content for keyword in ENROLLED_KEYWORDS):
        return "已报名"
    if any(keyword in content for keyword in UNDERSTOOD_KEYWORDS):
        return "已了解"
    return None


def extract_profile_by_rules(query: str, answer: str = "") -> dict[str, Any]:
    """无 LLM 时的规则画像抽取。

    只抽取用户明确表达的信息，不猜测：姓名、预算、关注方向、咨询阶段、关注点摘要。

    Args:
        query: 用户本轮提问。
        answer: 客服本轮回答（仅作为补充上下文，规则主要看用户输入）。

    Returns:
        待更新的画像字段 dict，没有信号时返回空 dict。
    """
    text = f"{query or ''}\n{answer or ''}"
    updates: dict[str, Any] = {}

    name = detect_name(query)
    if name:
        updates["name"] = name

    budget = detect_budget(query)
    if budget is not None:
        updates["budget"] = budget

    direction = detect_intention_direction(query)
    if direction:
        updates["intention_direction"] = direction

    stage = detect_stage(query)
    if stage:
        updates["stage"] = stage

    # 关注点摘要只取用户原话，最长 60 字，避免污染画像
    snippet = re.sub(r"\s+", " ", str(query or "")).strip()
    if snippet:
        updates["notes"] = snippet[:60]

    return updates


def _sanitize_updates(updates: Mapping[str, Any] | None) -> dict[str, Any]:
    """过滤 LLM 抽取结果：只保留合法字段并做类型/长度校验。"""
    result: dict[str, Any] = {}
    for field in PROFILE_FIELDS:
        if not updates or field not in updates:
            continue
        value = updates[field]
        if value is None or value == "":
            continue
        if field == "budget":
            try:
                amount = float(value)
            except (TypeError, ValueError):
                continue
            if 500 <= amount <= 1_000_000:
                result["budget"] = round(amount, 2)
        elif field == "stage":
            if str(value) in STAGE_VALUES:
                result["stage"] = str(value)
        elif field == "name":
            text = str(value).strip()
            if 1 <= len(text) <= 20:
                result["name"] = text
        else:
            text = str(value).strip()
            if 0 < len(text) <= 200:
                result[field] = text
    return result


def extract_profile_with_llm(user_info: Mapping[str, Any] | None, query: str, answer: str) -> dict[str, Any]:
    """用 LLM 从本轮对话中抽取画像增量。

    无 API Key / 模型异常 / JSON 解析失败时返回空 dict（由调用方降级规则抽取）。

    Args:
        user_info: 当前画像。
        query: 用户提问。
        answer: 客服回答。

    Returns:
        待更新字段 dict，可能为空。
    """
    if not is_llm_available():
        return {}
    try:
        messages = prompts.build_profile_extraction_messages(user_info, query, answer)
        response = chat_completion(messages, temperature=0.1, max_tokens=400)
    except (LLMUnavailableError, Exception) as exc:  # noqa: BLE001 - 任何 LLM 异常都降级
        logger.warning("LLM 画像抽取失败，将使用规则抽取：%s", exc)
        return {}

    payload = extract_json(str(response.get("content") or ""))
    if not isinstance(payload, dict):
        logger.warning("LLM 画像抽取结果不是合法 JSON，降级为规则抽取")
        return {}
    sanitized = _sanitize_updates(payload)
    if sanitized:
        logger.info("LLM 抽取到画像增量：%s", sanitized)
    return sanitized


def apply_profile_updates(phone_number: str, updates: Mapping[str, Any] | None) -> dict:
    """把画像增量写入数据库，返回最新画像。

    Args:
        phone_number: 手机号。
        updates: 待更新字段（已经过 :func:`_sanitize_updates` 过滤）。

    Returns:
        最新用户画像 dict；写库失败时返回原画像并记录日志，不抛异常。
    """
    if not updates:
        return get_user_profile(phone_number)
    try:
        latest = db.update_user_info(phone_number, **dict(updates))
        logger.info("用户画像已更新：%s -> %s", phone_number, list(updates.keys()))
        return latest or get_user_profile(phone_number)
    except Exception as exc:  # noqa: BLE001 - 画像更新失败不应影响本轮回答
        logger.error("用户画像更新失败：%s，错误：%s", phone_number, exc)
        return get_user_profile(phone_number)


def update_user_profile(
    phone_number: str,
    query: str,
    answer: str,
    user_info: Mapping[str, Any] | None = None,
) -> dict:
    """一轮对话结束后更新用户画像。

    先尝试 LLM 抽取；LLM 不可用、返回空或异常时用规则抽取；最终写回 SQLite。

    参数:
        phone_number: 手机号。
        query: 用户本轮提问。
        answer: 客服本轮回答。
        user_info: 更新前的画像（用于判断 notes 是否已有内容）。

    返回:
        更新后的最新画像 dict。
    """
    current = dict(user_info or get_user_profile(phone_number))
    updates = extract_profile_with_llm(current, query, answer)
    if not updates:
        updates = extract_profile_by_rules(query, answer)
        logger.info("使用规则抽取画像增量：%s", updates)

    # notes 只保留第一次记录的用户关注点，避免每轮被覆盖
    if updates.get("notes") and current.get("notes"):
        updates.pop("notes", None)
    # 姓名只允许从用户原话中设置，规则抽不到时保留已有姓名
    if updates.get("name") and current.get("name") and updates["name"] == current["name"]:
        updates.pop("name", None)

    return apply_profile_updates(phone_number, _sanitize_updates(updates))
