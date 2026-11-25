"""意图识别：优先大模型 function calling，失败时降级为关键词加权规则。

职责：
1. 定义 ``detect_intent`` 工具 schema（供 tool_choice 强制调用，也被 tools 层复用）；
2. ``keyword_intent``：纯规则的关键词加权打分（无 Key / 模型异常时的兜底路径）；
3. ``detect_intent_with_reason``：先走 LLM，再降级规则，并标记 ``mode`` 方便排查。

意图编号（与伪代码、prompts.py 保持一致）：
- 1 = AI 职业咨询（岗位、薪资、转行、行业前景……）
- 2 = 课程咨询（课程内容、价格、报名、试听、证书……）
- 0 = 其他 / 无法判断

设计原因：
- 验收环境没有 DEEPSEEK_API_KEY，规则兜底必须能独立、稳定通过「AI 职业规划→1、
  Python 课程多少钱→2、今天天气→0」三个用例；
- 课程类核心词权重高（2），价格类词权重低（1），职业类词权重 1：
  这样「Python 课程多少钱」= 课程(2)+多少钱(1) 明确指向课程；
- 两者分数相同时视为「无法判断」返回 0，避免规则误判。
"""

from __future__ import annotations

from typing import Iterable, Mapping

from config import get_logger
from llm.client import LLMUnavailableError, chat_completion, extract_json
from llm.prompts import build_intent_messages

logger = get_logger("crm.llm.intent")

# 意图编号常量
INTENT_CAREER: int = 1
INTENT_COURSE: int = 2
INTENT_OTHER: int = 0

# 合法意图编号集合
VALID_INTENT_CODES: tuple[int, ...] = (1, 2, 0)


# ---------------------------------------------------------------------------
# detect_intent 工具 schema（OpenAI tools 格式）
# 由 tools/functions.py 直接复用，保证「意图识别」与「function calling」定义一致。
# ---------------------------------------------------------------------------
DETECT_INTENT_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "detect_intent",
        "description": (
            "识别用户本轮咨询的意图编号并返回结果。"
            "1=AI 职业咨询（岗位前景、薪资、转行、就业、技能路线、行业趋势）；"
            "2=课程咨询（课程内容、价格、周期、报名、试听、证书、培训）；"
            "0=其他（问候、闲聊、与职业/课程无关或无法判断）。必须调用本工具，不要直接输出文字。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent_code": {
                    "type": "integer",
                    "enum": [1, 2, 0],
                    "description": "意图编号：1=AI 职业咨询，2=课程咨询，0=其他/无法判断",
                },
                "reason": {
                    "type": "string",
                    "description": "一句话中文说明判断依据，例如「用户询问 AI 岗位薪资」",
                },
            },
            "required": ["intent_code", "reason"],
        },
    },
}

# ---------------------------------------------------------------------------
# 关键词规则表
# 注意：按「子串包含」匹配，因此不要同时放「课程」和「课」这类互相包含的词，
# 否则同一处文字会被重复计分；短词只在语义独立时保留。
# ---------------------------------------------------------------------------
# 课程核心词：权重 2
COURSE_CORE_KEYWORDS: tuple[str, ...] = (
    "课程",
    "试听",
    "报名",
    "上课",
    "培训",
    "学习",
    "证书",
    "课时",
    "大纲",
    "讲师",
    "网课",
    "培训班",
    "学费",  # 学费偏课程属性，但价格类词整体权重低，这里仍按核心词计 2 会误导，见下方 PRICE
)
# 价格类词：权重 1（任务要求「价格类词权重低」）
PRICE_KEYWORDS: tuple[str, ...] = (
    "价格",
    "多少钱",
    "费用",
    "报价",
    "优惠",
    "便宜",
    "分期",
    "学费",
)
# 职业类词：权重 1
CAREER_KEYWORDS: tuple[str, ...] = (
    "职业",
    "岗位",
    "就业",
    "薪资",
    "工资",
    "转行",
    "前景",
    "招聘",
    "面试",
    "行业",
    "规划",
    "职位",
    "简历",
    "跳槽",
    "晋升",
    "职业发展",
    "技能路线",
)

# 从 COURSE_CORE_KEYWORDS 中剔除与价格词重复的项，避免重复计分
_COURSE_CORE_SET: tuple[str, ...] = tuple(
    word for word in COURSE_CORE_KEYWORDS if word not in PRICE_KEYWORDS
)

# 意图编号 → 中文名称（日志与 reason 使用）
_INTENT_NAMES: dict[int, str] = {1: "AI 职业咨询", 2: "课程咨询", 0: "其他/无法判断"}


def _match_keywords(query: str, keywords: Iterable[str]) -> list[str]:
    """返回文本中命中的关键词列表（保持关键词表顺序，去重）。"""
    hits: list[str] = []
    for word in keywords:
        if word and word in query and word not in hits:
            hits.append(word)
    return hits


def _keyword_scores(query: str) -> tuple[int, int, list[str], list[str]]:
    """计算课程 / 职业两类的加权分数与命中词。

    Args:
        query: 用户输入文本。

    Returns:
        ``(course_score, career_score, course_hits, career_hits)``
    """
    text = (query or "").strip().lower()
    if not text:
        return 0, 0, [], []

    course_hits = _match_keywords(text, _COURSE_CORE_SET)
    price_hits = _match_keywords(text, PRICE_KEYWORDS)
    career_hits = _match_keywords(text, CAREER_KEYWORDS)

    course_score = 2 * len(course_hits) + 1 * len(price_hits)
    career_score = 1 * len(career_hits)
    # 命中词列表合并价格词，便于日志/理由展示
    return course_score, career_score, course_hits + price_hits, career_hits


def keyword_intent(query: str) -> int:
    """基于关键词加权打分识别意图（规则兜底路径）。

    打分规则：
    - 课程核心词（课程/报名/试听/上课/培训/学习/证书…）每个 +2；
    - 价格类词（价格/多少钱/费用/优惠…）每个 +1（价格类权重低）；
    - 职业类词（职业/岗位/就业/薪资/工资/转行/前景/招聘/面试/行业/规划…）每个 +1；
    - 课程分 > 职业分 → 2；职业分 > 课程分 → 1；同分或都未命中 → 0（无法判断）。

    Args:
        query: 用户本轮输入。

    Returns:
        意图编号 1 / 2 / 0。
    """
    course_score, career_score, _, _ = _keyword_scores(query)
    if course_score > career_score:
        return INTENT_COURSE
    if career_score > course_score:
        return INTENT_CAREER
    return INTENT_OTHER


def _coerce_intent_code(value: object) -> int | None:
    """把模型给出的意图编号转成合法 int；非法返回 None。

    注意：``bool`` 是 ``int`` 的子类，True/False 不应被当作 1/0，需显式排除。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        code = value
    elif isinstance(value, float) and value.is_integer():
        code = int(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            code = int(text)
        except ValueError:
            return None
    else:
        return None
    return code if code in VALID_INTENT_CODES else None


def _rule_result(query: str, exc: Exception | None = None) -> dict:
    """构造规则兜底结果（含命中关键词理由）。

    Args:
        query: 用户输入。
        exc: 触发降级的异常（可为 None，仅用于日志）。

    Returns:
        ``{"intent_code": int, "reason": str, "mode": "rule"}``
    """
    course_score, career_score, course_hits, career_hits = _keyword_scores(query)
    intent_code = keyword_intent(query)
    if exc is not None:
        logger.warning("意图识别 LLM 路径失败，降级关键词规则：%s", exc)

    if intent_code == INTENT_COURSE:
        reason = f"命中课程类关键词「{'、'.join(course_hits[:5])}」，判断为课程咨询"
    elif intent_code == INTENT_CAREER:
        reason = f"命中职业类关键词「{'、'.join(career_hits[:5])}」，判断为 AI 职业咨询"
    elif course_hits or career_hits:
        reason = (
            f"课程分 {course_score} 与职业分 {career_score} 相当，"
            f"无法明确判断（课程词：{'、'.join(course_hits[:3]) or '无'}；"
            f"职业词：{'、'.join(career_hits[:3]) or '无'}）"
        )
    else:
        reason = "未命中职业/课程关键词，判断为其他咨询"
    return {"intent_code": intent_code, "reason": reason, "mode": "rule"}


def detect_intent_with_reason(
    query: str, history: list[dict] | None = None
) -> dict:
    """识别意图并返回判断依据；先走 LLM，失败自动降级为关键词规则。

    Args:
        query: 用户本轮输入。
        history: 最近聊天历史（用于理解「多少钱」这类省略句），可为 None。

    Returns:
        ``{"intent_code": 1/2/0, "reason": str, "mode": "llm" | "rule"}``
    """
    text = (query or "").strip() if isinstance(query, str) else ""
    if not text:
        return {"intent_code": INTENT_OTHER, "reason": "用户输入为空，判断为其他咨询", "mode": "rule"}

    # ---------- 路径一：LLM function calling（强制调用 detect_intent） ----------
    try:
        response = chat_completion(
            messages=build_intent_messages(text, history),
            tools=[DETECT_INTENT_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "detect_intent"}},
            temperature=0.0,
        )
        for call in response.get("tool_calls") or []:
            if call.get("name") != "detect_intent":
                continue
            arguments = call.get("arguments") or {}
            code = _coerce_intent_code(arguments.get("intent_code"))
            if code is None:
                logger.warning("模型返回非法 intent_code=%r，降级规则", arguments.get("intent_code"))
                break
            reason = str(arguments.get("reason") or "").strip() or "模型未给出判断依据"
            logger.info("意图识别（LLM）：query=%r intent_code=%s reason=%s", text[:50], code, reason)
            return {"intent_code": code, "reason": reason, "mode": "llm"}

        # 模型没有按约定调用工具时，尝试从文本里直接解析 JSON（兼容部分兼容服务）
        parsed = extract_json(response.get("content") or "")
        if isinstance(parsed, dict):
            code = _coerce_intent_code(parsed.get("intent_code"))
            if code is not None:
                reason = str(parsed.get("reason") or "").strip() or "模型以 JSON 文本返回判断依据"
                logger.info("意图识别（LLM 文本 JSON）：query=%r intent_code=%s", text[:50], code)
                return {"intent_code": code, "reason": reason, "mode": "llm"}

        logger.warning("LLM 未返回合法的 detect_intent 结果，降级关键词规则")
        return _rule_result(text)
    except LLMUnavailableError as exc:
        # 无 Key：预期内的降级路径，只记 warning 不抛异常
        return _rule_result(text, exc)
    except Exception as exc:  # 网络错误 / 返回结构异常 / 解析异常
        logger.warning("意图识别 LLM 调用异常，降级关键词规则：%s", exc)
        return _rule_result(text, exc)


def detect_intent_code(query: str, history: list[dict] | None = None) -> int:
    """识别用户意图编号（主流程 ``SeaChatInterview`` 使用的入口）。

    Args:
        query: 用户本轮输入。
        history: 最近聊天历史，可为 None。

    Returns:
        意图编号 1 / 2 / 0；任何异常都降级为 0，保证主流程不中断。
    """
    try:
        return int(detect_intent_with_reason(query, history)["intent_code"])
    except Exception as exc:  # 理论上不会发生，兜底保证不把异常抛给 Flask
        logger.warning("意图识别出现未预期异常，按其他意图处理：%s", exc)
        return INTENT_OTHER
