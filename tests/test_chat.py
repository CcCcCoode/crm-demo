"""主流程业务测试（离线模式：无 API Key、hash 嵌入、无网络搜索）。

覆盖验收要点：
- 无历史 → 欢迎语 + 创建用户；
- 有历史 → 意图识别 1 / 2 / 0 三类输入返回不同结果；
- 工具调用记录（检索工具真实执行）；
- 对话记录落库、用户画像抽取与更新；
- 参数校验与友好错误；
- 流式事件序列完整。
"""

from __future__ import annotations

import pytest

from database import db
from services import chat as chat_service


def _first_turn(phone: str) -> dict:
    """新手机号第一轮：应返回欢迎语。"""
    result = chat_service.process_chat(phone, "你好，我想咨询一下")
    assert result["ok"] is True
    assert result["is_new"] is True
    assert result["mode"] == "welcome"
    assert "SeaChat" in result["reply"] or "智能顾问" in result["reply"]
    return result


def test_first_message_returns_welcome_and_creates_user():
    """无历史的首轮咨询返回欢迎语，并在 SQLite 中创建用户。"""
    phone = "13900001001"
    result = _first_turn(phone)

    user = db.get_user_info(phone)
    assert user is not None
    assert user["phone_number"] == phone
    assert user["stage"] in ("新咨询", "已了解", "已报名")

    history = db.fetch_chat_history(phone)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"].strip()
    assert result["reply"].strip() == history[1]["content"].strip()


@pytest.mark.parametrize(
    ("phone", "query", "expected_intent"),
    [
        ("13900001002", "AI 职业规划怎么做？想了解岗位前景和薪资", 1),
        ("13900001003", "Python 课程多少钱？上课周期多久", 2),
        ("13900001004", "今天天气怎么样？", 0),
    ],
)
def test_intent_classification(phone, query, expected_intent):
    """有历史后，三类输入分别命中意图 1 / 2 / 0，且回答各不相同。"""
    _first_turn(phone)
    result = chat_service.process_chat(phone, query)

    assert result["ok"] is True
    assert result["is_new"] is False
    assert result["intent_code"] == expected_intent
    assert result["reply"].strip()
    assert result["mode"] in ("llm", "rule")
    # 回答不能再是欢迎语
    assert result["mode"] != "welcome"


def test_tool_calls_are_recorded():
    """职业咨询会真实调用 web_search / search_market_docs 工具并记录。"""
    phone = "13900001005"
    _first_turn(phone)
    result = chat_service.process_chat(phone, "AI 行业现在就业前景和薪资怎么样？")

    assert result["intent_code"] == 1
    names = [item["name"] for item in result["tool_calls"]]
    assert "web_search" in names
    assert "search_market_docs" in names


def test_course_tool_calls_are_recorded():
    """课程咨询会真实调用 search_course_docs 工具。"""
    phone = "13900001006"
    _first_turn(phone)
    result = chat_service.process_chat(phone, "你们有哪些 Python 课程，价格多少？")

    assert result["intent_code"] == 2
    names = [item["name"] for item in result["tool_calls"]]
    assert "search_course_docs" in names


def test_chat_history_saved_in_order():
    """每轮对话按「用户 → 客服」顺序落库，并记录意图编号。"""
    phone = "13900001007"
    _first_turn(phone)
    chat_service.process_chat(phone, "想学数据分析课程")

    history = db.fetch_chat_history(phone)
    assert len(history) == 4
    assert [item["role"] for item in history] == ["user", "assistant", "user", "assistant"]
    # 第二轮的意图编号落库（课程咨询 = 2）
    assert history[2]["intent_code"] in (None, 2)
    assert history[3]["intent_code"] == 2


def test_profile_updated_from_conversation():
    """用户自报姓名 / 方向 / 预算后，画像写回 SQLite。"""
    phone = "13900001008"
    result = chat_service.process_chat(phone, "我叫李明，想转行做数据分析，预算 1 万左右")

    assert result["ok"] is True
    user = db.get_user_info(phone)
    assert user["name"] == "李明"
    assert user["intention_direction"] == "数据分析"
    assert user["budget"] == pytest.approx(10000.0)


def test_sql_agent_rule_mode():
    """SQL Agent 在无 Key 时用规则兜底，课程查询可返回数据。"""
    from database.sql_agent import answer_with_sql_agent

    result = answer_with_sql_agent("数据库里有哪些课程？")
    assert result["mode"] == "rule"
    assert result["ok"] is True
    assert result["rows"]
    assert result["answer"].strip()


def test_sql_agent_salary_threshold():
    """「薪资 2 万以上的职业」规则兜底能解析阈值并返回对应职业。"""
    from database.sql_agent import answer_with_sql_agent

    result = answer_with_sql_agent("薪资 2 万以上的职业有哪些")
    assert result["mode"] == "rule"
    assert result["ok"] is True
    rows = result["rows"]
    assert rows, f"应返回职业数据，实际：{result}"
    salaries = [row.get("salary_max") or row.get("salary_min") or 0 for row in rows]
    assert max(float(value) for value in salaries) >= 20000


def test_stream_events_sequence():
    """流式接口事件完整：start → intent → tool_call → delta → done，且文本与 done.reply 一致。"""
    phone = "13900001009"
    list(chat_service.stream_chat(phone, "你好"))  # 首轮欢迎语

    events = list(chat_service.stream_chat(phone, "Python 课程多少钱？"))
    types = [event["type"] for event in events]
    assert types[0] == "start"
    assert "intent" in types
    assert "tool_call" in types
    assert "delta" in types
    assert types[-1] == "done"

    deltas = "".join(event["content"] for event in events if event["type"] == "delta")
    done = events[-1]
    assert done["reply"].strip() == deltas.strip()
    assert done["intent_code"] == 2
    assert done["tool_calls"]


def test_new_user_welcome_only_once():
    """欢迎语只在新会话出现一次，第二轮进入正常意图流程。"""
    phone = "13900001010"
    first = chat_service.process_chat(phone, "你好")
    second = chat_service.process_chat(phone, "我想了解课程")

    assert first["mode"] == "welcome"
    assert second["mode"] != "welcome"


@pytest.mark.parametrize(
    ("phone", "query"),
    [
        ("12345", "你好"),
        ("13900001011", ""),
        ("13900001011", "   "),
    ],
)
def test_invalid_input_returns_friendly_error(phone, query):
    """非法手机号 / 空问题返回友好错误，不抛异常。"""
    result = chat_service.process_chat(phone, query)
    assert result["ok"] is False
    assert result["error"]
