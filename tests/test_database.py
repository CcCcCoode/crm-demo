"""database 数据层测试。

覆盖范围（对应契约第 2、3 节）：
- 建表幂等、示例数据、连接参数；
- 用户画像 CRUD（含 budget 字符串/数字、非法 stage 忽略、局部更新）；
- 聊天历史写入/读取顺序与 limit 语义；
- 只读 SQL 校验（拒绝 DELETE / DROP / 多语句）与查询执行；
- get_schema_description / list_tables / get_database_stats；
- SQL Agent 规则兜底：「有哪些课程」「薪资 2 万以上的职业有哪些」「最贵的课程」及无法映射场景。

说明：
- 所有用例使用随机手机号，避免共享的 session 级数据库互相污染；
- 不依赖 DEEPSEEK_API_KEY（conftest 中已置空），SQL Agent 走规则兜底；
- 只导入 database 包，不触碰其它 agent 的模块。
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

from database import db, sql_agent

# 规则兜底返回结构的固定键集合（契约第 3 节）
RESULT_KEYS = {"ok", "answer", "sql", "rows", "mode", "error"}


@pytest.fixture()
def phone() -> str:
    """生成每个用例独立的 11 位数字手机号，避免共享数据库主键冲突。"""
    return "139" + f"{uuid.uuid4().int % 100_000_000:08d}"


@pytest.fixture()
def max_course_price() -> float:
    """读取课程表中的最高价格，供「最贵的课程」断言使用。"""
    rows = db.execute_sql("SELECT MAX(price) AS price FROM courses")
    assert rows and rows[0]["price"] is not None
    return float(rows[0]["price"])


# ---------------------------------------------------------------------------
# 建表与连接
# ---------------------------------------------------------------------------
def test_init_db_idempotent() -> None:
    """init_db 可重复执行：不重复建表、不重复插入示例数据。"""
    db.init_db()
    first_stats = db.get_database_stats()
    db.init_db()
    second_stats = db.get_database_stats()

    assert first_stats == second_stats
    assert set(first_stats) == {"users", "chat_history", "courses", "career_info"}
    assert first_stats["courses"] >= 6
    assert first_stats["career_info"] >= 6


def test_get_connection_settings() -> None:
    """连接默认 row_factory=sqlite3.Row，且开启外键约束。"""
    connection = db.get_connection()
    try:
        assert connection.row_factory is sqlite3.Row
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        assert int(foreign_keys) == 1
    finally:
        connection.close()


def test_list_tables_contains_all_business_tables() -> None:
    """list_tables 返回四张业务表且不含 sqlite_ 系统表。"""
    tables = db.list_tables()
    assert {"users", "chat_history", "courses", "career_info"} <= set(tables)
    assert all(not name.startswith("sqlite_") for name in tables)


# ---------------------------------------------------------------------------
# 用户画像 CRUD
# ---------------------------------------------------------------------------
def test_allowed_user_fields_contract() -> None:
    """用户画像字段白名单与契约一致。"""
    assert db.ALLOWED_USER_FIELDS == (
        "name",
        "intention_direction",
        "budget",
        "stage",
        "notes",
    )


def test_create_user_and_get_user_info(phone: str) -> None:
    """create_user 新建用户并返回完整行，get_user_info 能读回同样内容。"""
    created = db.create_user(
        phone,
        name="张三",
        intention_direction="AI 应用开发",
        budget="1.5万",
        stage="新咨询",
        notes="想转行",
    )

    assert isinstance(created, dict)
    assert created["phone_number"] == phone
    assert created["name"] == "张三"
    assert created["intention_direction"] == "AI 应用开发"
    assert created["budget"] == 15000.0
    assert created["stage"] == "新咨询"
    assert created["notes"] == "想转行"
    assert created["created_at"] and created["updated_at"]

    fetched = db.get_user_info(phone)
    assert fetched is not None
    assert fetched["phone_number"] == phone
    assert fetched["name"] == "张三"
    assert fetched["budget"] == 15000.0


def test_create_user_updates_existing(phone: str) -> None:
    """同一手机号再次 create_user 时更新传入字段，而不是插入重复行。"""
    db.create_user(phone, name="李四", budget=12800)
    updated = db.create_user(phone, name="李四四", stage="已了解")

    assert updated["phone_number"] == phone
    assert updated["name"] == "李四四"
    assert updated["stage"] == "已了解"
    # 未传入的 budget 保持原值
    assert updated["budget"] == 12800.0


def test_update_user_info_partial_and_budget(phone: str) -> None:
    """update_user_info 只更新传入字段；budget 支持字符串与数字。"""
    db.create_user(phone, name="王五", intention_direction="数据分析", budget=9800)

    # 局部更新：只改 stage，其它字段保持不变
    updated = db.update_user_info(phone, stage="已报名")
    assert updated["stage"] == "已报名"
    assert updated["name"] == "王五"
    assert updated["intention_direction"] == "数据分析"
    assert updated["budget"] == 9800.0

    # budget 字符串："2万" → 20000.0
    assert db.update_user_info(phone, budget="2万")["budget"] == 20000.0
    # budget 中文数字："两万" → 20000.0
    assert db.update_user_info(phone, budget="两万")["budget"] == 20000.0
    # budget 数字：19800 → 19800.0
    assert db.update_user_info(phone, budget=19800)["budget"] == 19800.0
    # budget 带货币符号与千分位
    assert db.update_user_info(phone, budget="¥19,800")["budget"] == 19800.0


def test_update_user_info_ignores_invalid_values(phone: str) -> None:
    """非法 stage、未知字段被忽略，已有数据不被破坏。"""
    db.create_user(phone, name="赵六", stage="新咨询")
    updated = db.update_user_info(
        phone,
        name="赵六六",
        stage="已退学",  # 不在允许枚举内
        unknown_field="hack",  # 不在白名单内
        budget="不是数字",  # 无法解析
    )

    assert updated["name"] == "赵六六"
    assert updated["stage"] == "新咨询"  # 非法 stage 被忽略
    assert updated["budget"] is None  # 非法 budget 被忽略，保持原值 None


def test_create_user_requires_phone() -> None:
    """手机号为空时 create_user 抛 ValueError。"""
    with pytest.raises(ValueError):
        db.create_user("")


def test_get_user_info_missing_returns_none(phone: str) -> None:
    """查询不存在的用户返回 None。"""
    assert db.get_user_info(phone) is None


# ---------------------------------------------------------------------------
# 聊天历史
# ---------------------------------------------------------------------------
def test_insert_message_returns_increasing_id(phone: str) -> None:
    """insert_message 返回自增 id，且 id 递增。"""
    first = db.insert_message(phone, "user", "你好", intent_code=1)
    second = db.insert_message(phone, "assistant", "您好，请问想了解什么？")
    assert isinstance(first, int) and isinstance(second, int)
    assert second > first


def test_insert_message_rejects_invalid_role(phone: str) -> None:
    """role 只允许 user / assistant，其它值抛 ValueError。"""
    with pytest.raises(ValueError):
        db.insert_message(phone, "system", "非法角色")


def test_fetch_chat_history_order_and_limit(phone: str) -> None:
    """历史按时间正序；limit 返回最近 N 条但仍为正序。"""
    contents = [f"第{i}条消息" for i in range(5)]
    for index, content in enumerate(contents):
        role = "user" if index % 2 == 0 else "assistant"
        db.insert_message(phone, role, content)

    all_messages = db.fetch_chat_history(phone)
    assert [message["content"] for message in all_messages] == contents
    assert all(
        {"id", "phone_number", "role", "content", "intent_code", "created_at"} <= set(message)
        for message in all_messages
    )

    recent = db.fetch_chat_history(phone, limit=2)
    assert [message["content"] for message in recent] == contents[-2:]
    assert db.fetch_chat_history(phone, limit=0) == []
    assert db.fetch_chat_history(phone, limit=-1) == []
    assert db.fetch_chat_history("") == []


def test_update_chat_history_saves_round(phone: str) -> None:
    """update_chat_history 依次写入 user 提问与 assistant 回答。"""
    db.update_chat_history(phone, "有哪些 AI 课程？", "我们有三门 AI 相关课程。", intent_code=1)
    messages = db.fetch_chat_history(phone)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "有哪些 AI 课程？"
    assert messages[0]["intent_code"] == 1
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "我们有三门 AI 相关课程。"
    # 本轮意图编号同时写入两行，读取端任取其一都能拿到
    assert messages[1]["intent_code"] == 1


# ---------------------------------------------------------------------------
# 只读 SQL 校验与执行
# ---------------------------------------------------------------------------
def test_is_readonly_sql_accepts_select_and_with() -> None:
    """合法的单条 SELECT/WITH（含注释、字符串内分号、末尾分号）返回 True。"""
    assert db.is_readonly_sql("SELECT * FROM courses") is True
    assert db.is_readonly_sql("select name, price from courses limit 3;") is True
    assert db.is_readonly_sql("-- 课程查询\nSELECT * FROM courses") is True
    assert db.is_readonly_sql("/* 块注释 */ SELECT COUNT(*) FROM users") is True
    assert db.is_readonly_sql("WITH t AS (SELECT 1 AS x) SELECT * FROM t") is True
    assert db.is_readonly_sql("SELECT ';' AS sep FROM courses LIMIT 1") is True


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "-- 只有注释",
        "DELETE FROM users",
        "delete from users where phone_number = '1'",
        "UPDATE users SET name = 'x'",
        "INSERT INTO users (phone_number) VALUES ('1')",
        "DROP TABLE users",
        "ALTER TABLE users ADD COLUMN x TEXT",
        "PRAGMA table_info(users)",
        "VACUUM",
        "SELECT * FROM users; DELETE FROM users",
        "SELECT * FROM users; SELECT * FROM courses",
        "WITH t AS (SELECT 1) DELETE FROM users",
    ],
)
def test_is_readonly_sql_rejects_unsafe(sql: str) -> None:
    """写操作、多语句、空语句一律判为非法。"""
    assert db.is_readonly_sql(sql) is False


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM users",
        "DROP TABLE users",
        "UPDATE chat_history SET content = 'x'",
        "SELECT * FROM users; DELETE FROM users",
        "SELECT * FROM courses; SELECT * FROM career_info",
    ],
)
def test_execute_sql_rejects_unsafe(sql: str) -> None:
    """execute_sql 对写操作与多语句抛 ValueError，并给出统一提示。"""
    with pytest.raises(ValueError) as error_info:
        db.execute_sql(sql)
    assert db.FORBIDDEN_SQL_MESSAGE in str(error_info.value)


def test_execute_sql_returns_dict_rows() -> None:
    """execute_sql 返回 dict 行列表，末尾分号可正常执行。"""
    rows = db.execute_sql("SELECT name, price FROM courses ORDER BY price DESC;")
    assert rows
    assert isinstance(rows[0], dict)
    assert "name" in rows[0] and "price" in rows[0]
    assert rows[0]["price"] >= rows[-1]["price"]


def test_execute_sql_limits_rows() -> None:
    """结果行数不超过 config.SQL_MAX_ROWS。"""
    rows = db.execute_sql("SELECT * FROM career_info")
    from config import config

    assert len(rows) <= int(config.SQL_MAX_ROWS)


def test_execute_sql_bad_table_returns_empty() -> None:
    """表不存在属于执行期错误：记录日志并返回空列表，不抛异常。"""
    assert db.execute_sql("SELECT * FROM not_exist_table") == []


# ---------------------------------------------------------------------------
# 表结构说明与统计
# ---------------------------------------------------------------------------
def test_get_schema_description_non_empty() -> None:
    """schema 描述非空，包含四张表名、中文含义与示例数据。"""
    description = db.get_schema_description()
    assert isinstance(description, str)
    assert description.strip()
    for table in ("users", "chat_history", "courses", "career_info"):
        assert table in description
    assert "示例数据" in description
    assert "课程" in description and "职业" in description


# ---------------------------------------------------------------------------
# SQL Agent 规则兜底（无 DEEPSEEK_API_KEY）
# ---------------------------------------------------------------------------
def test_sql_tool_schema_contract() -> None:
    """工具 schema 符合 OpenAI function calling 结构。"""
    assert sql_agent.SQL_TOOL_SCHEMA["type"] == "function"
    function = sql_agent.SQL_TOOL_SCHEMA["function"]
    assert function["name"] == "query_database"
    assert function["parameters"]["required"] == ["sql"]
    assert "sql" in function["parameters"]["properties"]


def test_rule_based_course_list() -> None:
    """「有哪些课程」返回课程列表且 ok=True。"""
    result = sql_agent.rule_based_answer("有哪些课程")

    assert set(result) == RESULT_KEYS
    assert result["ok"] is True
    assert result["mode"] == "rule"
    assert result["error"] is None
    assert result["rows"]
    assert len(result["rows"]) >= 6
    assert all("name" in row and "price" in row for row in result["rows"])
    assert "Python" in "".join(str(row["name"]) for row in result["rows"])
    assert "课程" in result["answer"]
    assert len(result["sql"]) == 1


def test_rule_based_salary_careers() -> None:
    """「薪资 2 万以上的职业有哪些」按 salary_max >= 20000 返回正确行。"""
    result = sql_agent.rule_based_answer("薪资 2 万以上的职业有哪些")

    assert result["ok"] is True
    assert result["mode"] == "rule"
    assert result["rows"]
    assert all(float(row["salary_max"]) >= 20000 for row in result["rows"])
    titles = {row["title"] for row in result["rows"]}
    assert "大模型算法工程师" in titles
    assert "算法工程师" in titles
    # 薪资低于 2 万的职业不应出现
    assert "数据运营专员" not in titles
    assert "职业" in result["answer"]


def test_rule_based_most_expensive_course(max_course_price: float) -> None:
    """「最贵的课程」返回价格最高的课程。"""
    result = sql_agent.rule_based_answer("最贵的课程是什么")

    assert result["ok"] is True
    assert result["rows"]
    assert float(result["rows"][0]["price"]) == max_course_price
    assert "最贵" in result["answer"]


def test_rule_based_course_price_by_topic() -> None:
    """「Python 课程多少钱」按主题过滤并返回价格。"""
    result = sql_agent.rule_based_answer("Python 课程多少钱")

    assert result["ok"] is True
    assert result["rows"]
    assert all("Python" in str(row["name"]) for row in result["rows"])
    assert "price" in result["rows"][0]


def test_rule_based_user_count() -> None:
    """「现在有多少用户」返回用户统计。"""
    result = sql_agent.rule_based_answer("现在有多少用户")

    assert result["ok"] is True
    assert result["mode"] == "rule"
    assert "用户" in result["answer"]
    assert result["rows"] and "user_count" in result["rows"][0]


def test_rule_based_unmapped_returns_not_ok() -> None:
    """无法映射的问题返回 ok=False，但仍有可读的中文提示。"""
    result = sql_agent.rule_based_answer("今天天气怎么样")

    assert set(result) == RESULT_KEYS
    assert result["ok"] is False
    assert result["mode"] == "rule"
    assert result["answer"]
    assert result["error"]
    assert result["rows"] == []
    assert result["sql"] == []


def test_rule_based_empty_question() -> None:
    """空问题返回 ok=False，不抛异常。"""
    result = sql_agent.rule_based_answer("")
    assert result["ok"] is False
    assert result["answer"]


def test_answer_with_sql_agent_falls_back_without_llm() -> None:
    """没有 DEEPSEEK_API_KEY 时 answer_with_sql_agent 自动走规则兜底。"""
    result = sql_agent.answer_with_sql_agent("有哪些课程")

    assert set(result) == RESULT_KEYS
    assert result["mode"] == "rule"
    assert result["ok"] is True
    assert result["rows"]


def test_answer_with_sql_agent_empty_question() -> None:
    """空问题直接返回 ok=False（不进入 LLM 路径）。"""
    result = sql_agent.answer_with_sql_agent("")
    assert result["ok"] is False
    assert set(result) == RESULT_KEYS


def test_generate_sql_without_llm_returns_none() -> None:
    """无可用 LLM 时 generate_sql 返回 None（不抛异常）。"""
    assert sql_agent.generate_sql("有哪些课程") is None
