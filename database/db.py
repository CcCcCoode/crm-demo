"""SQLite 数据访问层（database/db.py）。

职责：
- 管理 SQLite 连接（行工厂、外键、超时、自动建父目录）；
- 执行 ``database/schema.sql`` 建表并在课程/职业表为空时灌入中文示例数据；
- 提供用户画像 CRUD、聊天历史读写；
- 提供只读 SQL 安全校验与执行能力（SQL Agent / query_database 工具使用）；
- 生成注入 SQL Agent 提示词的中文表结构说明与统计信息。

设计原因：
- 全项目只有本模块直接接触 SQLite，其它模块通过函数调用，便于统一做安全与异常处理；
- 所有 SQL 均使用参数化查询；只读 SQL 在执行前做「去注释 + 单语句 + SELECT/WITH 前缀」校验；
- 函数返回普通 dict / list[dict]，避免把 sqlite3.Row 泄漏到业务层（Flask 无法直接序列化 Row）。
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import config, get_logger

logger = get_logger("crm.database")

# 用户画像允许写入的字段（与契约 ALLOWED_USER_FIELDS 完全一致）
ALLOWED_USER_FIELDS: tuple[str, ...] = ("name", "intention_direction", "budget", "stage", "notes")

# 咨询阶段白名单（与 schema.sql 的 CHECK 约束保持一致）
VALID_STAGES: tuple[str, ...] = ("新咨询", "已了解", "已报名")

# 聊天消息角色白名单
VALID_CHAT_ROLES: tuple[str, ...] = ("user", "assistant")

# 非法 SQL 的统一错误信息（契约要求 execute_sql 抛出的 ValueError 使用该文本）
FORBIDDEN_SQL_MESSAGE: str = "出于安全限制，只能执行只读的 SELECT 查询。"

# 建表脚本与示例数据分界标记（见 schema.sql）
_SCHEMA_FILE: Path = Path(__file__).resolve().parent / "schema.sql"
_SEED_MARKER: str = "-- @@SEED_DATA@@"
# 分界标记必须独占一行（允许前后空白）：避免注释里提到标记时被误判为分界点
_SEED_MARKER_PATTERN = re.compile(r"(?m)^[ \t]*--[ \t]*@@SEED_DATA@@[ \t]*$")

# 只读 SQL 必须匹配的前缀：SELECT 或 WITH（CTE）
_READONLY_SQL_PREFIX = re.compile(r"(?is)^\s*(?:select|with)\b")

# 即使以 WITH 开头，只要出现这些关键字就判定为写操作/危险语句并拒绝
_FORBIDDEN_SQL_KEYWORDS: tuple[str, ...] = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "analyze",
    "truncate",
    "grant",
    "revoke",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "release",
    "transaction",
    "exec",
    "execute",
)

# 预算字符串解析：支持 "2万"、"1.5w"、"20000"、"¥19800" 等写法
_BUDGET_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(万|w|W|k|K|千|元)?")
# 中文数字 + 万（如 "两万"）
_BUDGET_CHINESE_PATTERN = re.compile(r"([一两二三四五六七八九十])\s*万")
_BUDGET_CHINESE_DIGITS: dict[str, float] = {
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

# 中文表结构说明（静态部分；示例数据在 get_schema_description() 中动态追加）
_SCHEMA_STATIC_DESCRIPTION: str = """数据库为 SQLite（UTF-8 编码），共 4 张业务表，只允许只读 SELECT 查询：

1. 表 users —— 用户画像表（每个手机号一行）
   - phone_number TEXT 主键：用户手机号
   - name TEXT：用户姓名/称呼
   - intention_direction TEXT：意向方向（如 AI 职业、Python 课程、数据分析）
   - budget REAL：预算金额，单位「元」
   - stage TEXT：咨询阶段，取值只能是 新咨询 / 已了解 / 已报名
   - notes TEXT：用户关注点与备注
   - created_at TEXT：创建时间（本地时间文本）
   - updated_at TEXT：最后更新时间（本地时间文本）

2. 表 chat_history —— 聊天历史表（每轮对话保存 user、assistant 两条记录）
   - id INTEGER 主键自增：消息先后顺序
   - phone_number TEXT：所属用户手机号
   - role TEXT：消息角色，只能是 user / assistant
   - content TEXT：消息内容
   - intent_code INTEGER：本轮意图编号（1=AI 职业咨询，2=课程咨询，0=其他）
   - created_at TEXT：消息时间（本地时间文本）

3. 表 courses —— 课程表
   - id INTEGER 主键自增
   - name TEXT 课程名称（唯一，中文）
   - category TEXT 课程分类（编程开发/数据分析/人工智能/前端开发/算法/办公自动化）
   - price REAL 课程价格，单位「元」
   - duration TEXT 学习周期（如 "24 周"）
   - level TEXT 基础要求
   - suitable_for TEXT 适合人群
   - description TEXT 课程简介
   - created_at TEXT 创建时间

4. 表 career_info —— 职业信息表
   - id INTEGER 主键自增
   - title TEXT 岗位名称（唯一）
   - industry TEXT 所属行业
   - salary_min INTEGER 月薪下限，单位「元」
   - salary_max INTEGER 月薪上限，单位「元」
   - skills TEXT 核心技能要求
   - description TEXT 岗位职责说明
   - created_at TEXT 创建时间

查询提示：
- 课程列表/价格 → courses 表；岗位列表/薪资 → career_info 表（薪资为月薪）；
- 用户数量/画像 → users 表；聊天记录数量 → chat_history 表；
- 生成 SQL 时只允许单条 SELECT / WITH 查询，禁止任何写操作；建议加 LIMIT 限制行数。
"""


# ---------------------------------------------------------------------------
# 连接管理
# ---------------------------------------------------------------------------
def get_connection() -> sqlite3.Connection:
    """创建并返回一个 SQLite 连接。

    用途：
        所有数据库函数都通过该函数获取连接，保证 row_factory / 外键 / 超时配置一致。

    参数：
        无（数据库路径来自 ``config.SQLITE_PATH``）。

    返回值：
        ``sqlite3.Connection``：``row_factory=sqlite3.Row``，开启外键约束，
        超时 10 秒；数据库父目录不存在时自动创建（UTF-8 由 SQLite 默认保证）。
    """
    db_path = Path(config.SQLITE_PATH)
    # 自动创建父目录，避免首次运行因 data/ 不存在而报 unable to open database file
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - 依赖文件系统状态
        logger.error("创建数据库目录失败：%s", exc)
        raise RuntimeError(f"创建数据库目录失败：{db_path.parent}") from exc

    # timeout=10：并发写时最多等待 10 秒锁；不使用 check_same_thread 等额外参数
    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    # 每次连接都显式开启外键与忙等待，避免连接池/多线程下配置丢失
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """数据库事务上下文管理器：自动提交/回滚并关闭连接。

    用途：
        统一处理连接的提交、异常回滚与关闭，避免连接泄漏。

    参数：
        无。

    返回值：
        迭代器，产出已配置好的 ``sqlite3.Connection``。

    异常：
        业务代码内部抛出的异常会先回滚事务，再原样向上抛出。
    """
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        # 任何异常都回滚，保证不会写入半成品数据
        connection.rollback()
        raise
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 数据库初始化
# ---------------------------------------------------------------------------
def _load_schema_sections() -> tuple[str, str]:
    """读取 schema.sql 并切分为「建表 DDL」与「示例数据」两段。

    参数：
        无。

    返回值：
        ``(ddl_sql, seed_sql)`` 二元组；未找到分界标记时示例数据部分为空串。

    异常：
        文件不存在或读取失败时抛 ``RuntimeError``（初始化不允许静默失败）。
    """
    try:
        schema_text = _SCHEMA_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("读取建表脚本失败：%s", exc)
        raise RuntimeError(f"读取建表脚本失败：{_SCHEMA_FILE}") from exc

    separator = _SEED_MARKER_PATTERN.search(schema_text)
    if separator is not None:
        ddl_text = schema_text[: separator.start()]
        seed_text = schema_text[separator.end():]
    else:
        logger.warning("schema.sql 未找到独占一行的示例数据标记 %s，将只建表", _SEED_MARKER)
        ddl_text, seed_text = schema_text, ""
    return ddl_text, seed_text


def _table_is_empty(connection: sqlite3.Connection, table: str) -> bool:
    """判断指定表是否为空（内部函数，table 只能传本模块内的固定表名）。

    参数：
        connection: 已打开的数据库连接。
        table: 表名（仅用于内部固定表名，绝不接收用户输入）。

    返回值：
        表不存在或行数为 0 时返回 True，否则返回 False。
    """
    try:
        row = connection.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()
    except sqlite3.Error as exc:
        logger.warning("统计表 %s 行数失败：%s", table, exc)
        return True
    return int(row["total"]) == 0


def init_db() -> None:
    """初始化数据库：建表（幂等）并按需写入课程/职业示例数据。

    用途：
        应用启动、测试会话初始化时调用；重复调用不会重复建表或插入示例数据。

    参数：
        无。

    返回值：
        无。

    异常：
        建表脚本读取失败或 SQL 执行失败时抛 ``RuntimeError``，并记录 error 日志。
    """
    config.ensure_dirs()
    ddl_sql, seed_sql = _load_schema_sections()
    try:
        with _transaction() as connection:
            # 1) 建表 + 索引：全部带 IF NOT EXISTS，天然幂等
            connection.executescript(ddl_sql)
            # 2) 只有课程表或职业表为空时才灌入示例数据
            need_seed = _table_is_empty(connection, "courses") or _table_is_empty(
                connection, "career_info"
            )
            if need_seed and seed_sql.strip():
                connection.executescript(seed_sql)
                logger.info("课程/职业表为空，已写入中文示例数据")
            else:
                logger.info("课程/职业示例数据已存在，跳过写入")
        logger.info("数据库初始化完成，当前数据量：%s", get_database_stats())
    except sqlite3.Error as exc:
        logger.error("数据库初始化失败：%s", exc)
        raise RuntimeError("数据库初始化失败，请检查数据目录权限或建表脚本") from exc


# ---------------------------------------------------------------------------
# 内部工具：字段清洗与预算解析
# ---------------------------------------------------------------------------
def _normalize_phone(phone_number: Any) -> str:
    """校验并归一化手机号。

    参数：
        phone_number: 任意类型的手机号输入。

    返回值：
        去掉首尾空白的手机号字符串。

    异常：
        手机号为空时抛 ``ValueError``。
    """
    phone = str(phone_number or "").strip()
    if not phone:
        raise ValueError("手机号不能为空")
    return phone


def _coerce_budget(value: Any) -> float | None:
    """把预算输入转换成浮点数（单位：元）。

    支持 ``20000`` / ``"20000"`` / ``"2万"`` / ``"1.5w"`` / ``"¥19800"`` / ``"两万"`` 等写法。

    参数：
        value: 预算原始值。

    返回值：
        预算金额（float，保留两位小数）；``None`` 或空字符串返回 ``None``。

    异常：
        无法解析或为负数时抛 ``ValueError``（由调用方决定忽略该字段）。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool 是 int 子类，但语义上不是合法预算，直接拒绝
        raise ValueError(f"非法预算值：{value!r}")
    if isinstance(value, (int, float)):
        amount = float(value)
    else:
        text = str(value).strip().replace(",", "").replace("，", "")
        if not text:
            return None
        match = _BUDGET_PATTERN.search(text)
        if match is None:
            # 再尝试中文数字 + 万（"两万"）
            chinese_match = _BUDGET_CHINESE_PATTERN.search(text)
            if chinese_match is None:
                raise ValueError(f"无法解析预算：{value!r}")
            amount = _BUDGET_CHINESE_DIGITS[chinese_match.group(1)] * 10000.0
        else:
            amount = float(match.group(1))
            unit = match.group(2) or ""
            if unit in {"万", "w", "W"}:
                amount *= 10000.0
            elif unit in {"k", "K", "千"}:
                amount *= 1000.0
            # unit 为「元」或不带单位时按元处理
            if unit == "":
                chinese_match = _BUDGET_CHINESE_PATTERN.search(text)
                if chinese_match is not None:
                    amount = _BUDGET_CHINESE_DIGITS[chinese_match.group(1)] * 10000.0
    if amount < 0:
        raise ValueError(f"预算不能为负数：{value!r}")
    return round(amount, 2)


def _normalize_stage(stage: Any) -> str | None:
    """校验咨询阶段取值。

    参数：
        stage: 待校验的阶段文本。

    返回值：
        合法时返回 新咨询 / 已了解 / 已报名；非法返回 ``None``。
    """
    text = str(stage or "").strip()
    return text if text in VALID_STAGES else None


def _clean_user_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """过滤并清洗用户画像字段（只接受 ALLOWED_USER_FIELDS 白名单）。

    参数：
        fields: 业务层传入的 ``**fields`` 字典。

    返回值：
        可直接用于 UPDATE 的字段字典（键顺序固定为白名单顺序）；
        非法字段被忽略并记录 warning，非法预算/阶段同样忽略。
    """
    cleaned: dict[str, Any] = {}
    for key in fields:
        if key not in ALLOWED_USER_FIELDS:
            logger.warning("忽略未授权的用户画像字段：%s", key)

    for key in ALLOWED_USER_FIELDS:
        if key not in fields:
            continue
        value = fields[key]
        if key == "budget":
            try:
                cleaned[key] = _coerce_budget(value)
            except ValueError as exc:
                logger.warning("忽略非法预算值 %r：%s", value, exc)
            continue
        if key == "stage":
            stage = _normalize_stage(value)
            if stage is None:
                logger.warning("忽略非法咨询阶段 %r，只允许 %s", value, "/".join(VALID_STAGES))
                continue
            cleaned[key] = stage
            continue
        if value is None:
            cleaned[key] = None
            continue
        text = str(value).strip()
        cleaned[key] = text or None
    return cleaned


def _apply_user_update(
    connection: sqlite3.Connection, phone_number: str, fields: dict[str, Any]
) -> None:
    """在已开启的事务中更新用户画像字段并刷新 updated_at。

    参数：
        connection: 已打开的数据库连接。
        phone_number: 用户手机号。
        fields: 已清洗的字段字典（键来自白名单，不存在注入风险）。

    返回值：
        无。
    """
    if not fields:
        return
    # 字段名来自 ALLOWED_USER_FIELDS 白名单，值使用参数占位符，安全
    assignments = ", ".join(f"{column} = ?" for column in fields)
    values = list(fields.values()) + [phone_number]
    connection.execute(
        f"UPDATE users SET {assignments}, updated_at = datetime('now', 'localtime') "
        "WHERE phone_number = ?",
        values,
    )


# ---------------------------------------------------------------------------
# 用户画像 CRUD
# ---------------------------------------------------------------------------
def create_user(phone_number: str, **fields: Any) -> dict:
    """创建用户画像；手机号已存在时更新传入的字段。

    参数：
        phone_number: 用户手机号（业务主键，不能为空）。
        **fields: 画像字段，只接受 ``name`` / ``intention_direction`` /
            ``budget`` / ``stage`` / ``notes``；budget 会转换为 float（"2万" → 20000.0）。

    返回值：
        写入后的完整用户行 dict（含 created_at / updated_at）。

    异常：
        手机号为空抛 ``ValueError``；数据库错误记录日志后抛 ``RuntimeError``。
    """
    phone = _normalize_phone(phone_number)
    cleaned = _clean_user_fields(fields)
    try:
        with _transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM users WHERE phone_number = ?", (phone,)
            ).fetchone()
            if exists is None:
                connection.execute("INSERT INTO users (phone_number) VALUES (?)", (phone,))
                logger.info("创建用户画像：%s", phone)
            if cleaned:
                _apply_user_update(connection, phone, cleaned)
                logger.info("更新用户画像：%s，字段=%s", phone, ",".join(cleaned))
    except sqlite3.Error as exc:
        logger.error("创建/更新用户画像失败（%s）：%s", phone, exc)
        raise RuntimeError("保存用户画像失败，请稍后重试") from exc

    user = get_user_info(phone)
    if user is None:
        logger.error("用户 %s 写入后查询结果为空", phone)
        return {"phone_number": phone}
    return user


def get_user_info(phone_number: str) -> dict | None:
    """按手机号查询用户画像。

    参数：
        phone_number: 用户手机号。

    返回值：
        完整用户行 dict（含 created_at / updated_at）；用户不存在或手机号为空返回 ``None``。
    """
    phone = str(phone_number or "").strip()
    if not phone:
        return None
    try:
        with _transaction() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE phone_number = ?", (phone,)
            ).fetchone()
        return dict(row) if row is not None else None
    except sqlite3.Error as exc:
        logger.warning("查询用户画像失败（%s）：%s", phone, exc)
        return None


def update_user_info(phone_number: str, **fields: Any) -> dict:
    """更新用户画像；用户不存在时自动创建。

    参数：
        phone_number: 用户手机号。
        **fields: 只接受 ``ALLOWED_USER_FIELDS``；budget 字符串（"2万"）会转为 float；
            stage 只允许 新咨询/已了解/已报名，非法值忽略并记录日志；未知字段忽略。

    返回值：
        更新后的完整用户行 dict（含 created_at / updated_at）。

    异常：
        手机号为空抛 ``ValueError``；数据库错误记录日志后抛 ``RuntimeError``。
    """
    phone = _normalize_phone(phone_number)
    cleaned = _clean_user_fields(fields)
    try:
        with _transaction() as connection:
            # INSERT OR IGNORE：不存在则先建空画像，保证「不存在则创建」语义
            connection.execute("INSERT OR IGNORE INTO users (phone_number) VALUES (?)", (phone,))
            if cleaned:
                _apply_user_update(connection, phone, cleaned)
                logger.info("局部更新用户画像：%s，字段=%s", phone, ",".join(cleaned))
    except sqlite3.Error as exc:
        logger.error("更新用户画像失败（%s）：%s", phone, exc)
        raise RuntimeError("更新用户画像失败，请稍后重试") from exc

    user = get_user_info(phone)
    if user is None:
        logger.error("用户 %s 更新后查询结果为空", phone)
        return {"phone_number": phone}
    return user


# ---------------------------------------------------------------------------
# 聊天历史
# ---------------------------------------------------------------------------
def fetch_chat_history(phone_number: str, limit: int | None = None) -> list[dict]:
    """按时间正序读取用户聊天历史。

    参数：
        phone_number: 用户手机号。
        limit: 只取最近 N 条（仍按时间正序）；``None`` 表示读取全部，``<=0`` 返回空列表。

    返回值：
        消息 dict 列表，元素含 ``id / phone_number / role / content / intent_code / created_at``；
        手机号为空或查询失败返回空列表（不抛异常）。
    """
    phone = str(phone_number or "").strip()
    if not phone:
        return []

    max_rows: int | None = None
    if limit is not None:
        try:
            max_rows = int(limit)
        except (TypeError, ValueError):
            logger.warning("fetch_chat_history 的 limit=%r 非法，按不限制处理", limit)
            max_rows = None
        else:
            if max_rows <= 0:
                return []

    base_select = (
        "SELECT id, phone_number, role, content, intent_code, created_at "
        "FROM chat_history WHERE phone_number = ?"
    )
    try:
        with _transaction() as connection:
            if max_rows is None:
                rows = connection.execute(
                    f"{base_select} ORDER BY created_at ASC, id ASC", (phone,)
                ).fetchall()
            else:
                # 先按倒序取最近 N 条，再在外层按正序返回，保证顺序语义
                rows = connection.execute(
                    f"SELECT * FROM ({base_select} ORDER BY created_at DESC, id DESC LIMIT ?) "
                    "ORDER BY created_at ASC, id ASC",
                    (phone, max_rows),
                ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        logger.warning("查询聊天历史失败（%s）：%s", phone, exc)
        return []


def insert_message(
    phone_number: str, role: str, content: str, intent_code: int | None = None
) -> int:
    """插入一条聊天消息。

    参数：
        phone_number: 用户手机号。
        role: 消息角色，只允许 ``user`` / ``assistant``。
        content: 消息内容。
        intent_code: 本轮意图编号（1/2/0），可为 ``None``。

    返回值：
        新消息的自增 id。

    异常：
        手机号为空、角色非法抛 ``ValueError``；数据库错误抛 ``RuntimeError``。
    """
    phone = _normalize_phone(phone_number)
    normalized_role = str(role or "").strip().lower()
    if normalized_role not in VALID_CHAT_ROLES:
        raise ValueError(f"非法消息角色：{role!r}，只允许 user 或 assistant")
    text = "" if content is None else str(content)

    code: int | None = None
    if intent_code is not None:
        try:
            code = int(intent_code)
        except (TypeError, ValueError):
            logger.warning("忽略非法意图编号：%r", intent_code)

    try:
        with _transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO chat_history (phone_number, role, content, intent_code) "
                "VALUES (?, ?, ?, ?)",
                (phone, normalized_role, text, code),
            )
            message_id = int(cursor.lastrowid or 0)
        logger.info("写入聊天消息：id=%s，phone=%s，role=%s", message_id, phone, normalized_role)
        return message_id
    except sqlite3.Error as exc:
        logger.error("写入聊天消息失败（%s）：%s", phone, exc)
        raise RuntimeError("保存聊天记录失败，请稍后重试") from exc


def update_chat_history(
    phone_number: str, query: str, response: str, intent_code: int | None = None
) -> None:
    """保存一轮完整对话（先写 user 消息，再写 assistant 消息）。

    参数：
        phone_number: 用户手机号。
        query: 用户提问。
        response: 客服/模型回答。
        intent_code: 本轮意图编号，会同时写入 user 与 assistant 两行，可为 ``None``。

    返回值：
        无。

    异常：
        手机号为空或消息写入失败时向上抛出（由调用方处理）。

    说明：
        两条消息都带同一个意图编号，读取端既可按「用户消息」也可按「客服消息」
        取到本轮意图，避免不同调用方对存储位置的假设不一致。
    """
    user_message_id = insert_message(phone_number, "user", query, intent_code)
    assistant_message_id = insert_message(phone_number, "assistant", response, intent_code)
    logger.info(
        "保存一轮对话：phone=%s，user_msg_id=%s，assistant_msg_id=%s",
        phone_number,
        user_message_id,
        assistant_message_id,
    )


# ---------------------------------------------------------------------------
# 只读 SQL 安全校验与执行
# ---------------------------------------------------------------------------
def _strip_sql_comments(sql: str) -> str:
    """去掉 SQL 中的 ``--`` 行注释与 ``/* */`` 块注释（保留字符串字面量）。

    参数：
        sql: 原始 SQL 文本。

    返回值：
        去注释后的 SQL 文本；块注释替换为一个空格，避免相邻 token 粘连。

    说明：
        解析时跟踪单引号/双引号字符串，避免把字符串里的 ``--``、``;`` 误判为注释或语句分隔符。
    """
    result: list[str] = []
    index = 0
    length = len(sql)
    in_single_quote = False
    in_double_quote = False

    while index < length:
        char = sql[index]

        if in_single_quote:
            result.append(char)
            if char == "'":
                # SQL 中的 '' 表示转义单引号，整体保留
                if index + 1 < length and sql[index + 1] == "'":
                    result.append(sql[index + 1])
                    index += 2
                    continue
                in_single_quote = False
            index += 1
            continue

        if in_double_quote:
            result.append(char)
            if char == '"':
                if index + 1 < length and sql[index + 1] == '"':
                    result.append(sql[index + 1])
                    index += 2
                    continue
                in_double_quote = False
            index += 1
            continue

        if char == "'":
            in_single_quote = True
            result.append(char)
            index += 1
            continue
        if char == '"':
            in_double_quote = True
            result.append(char)
            index += 1
            continue

        # 行注释：-- 直到行尾
        if char == "-" and index + 1 < length and sql[index + 1] == "-":
            index += 2
            while index < length and sql[index] not in "\r\n":
                index += 1
            continue

        # 块注释：/* ... */
        if char == "/" and index + 1 < length and sql[index + 1] == "*":
            index += 2
            while index + 1 < length and not (sql[index] == "*" and sql[index + 1] == "/"):
                index += 1
            index += 2
            result.append(" ")
            continue

        result.append(char)
        index += 1

    return "".join(result)


def _contains_semicolon_outside_literals(sql: str) -> bool:
    """判断 SQL 中是否存在字符串字面量之外的语句分隔符（分号）。

    参数：
        sql: 已去掉注释的 SQL 文本。

    返回值：
        存在字符串外的分号返回 True（说明可能是多语句），否则 False。
    """
    in_single_quote = False
    in_double_quote = False
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        if in_single_quote:
            if char == "'":
                if index + 1 < length and sql[index + 1] == "'":
                    index += 2
                    continue
                in_single_quote = False
            index += 1
            continue
        if in_double_quote:
            if char == '"':
                if index + 1 < length and sql[index + 1] == '"':
                    index += 2
                    continue
                in_double_quote = False
            index += 1
            continue
        if char == "'":
            in_single_quote = True
            index += 1
            continue
        if char == '"':
            in_double_quote = True
            index += 1
            continue
        if char == ";":
            return True
        index += 1
    return False


def _prepare_sql(sql: str) -> str:
    """规范化待执行 SQL：去注释、去首尾空白、去掉末尾分号。

    参数：
        sql: 已通过只读校验的 SQL。

    返回值：
        可直接交给 sqlite3 执行的单条 SQL 文本。
    """
    cleaned = _strip_sql_comments(sql).strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def is_readonly_sql(sql: str) -> bool:
    """判断是否为单条只读 SELECT/WITH 语句。

    参数：
        sql: 待判断的 SQL 文本。

    返回值：
        合法只读单语句返回 True；空串、多语句、以写操作/危险关键字开头
        （或 WITH 中夹带写操作）返回 False。

    说明：
        判断前先去掉 ``--`` / ``/* */`` 注释，再做 SELECT/WITH 前缀与关键字黑名单校验，
        并拒绝「分号后还有内容」的多语句形式。
    """
    if not isinstance(sql, str):
        return False

    body = _strip_sql_comments(sql).strip()
    if not body:
        return False

    # 允许末尾一个或多个分号（"SELECT 1;" 合法）
    while body.endswith(";"):
        body = body[:-1].rstrip()
    if not body:
        return False

    # 1) 必须是 SELECT 或 WITH 开头
    if _READONLY_SQL_PREFIX.match(body) is None:
        logger.warning("只读 SQL 校验失败：不是 SELECT/WITH 开头")
        return False

    # 2) 字符串字面量之外不允许再有分号（多语句）
    if _contains_semicolon_outside_literals(body):
        logger.warning("只读 SQL 校验失败：包含多条语句")
        return False

    # 3) 关键字黑名单：防止 WITH ... INSERT/UPDATE/DELETE 等写操作绕过前缀检查
    lowered = body.lower()
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered) is not None:
            logger.warning("只读 SQL 校验失败：包含危险关键字 %s", keyword)
            return False
    return True


def execute_sql(sql: str) -> list[dict]:
    """只读执行一条 SQL，返回结果行（最多 ``config.SQL_MAX_ROWS`` 行）。

    参数：
        sql: 用户/SQL Agent 提供的 SQL 文本。

    返回值：
        ``list[dict]``：查询结果，键为列名；最多 ``config.SQL_MAX_ROWS`` 行。
        执行期 SQL 错误（例如表不存在）记录日志后返回空列表，不抛异常。

    异常：
        非只读语句、多语句、写操作抛 ``ValueError(FORBIDDEN_SQL_MESSAGE)``。
    """
    if not is_readonly_sql(sql):
        logger.warning("拒绝执行非只读 SQL：%s", str(sql)[:200])
        raise ValueError(FORBIDDEN_SQL_MESSAGE)

    prepared_sql = _prepare_sql(sql)
    try:
        max_rows = int(config.SQL_MAX_ROWS)
    except (TypeError, ValueError):
        max_rows = 100
    max_rows = max(1, max_rows)

    try:
        with _transaction() as connection:
            cursor = connection.execute(prepared_sql)
            rows = cursor.fetchmany(max_rows)
        result = [dict(row) for row in rows]
        logger.info("执行只读 SQL 成功，返回 %d 行：%s", len(result), prepared_sql[:200])
        return result
    except sqlite3.Error as exc:
        logger.warning("只读 SQL 执行失败（%s）：%s", exc, prepared_sql[:200])
        return []


# ---------------------------------------------------------------------------
# 表结构说明与统计
# ---------------------------------------------------------------------------
def list_tables() -> list[str]:
    """列出数据库中的业务表名。

    参数：
        无。

    返回值：
        表名列表（按名称升序，排除 sqlite_ 系统表）；查询失败返回空列表。
    """
    try:
        with _transaction() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [str(row["name"]) for row in rows]
    except sqlite3.Error as exc:
        logger.warning("查询数据表列表失败：%s", exc)
        return []


def get_schema_description() -> str:
    """返回注入 SQL Agent 提示词的中文表结构说明。

    参数：
        无。

    返回值：
        包含 4 张表的表名、字段、类型、含义，以及每张表 2 行示例数据的文本；
        数据库尚未初始化时仍返回静态结构说明（不会抛异常）。
    """
    try:
        sample_blocks: list[str] = []
        for table in ("users", "chat_history", "courses", "career_info"):
            try:
                sample_rows = execute_sql(f"SELECT * FROM {table} LIMIT 2")
            except ValueError:  # pragma: no cover - 固定 SQL 不会触发
                sample_rows = []
            if sample_rows:
                formatted = "\n".join(
                    "  " + json.dumps(row, ensure_ascii=False, default=str) for row in sample_rows
                )
            else:
                formatted = "  （暂无数据）"
            sample_blocks.append(f"- {table} 示例数据：\n{formatted}")
        return _SCHEMA_STATIC_DESCRIPTION + "\n## 每表示例数据\n" + "\n".join(sample_blocks)
    except Exception as exc:  # pragma: no cover - 兜底，保证提示词永远可用
        logger.warning("生成表结构描述失败，返回静态说明：%s", exc)
        return _SCHEMA_STATIC_DESCRIPTION


def get_database_stats() -> dict[str, int]:
    """统计各业务表的数据量。

    参数：
        无。

    返回值：
        ``{"users": n, "chat_history": n, "courses": n, "career_info": n}``；
        单表统计失败时该表计为 0，不抛异常。
    """
    stats: dict[str, int] = {"users": 0, "chat_history": 0, "courses": 0, "career_info": 0}
    try:
        with _transaction() as connection:
            for table in stats:
                try:
                    row = connection.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()
                    stats[table] = int(row["total"]) if row is not None else 0
                except sqlite3.Error as exc:
                    logger.warning("统计表 %s 数据量失败：%s", table, exc)
    except sqlite3.Error as exc:
        logger.warning("连接数据库统计失败：%s", exc)
    return stats
