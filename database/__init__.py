"""database 包：AI CRM 项目的 SQLite 数据层。

对外导出四个模块的能力：
- ``database.db``：连接管理、建表与示例数据、用户/聊天记录 CRUD、只读 SQL 执行；
- ``database.sql_agent``：自然语言 → SQL → 执行 → 中文总结（LLM 优先，规则兜底）。

注意：
- ``db`` 必须先于 ``sql_agent`` 导入：``sql_agent`` 在模块顶层依赖 ``database.db``，
  而 ``database.db`` 不依赖本包的其它模块，因此不会产生循环导入；
- 调用方可以按需直接 ``import database.db`` / ``import database.sql_agent``，
  也可以 ``from database import init_db`` 使用这里汇总的常用入口。
"""

from __future__ import annotations

# 先导出 db 层（sql_agent 依赖它，顺序不可颠倒）
from database.db import (
    ALLOWED_USER_FIELDS,
    FORBIDDEN_SQL_MESSAGE,
    create_user,
    execute_sql,
    fetch_chat_history,
    get_connection,
    get_database_stats,
    get_schema_description,
    get_user_info,
    init_db,
    insert_message,
    is_readonly_sql,
    list_tables,
    update_chat_history,
    update_user_info,
)

# 再导出 SQL Agent（其模块顶层会 from database.db import ...，此时 db 已加载）
from database.sql_agent import (
    SQL_TOOL_SCHEMA,
    answer_with_sql_agent,
    generate_sql,
    rule_based_answer,
)

__all__ = [
    # db
    "ALLOWED_USER_FIELDS",
    "FORBIDDEN_SQL_MESSAGE",
    "get_connection",
    "init_db",
    "create_user",
    "get_user_info",
    "update_user_info",
    "fetch_chat_history",
    "insert_message",
    "update_chat_history",
    "is_readonly_sql",
    "execute_sql",
    "get_schema_description",
    "list_tables",
    "get_database_stats",
    # sql_agent
    "SQL_TOOL_SCHEMA",
    "answer_with_sql_agent",
    "rule_based_answer",
    "generate_sql",
]
