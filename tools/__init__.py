"""工具层（tools 包）。

对外统一出口::

    from tools import TOOL_SCHEMAS, get_openai_tools, get_tool_names, execute_tool

6 个工具：``detect_intent``、``web_search``、``search_market_docs``、
``search_course_docs``、``query_database``、``save_chat_record``。

注意：``llm`` 包不会反向 import 本包，二者不会形成循环依赖。
"""

from __future__ import annotations

from tools.functions import (
    SAVE_CHAT_RECORD_TOOL_SCHEMA,
    SEARCH_COURSE_DOCS_TOOL_SCHEMA,
    SEARCH_MARKET_DOCS_TOOL_SCHEMA,
    TOOL_SCHEMAS,
    WEB_SEARCH_TOOL_SCHEMA,
    execute_tool,
    get_openai_tools,
    get_tool_names,
)

__all__ = [
    "TOOL_SCHEMAS",
    "WEB_SEARCH_TOOL_SCHEMA",
    "SEARCH_MARKET_DOCS_TOOL_SCHEMA",
    "SEARCH_COURSE_DOCS_TOOL_SCHEMA",
    "SAVE_CHAT_RECORD_TOOL_SCHEMA",
    "get_openai_tools",
    "get_tool_names",
    "execute_tool",
]
