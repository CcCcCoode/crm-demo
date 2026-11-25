"""AI 能力层（llm 包）。

对外统一出口，方便上层这样使用::

    from llm import chat_completion, detect_intent_code, web_search

包含三个子模块：
- ``llm.client``：OpenAI 兼容客户端、非流式/流式调用、function calling 循环；
- ``llm.intent``：意图识别（LLM function calling + 关键词规则兜底）；
- ``llm.web_search``：联网搜索（SerpAPI / DuckDuckGo）与本地知识库兜底。

注意：本包不依赖 ``tools``，避免与工具层形成循环依赖。
"""

from __future__ import annotations

from llm.client import (
    LLMResponseError,
    LLMUnavailableError,
    chat_completion,
    extract_json,
    get_client,
    get_llm_status,
    is_llm_available,
    reset_client,
    run_streaming_tool_loop,
    run_tool_calling_loop,
    simple_chat,
    stream_completion,
)
from llm.intent import (
    DETECT_INTENT_TOOL_SCHEMA,
    INTENT_CAREER,
    INTENT_COURSE,
    INTENT_OTHER,
    VALID_INTENT_CODES,
    detect_intent_code,
    detect_intent_with_reason,
    keyword_intent,
)
from llm.web_search import (
    get_search_status,
    is_search_available,
    reset_search_status,
    web_search,
)

__all__ = [
    # client
    "LLMUnavailableError",
    "LLMResponseError",
    "get_client",
    "is_llm_available",
    "get_llm_status",
    "reset_client",
    "chat_completion",
    "run_tool_calling_loop",
    "stream_completion",
    "run_streaming_tool_loop",
    "simple_chat",
    "extract_json",
    # intent
    "INTENT_CAREER",
    "INTENT_COURSE",
    "INTENT_OTHER",
    "VALID_INTENT_CODES",
    "DETECT_INTENT_TOOL_SCHEMA",
    "keyword_intent",
    "detect_intent_code",
    "detect_intent_with_reason",
    # web search
    "web_search",
    "is_search_available",
    "get_search_status",
    "reset_search_status",
]
