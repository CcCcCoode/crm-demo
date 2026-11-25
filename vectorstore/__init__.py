"""向量层包（文本嵌入 + ChromaDB 持久化检索）。

模块划分：

- :mod:`vectorstore.embeddings`：文本 → 向量的统一入口，支持 bge / hash / openai
  三种后端，并保证任何后端不可用时都能自动降级为确定性 hash 向量；
- :mod:`vectorstore.store`：ChromaDB 持久化集合的增删改查、检索与示例数据初始化。

这里做轻量再导出，方便 ``from vectorstore import search`` 这类调用；
需要按契约使用时可继续 ``from vectorstore.store import search`` / ``from vectorstore import store``。
"""

from __future__ import annotations

from vectorstore import embeddings, store
from vectorstore.embeddings import (
    embed_query,
    embed_texts,
    get_embedding_dim,
    get_embedding_info,
    reset_embedding_backend,
)
from vectorstore.store import (
    VectorStoreError,
    add_documents,
    clear_collection,
    count_documents,
    delete_documents,
    get_collection_stats,
    get_documents,
    init_vectorstores,
    list_collections,
    reset_vectorstore,
    search,
    update_documents,
)

__all__ = [
    # 子模块
    "embeddings",
    "store",
    # 嵌入相关
    "embed_texts",
    "embed_query",
    "get_embedding_dim",
    "get_embedding_info",
    "reset_embedding_backend",
    # 存储相关
    "VectorStoreError",
    "init_vectorstores",
    "add_documents",
    "update_documents",
    "delete_documents",
    "search",
    "list_collections",
    "get_collection_stats",
    "count_documents",
    "get_documents",
    "clear_collection",
    "reset_vectorstore",
]
