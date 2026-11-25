"""向量层单元测试（嵌入后端 + ChromaDB 持久化存储）。

测试环境由 ``tests/conftest.py`` 统一设置：``EMBEDDING_PROVIDER=hash``、
``CRM_DATA_DIR`` 指向临时目录，因此本文件不会下载 BGE 模型，也不会污染真实数据。
每个涉及写入的用例都使用独立的临时集合名，并在用例结束后清理，避免相互影响。
"""

from __future__ import annotations

import math
import sys
import uuid

import pytest

from config import config, get_logger
from vectorstore import embeddings, store

logger = get_logger("crm.vectorstore.tests")


def _cosine(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度（用于验证 hash 嵌入的语义相似性）。

    Args:
        left: 第一个向量。
        right: 第二个向量。

    Returns:
        余弦相似度，范围约 [-1, 1]，越大越相似。
    """
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@pytest.fixture()
def temp_collection(monkeypatch):
    """创建一个临时集合用于写入类测试。

    通过 monkeypatch 把随机的临时集合名临时加入 ``config.COLLECTIONS`` 白名单，
    用例结束后删除该集合并恢复配置，因此不会影响 course / market 中的示例数据。

    Args:
        monkeypatch: pytest 提供的配置修改夹具。

    Yields:
        临时集合名（由 tmp + 10 位随机十六进制组成）。
    """
    collection_name = f"tmp{uuid.uuid4().hex[:10]}"
    monkeypatch.setattr(config, "COLLECTIONS", tuple(config.COLLECTIONS) + (collection_name,))
    yield collection_name
    try:
        store.clear_collection(collection_name)
    except Exception as exc:  # 清理失败只告警，避免掩盖用例本身的断言失败
        logger.warning("清理临时向量集合 %s 失败：%s", collection_name, exc)


# ---------------------------------------------------------------------------
# 初始化与示例数据
# ---------------------------------------------------------------------------
def test_init_vectorstores_idempotent() -> None:
    """验证 init_vectorstores 幂等：重复调用不会重复灌入示例文档。"""
    store.init_vectorstores()
    counts_before = {name: store.count_documents(name) for name in config.COLLECTIONS}
    assert all(count >= 6 for count in counts_before.values()), counts_before

    store.init_vectorstores()
    counts_after = {name: store.count_documents(name) for name in config.COLLECTIONS}
    assert counts_after == counts_before


def test_sample_documents_match_business_data() -> None:
    """验证示例文档数量、固定 id 与关键字段和 SQLite 课程/职业数据口径一致。"""
    course_docs = store.get_documents(config.COLLECTION_COURSE, limit=100)
    market_docs = store.get_documents(config.COLLECTION_MARKET, limit=100)
    assert len(course_docs) >= 6
    assert len(market_docs) >= 6

    course_by_id = {doc["id"]: doc for doc in course_docs}
    assert "course-001" in course_by_id
    course_001 = course_by_id["course-001"]
    assert course_001["metadata"]["price"] == 12800
    assert course_001["metadata"]["duration"] == "24 周"
    assert course_001["metadata"]["suitable_for"] == "零基础转行、在校学生"
    assert "适合零基础转行" in course_001["document"]

    market_by_id = {doc["id"]: doc for doc in market_docs}
    assert "market-001" in market_by_id
    market_001 = market_by_id["market-001"]
    assert market_001["metadata"]["salary_min"] == 15000
    assert market_001["metadata"]["salary_max"] == 28000
    assert "技能要求" in market_001["document"]
    assert "月薪" in market_001["document"]


def test_search_course_by_chinese_query() -> None:
    """验收场景：查询「Python 课程多少钱」能检索到课程文档。"""
    results = store.search(config.COLLECTION_COURSE, "Python 课程多少钱", top_k=3)
    assert results, "查询课程价格应至少返回一条课程文档"
    assert any("Python" in item["document"] for item in results)
    assert all(item["metadata"].get("price") for item in results)
    distances = [item["distance"] for item in results]
    assert distances == sorted(distances)


# ---------------------------------------------------------------------------
# 嵌入后端
# ---------------------------------------------------------------------------
def test_get_embedding_info_structure() -> None:
    """验证 get_embedding_info 返回完整字段，且测试环境使用 hash 后端。"""
    info = embeddings.get_embedding_info()
    assert set(info) == {"configured_provider", "provider", "model", "dim", "backend", "ready", "detail"}
    assert info["provider"] == "hash"
    assert info["backend"] == "hash"
    assert info["ready"] is True
    assert info["dim"] == config.EMBEDDING_DIM
    assert embeddings.get_embedding_dim() == config.EMBEDDING_DIM


def test_embed_texts_deterministic() -> None:
    """验证 hash 嵌入确定性：同文本多次调用、重置后端后仍得到完全相同的向量。"""
    text = "Python 全栈开发课程，学费 12800 元，学习周期 24 周"
    first = embeddings.embed_texts([text])[0]
    second = embeddings.embed_texts([text])[0]
    assert first == second

    embeddings.reset_embedding_backend()
    try:
        third = embeddings.embed_texts([text])[0]
        assert third == first
    finally:
        embeddings.reset_embedding_backend()

    other = embeddings.embed_texts(["今天天气很好，适合去公园散步"])[0]
    assert other != first


def test_embed_query_matches_batch_item() -> None:
    """验证 embed_query 单条结果与 embed_texts 批处理结果一致，且向量已归一化。"""
    text = "Python 课程多少钱"
    vector = embeddings.embed_query(text)
    batch_vector = embeddings.embed_texts([text])[0]
    assert vector == batch_vector
    assert len(vector) == embeddings.get_embedding_dim()
    norm = math.sqrt(sum(value * value for value in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_embed_texts_empty_and_invalid_input() -> None:
    """验证空批次返回空列表，非法入参抛 ValueError。"""
    assert embeddings.embed_texts([]) == []
    with pytest.raises(ValueError):
        embeddings.embed_texts(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        embeddings.embed_texts("单条字符串")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        embeddings.embed_texts(["合法文本", 123])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        embeddings.embed_query(123)  # type: ignore[arg-type]


def test_hash_embedding_chinese_similarity() -> None:
    """验证 hash 嵌入对相近中文语义的余弦相似度高于无关文本。"""
    query = embeddings.embed_query("Python 课程多少钱")
    related = embeddings.embed_texts(["《Python 全栈开发实战》课程，学费 12800 元，学习周期 24 周"])[0]
    unrelated = embeddings.embed_texts(["今天天气晴朗，适合出门散步和晒太阳"])[0]
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_bge_failure_degrades_to_hash(monkeypatch) -> None:
    """验证 BGE 不可用（模拟依赖缺失）时自动降级 hash 且不抛异常。"""
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embeddings.reset_embedding_backend()
    try:
        vectors = embeddings.embed_texts(["Python 课程多少钱"])
        assert len(vectors) == 1
        assert len(vectors[0]) == config.EMBEDDING_DIM

        info = embeddings.get_embedding_info()
        assert info["configured_provider"] == "bge"
        assert info["provider"] == "hash"
        assert info["ready"] is True
        assert "降级" in info["detail"]
    finally:
        embeddings.reset_embedding_backend()


def test_openai_without_key_degrades_to_hash(monkeypatch) -> None:
    """验证配置 openai 后端但未配置 key 时自动降级 hash 且不抛异常。"""
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_EMBEDDING_API_KEY", "")
    embeddings.reset_embedding_backend()
    try:
        vector = embeddings.embed_query("Python 课程多少钱")
        assert len(vector) == config.EMBEDDING_DIM
        info = embeddings.get_embedding_info()
        assert info["configured_provider"] == "openai"
        assert info["provider"] == "hash"
    finally:
        embeddings.reset_embedding_backend()


# ---------------------------------------------------------------------------
# 增删改查与检索
# ---------------------------------------------------------------------------
def test_add_and_search_roundtrip(temp_collection: str) -> None:
    """验证新增文档后可按语义检索到，且返回结构、metadata 清洗符合契约。"""
    document = "量子计算入门课程，价格 9999 元，适合零基础学员"
    doc_id = f"{temp_collection}-doc-1"
    returned_ids = store.add_documents(
        temp_collection,
        [document],
        [{"name": "量子计算入门", "price": 9999, "tags": ["AI", "量子"]}],
        [doc_id],
    )
    assert returned_ids == [doc_id]
    assert store.count_documents(temp_collection) == 1

    results = store.search(temp_collection, "量子计算课程价格", top_k=3)
    assert len(results) == 1
    item = results[0]
    assert set(item) == {"id", "document", "metadata", "distance"}
    assert item["id"] == doc_id
    assert item["document"] == document
    assert isinstance(item["distance"], float)
    # 列表型 metadata 会被转成「、」拼接的字符串（Chroma 只接受标量）
    assert item["metadata"]["tags"] == "AI、量子"
    assert item["metadata"]["price"] == 9999


def test_add_with_empty_metadatas_omits_metadata(temp_collection: str) -> None:
    """验证 metadata 清洗后全部为空时整体不传 metadatas，写入仍成功（Chroma 不接受空字典）。"""
    ids = store.add_documents(
        temp_collection,
        ["无元数据文档一，价格 100 元", "无元数据文档二，价格 200 元"],
        [{"empty": None}, {}],
    )
    assert len(ids) == 2
    assert store.count_documents(temp_collection) == 2
    documents = store.get_documents(temp_collection, limit=10)
    assert all(doc["metadata"] == {} for doc in documents)


def test_search_empty_collection_returns_empty(temp_collection: str) -> None:
    """验证空集合检索返回空列表而不是抛异常。"""
    assert store.count_documents(temp_collection) == 0
    assert store.search(temp_collection, "任意查询", top_k=3) == []


def test_search_top_k_and_distance_order(temp_collection: str) -> None:
    """验证 top_k 限制生效、结果按 distance 升序，top_k 大于总数时返回全部。"""
    documents = [
        "量子计算入门课程，价格 9999 元",
        "Python 数据分析课程，价格 9800 元",
        "Vue3 前端开发课程，价格 8800 元",
        "大模型微调实战课程，价格 19800 元",
        "自动化办公训练营，价格 3980 元",
    ]
    store.add_documents(temp_collection, documents)
    limited = store.search(temp_collection, "课程价格", top_k=3)
    assert len(limited) == 3
    distances = [item["distance"] for item in limited]
    assert distances == sorted(distances)

    all_results = store.search(temp_collection, "课程价格", top_k=10)
    assert len(all_results) == len(documents)
    assert [item["distance"] for item in all_results] == sorted(item["distance"] for item in all_results)


def test_update_documents_refreshes_embedding(temp_collection: str) -> None:
    """验收场景：update 后旧文本不再是最佳匹配，新文本可被检索到，metadata 保留。"""
    old_text = "量子计算入门课程，学费 9999 元，适合零基础学员"
    target_id = f"{temp_collection}-upd"
    distractors = [
        (f"{temp_collection}-dis-1", "量子计算基础课程，学费 8888 元，讲解量子比特原理"),
        (f"{temp_collection}-dis-2", "量子计算进阶课程，学费 9999 元，含算法实战项目"),
        (f"{temp_collection}-dis-3", "量子计算应用课程，学费 7777 元，适合有 Python 基础"),
    ]
    other_docs = ["今天天气晴朗，适合出门散步", "数据分析师月薪 12000 至 25000 元"]
    store.add_documents(
        temp_collection,
        [old_text] + [document for _, document in distractors] + other_docs,
        [{"price": 9999}] + [{"price": 8888}, {"price": 9999}, {"price": 7777}, {"note": "其他文档"}, {"note": "其他文档"}],
        [target_id] + [doc_id for doc_id, _ in distractors] + [f"{temp_collection}-other-{index}" for index in range(2)],
    )
    # 更新前：用旧文本自身检索，最佳匹配就是该文档
    before = store.search(temp_collection, old_text, top_k=1)
    assert before and before[0]["id"] == target_id

    new_text = "脑机接口进阶课程，价格 18888 元，适合进阶学员"
    store.update_documents(temp_collection, [target_id], [new_text])

    # 更新后：旧文本的最佳匹配不再是该文档（旧向量已被新文本向量替换）
    after = store.search(temp_collection, old_text, top_k=1)
    assert after and after[0]["id"] != target_id
    # 新文本可被检索到，且文档内容已更新
    new_hits = store.search(temp_collection, new_text, top_k=1)
    assert new_hits and new_hits[0]["id"] == target_id
    target_doc = next(doc for doc in store.get_documents(temp_collection, limit=20) if doc["id"] == target_id)
    assert target_doc["document"] == new_text
    # 未传 metadatas 时，原有 metadata 必须保留
    assert target_doc["metadata"]["price"] == 9999


def test_delete_documents_removes_from_search(temp_collection: str) -> None:
    """验收场景：delete 后文档检索不到，且集合文档数正确减少。"""
    keep_id = f"{temp_collection}-keep"
    remove_id = f"{temp_collection}-remove"
    store.add_documents(
        temp_collection,
        ["Python 全栈开发课程，学费 12800 元", "区块链开发课程，学费 12000 元"],
        ids=[keep_id, remove_id],
    )
    assert store.count_documents(temp_collection) == 2

    store.delete_documents(temp_collection, [remove_id])
    assert store.count_documents(temp_collection) == 1

    hits = store.search(temp_collection, "区块链开发课程", top_k=10)
    assert all(item["id"] != remove_id for item in hits)
    remaining = store.get_documents(temp_collection, limit=10)
    assert [doc["id"] for doc in remaining] == [keep_id]


def test_update_and_delete_missing_ids_are_ignored(temp_collection: str) -> None:
    """验证更新/删除不存在的 id 不抛异常（与 ChromaDB 静默行为对齐）。"""
    store.update_documents(temp_collection, ["not-exists-id"], ["任意新文本"])
    store.delete_documents(temp_collection, ["not-exists-id"])
    assert store.count_documents(temp_collection) == 0


def test_metadata_none_values_are_filtered(temp_collection: str) -> None:
    """验证 metadata 中的 None 被过滤、列表被转字符串、标量正常保存。"""
    doc_id = f"{temp_collection}-meta"
    store.add_documents(
        temp_collection,
        ["测试文档，用于验证 metadata 清洗"],
        [{"empty": None, "count": 2, "tags": ["a", "b"], "ok": True, "ratio": 3.5, "nested": {"x": 1}}],
        [doc_id],
    )
    doc = next(item for item in store.get_documents(temp_collection, limit=10) if item["id"] == doc_id)
    metadata = doc["metadata"]
    assert "empty" not in metadata
    assert metadata["count"] == 2
    assert metadata["tags"] == "a、b"
    assert metadata["ok"] is True
    assert metadata["ratio"] == 3.5
    assert metadata["nested"] == "{'x': 1}"


def test_auto_generated_ids(temp_collection: str) -> None:
    """验证不传 ids 时自动生成带集合名前缀的唯一 id。"""
    returned_ids = store.add_documents(temp_collection, ["第一段文档", "第二段文档"])
    assert len(returned_ids) == 2
    assert len(set(returned_ids)) == 2
    assert all(item.startswith(f"{temp_collection}-") for item in returned_ids)
    assert store.count_documents(temp_collection) == 2


def test_reset_vectorstore_keeps_persisted_data(temp_collection: str) -> None:
    """验证 reset_vectorstore 只重置客户端缓存，数据仍在 Chroma 持久化目录中。"""
    doc_id = f"{temp_collection}-persist"
    store.add_documents(temp_collection, ["需要持久化的课程文档，价格 6666 元"], ids=[doc_id])
    store.reset_vectorstore()
    assert store.count_documents(temp_collection) == 1
    hits = store.search(temp_collection, "持久化课程文档价格", top_k=1)
    assert hits and hits[0]["id"] == doc_id


# ---------------------------------------------------------------------------
# 参数校验与集合管理
# ---------------------------------------------------------------------------
def test_invalid_collection_raises_value_error() -> None:
    """验证所有接口对非法集合名统一抛 ValueError（对应 HTTP 400）。"""
    invalid_name = "unknown_collection"
    calls = [
        lambda: store.add_documents(invalid_name, ["文档"]),
        lambda: store.update_documents(invalid_name, ["id"], ["文档"]),
        lambda: store.delete_documents(invalid_name, ["id"]),
        lambda: store.search(invalid_name, "查询"),
        lambda: store.count_documents(invalid_name),
        lambda: store.get_documents(invalid_name),
        lambda: store.clear_collection(invalid_name),
        lambda: store.add_documents(None, ["文档"]),  # type: ignore[arg-type]
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_invalid_arguments_raise_value_error() -> None:
    """验证文档/ids/metadatas/top_k/limit 等非法参数统一抛 ValueError。"""
    course = config.COLLECTION_COURSE
    with pytest.raises(ValueError):
        store.add_documents(course, [])
    with pytest.raises(ValueError):
        store.add_documents(course, "单条文档")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.add_documents(course, [123])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        store.add_documents(course, ["文档一", "文档二"], ids=["only-one"])
    with pytest.raises(ValueError):
        store.add_documents(course, ["文档一", "文档二"], metadatas=[{"only": "one"}])
    with pytest.raises(ValueError):
        store.add_documents(course, ["文档一", "文档二"], ids=["same-id", "same-id"])
    with pytest.raises(ValueError):
        store.add_documents(course, ["文档一", "文档二"], metadatas=[{"empty": None}, {"valid": 1}])
    with pytest.raises(ValueError):
        store.update_documents(course, ["only-id"], ["文档一", "文档二"])
    with pytest.raises(ValueError):
        store.update_documents(course, [], ["文档一"])
    with pytest.raises(ValueError):
        store.delete_documents(course, [])
    with pytest.raises(ValueError):
        store.search(course, "   ")
    with pytest.raises(ValueError):
        store.search(course, "Python", top_k=0)
    with pytest.raises(ValueError):
        store.search(course, "Python", top_k=-1)
    with pytest.raises(ValueError):
        store.get_documents(course, limit=0)
    with pytest.raises(ValueError):
        store.get_documents(course, limit=-5)


def test_list_collections_and_stats() -> None:
    """验证集合列表与统计接口结构一致，且包含 market / course。"""
    store.init_vectorstores()
    collections = store.list_collections()
    names = {item["name"] for item in collections}
    assert {config.COLLECTION_MARKET, config.COLLECTION_COURSE} <= names
    for item in collections:
        assert set(item) == {"name", "count"}
        assert item["count"] == store.count_documents(item["name"])

    stats = store.get_collection_stats()
    assert stats == collections


def test_clear_collection_then_reinit() -> None:
    """验证 clear_collection 清空集合后，init_vectorstores 能重新灌回示例文档。"""
    store.clear_collection(config.COLLECTION_COURSE)
    try:
        assert store.count_documents(config.COLLECTION_COURSE) == 0
        assert store.search(config.COLLECTION_COURSE, "Python 课程多少钱", top_k=3) == []

        store.init_vectorstores()
        assert store.count_documents(config.COLLECTION_COURSE) >= 6
        course_ids = {doc["id"] for doc in store.get_documents(config.COLLECTION_COURSE, limit=100)}
        assert "course-001" in course_ids
    finally:
        # 无论断言是否失败，都把示例数据恢复到可用状态，避免影响后续用例/测试文件
        store.init_vectorstores()
