"""文本嵌入模块（向量层「文本 → 向量」的统一入口）。

支持三种后端，由 ``config.EMBEDDING_PROVIDER`` 决定：

1. ``bge``（默认）：懒加载 ``sentence_transformers.SentenceTransformer``，
   使用中文 BGE 模型生成语义向量；首次加载失败（缺少依赖 / 无法联网 / 模型下载失败）
   时自动降级为 hash 后端，只记录 warning，绝不让上层接口崩溃。
2. ``hash``：纯标准库实现的确定性特征哈希向量（字符 1~3-gram + 英文切词 + sha1 散列），
   不需要模型和网络，同文本必得同向量，相近文本的余弦相似度更高；
   该后端同时是 bge / openai 不可用时的安全兜底，测试环境由 conftest 强制使用。
3. ``openai``：配置了 ``OPENAI_EMBEDDING_API_KEY`` 时调用 OpenAI 兼容的
   ``/v1/embeddings`` 接口；未配置 key 或调用失败时降级为 hash 后端。

之所以这样设计：DeepSeek 官方未提供 embedding 接口，验收机器也不能下载大模型，
因此「真实语义模型」与「确定性兜底」必须解耦，且后端切换对上层透明。
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from typing import Any

from config import config, get_logger

logger: logging.Logger = get_logger("crm.vectorstore")

# 支持的嵌入后端标识，与 config.EMBEDDING_PROVIDER 的取值保持一致
_SUPPORTED_PROVIDERS: tuple[str, ...] = ("bge", "hash", "openai")
# hash 后端在文本没有任何有效字符特征时使用的占位特征，保证空文本也能得到非零向量
_EMPTY_TEXT_FEATURE = "empty"
# 英文单词切分规则：字母开头，允许数字、下划线及常见技术符号（如 python3、c++、c#、node.js）
_ENGLISH_WORD_RE = re.compile(r"[a-z][a-z0-9_+#.\-]*")
# 连续空白正则：中文场景先去掉全部空白，再做字符 n-gram
_WHITESPACE_RE = re.compile(r"\s+")
# hash 特征权重：单字权重低、二元/三元权重高，英文整词权重最高（整词语义更强）
_WEIGHT_CHAR1 = 0.5
_WEIGHT_CHAR2 = 1.0
_WEIGHT_CHAR3 = 1.0
_WEIGHT_WORD = 2.0
# 默认向量维度（config.EMBEDDING_DIM 缺失或非法时使用）
_DEFAULT_DIM = 512

# 保护后端状态与懒加载实例的线程锁（嵌入调用可能来自 Flask 多线程）
_STATE_LOCK = threading.RLock()
# 当前生效后端的元信息缓存，为空表示尚未解析
_STATE: dict[str, Any] = {}
# 懒加载的 SentenceTransformer 实例（bge 后端）
_BGE_MODEL: Any | None = None
# 懒加载的 OpenAI 兼容客户端（openai 后端）
_OPENAI_CLIENT: Any | None = None


def _empty_state() -> dict[str, Any]:
    """构造一份空的后端状态字典。

    Returns:
        包含 configured_provider / provider / model / dim / backend / ready / detail
        七个字段、值均为占位值的字典。
    """
    return {
        "configured_provider": "",
        "provider": "",
        "model": "",
        "dim": 0,
        "backend": "",
        "ready": False,
        "detail": "",
    }


def _configured_provider() -> str:
    """读取并归一化配置中的嵌入后端名称。

    配置值大小写、首尾空白都被容忍；空值或不受支持的值回退为 ``hash``，
    避免因为环境变量写错导致整个向量层不可用。

    Returns:
        归一化后的后端名称：bge / hash / openai 之一。
    """
    raw = getattr(config, "EMBEDDING_PROVIDER", "") or ""
    provider = str(raw).strip().lower()
    if provider in _SUPPORTED_PROVIDERS:
        return provider
    if provider:
        logger.warning(
            "config.EMBEDDING_PROVIDER=%r 不受支持，已回退为 hash 后端（可选：%s）",
            raw,
            "/".join(_SUPPORTED_PROVIDERS),
        )
    return "hash"


def _configured_dim() -> int:
    """读取配置中的向量维度。

    Returns:
        正整数维度；配置缺失、非整数或非正数时回退为 512 并在日志中告警。
    """
    raw = getattr(config, "EMBEDDING_DIM", _DEFAULT_DIM)
    try:
        dim = int(raw)
    except (TypeError, ValueError):
        logger.warning("config.EMBEDDING_DIM=%r 不是合法整数，回退 %d 维", raw, _DEFAULT_DIM)
        return _DEFAULT_DIM
    if dim <= 0:
        logger.warning("config.EMBEDDING_DIM=%r 必须为正整数，回退 %d 维", raw, _DEFAULT_DIM)
        return _DEFAULT_DIM
    return dim


def _activate_hash(reason: str) -> None:
    """把当前生效后端切换为 hash，并记录切换原因。

    Args:
        reason: 触发降级的原因（写入状态 detail 与日志，便于线上排查）。

    Returns:
        None。
    """
    dim = _configured_dim()
    with _STATE_LOCK:
        _STATE.update(
            {
                "configured_provider": _configured_provider(),
                "provider": "hash",
                "model": f"sha1-hash-{dim}d",
                "dim": dim,
                "backend": "hash",
                "ready": True,
                "detail": reason,
            }
        )
    logger.info("嵌入后端切换为 hash（dim=%d）：%s", dim, reason)


def _load_bge_model() -> Any:
    """懒加载 BGE 句向量模型（sentence-transformers）。

    延迟导入 sentence_transformers，避免未安装该依赖的环境在 import 阶段就失败；
    模型实例会缓存在模块级变量中，后续调用直接复用。

    Returns:
        已加载的 SentenceTransformer 实例。

    Raises:
        Exception: 依赖缺失、模型下载失败或磁盘/设备异常时由底层抛出，
            由 :func:`_resolve_backend` / :func:`embed_texts` 捕获并降级 hash。
    """
    global _BGE_MODEL
    if _BGE_MODEL is not None:
        return _BGE_MODEL
    # 尽量在导入 sentence_transformers / huggingface_hub 之前设置镜像端点，
    # config 导入时通常已设置，这里再兜底一次以支持运行期改配置。
    hf_endpoint = str(getattr(config, "HF_ENDPOINT", "") or "").strip()
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - 延迟导入重依赖

    model_name = str(getattr(config, "EMBEDDING_MODEL", "") or "").strip()
    if not model_name:
        raise RuntimeError("config.EMBEDDING_MODEL 未配置，无法加载 BGE 模型")
    device = str(getattr(config, "EMBEDDING_DEVICE", "") or "cpu").strip() or "cpu"
    logger.info("正在懒加载 BGE 嵌入模型：%s（device=%s）……", model_name, device)
    _BGE_MODEL = SentenceTransformer(model_name, device=device)
    logger.info("BGE 嵌入模型加载完成：%s（device=%s）", model_name, device)
    return _BGE_MODEL


def _encode_with_bge(texts: list[str]) -> list[list[float]]:
    """用 BGE 模型把一批文本编码为归一化向量。

    Args:
        texts: 非空文本列表（调用方已保证元素是 str）。

    Returns:
        与 ``texts`` 等长的向量列表，每个向量为 float 列表且已做 L2 归一化。
    """
    model = _load_bge_model()
    batch_size = int(getattr(config, "EMBEDDING_BATCH_SIZE", 16) or 16)
    if batch_size <= 0:
        batch_size = 16
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vectors = [[float(value) for value in row] for row in matrix]
    if len(vectors) != len(texts):
        raise RuntimeError(f"BGE 编码数量不一致：期望 {len(texts)} 条，实际 {len(vectors)} 条")
    return vectors


def _encode_with_openai(texts: list[str]) -> list[list[float]]:
    """调用 OpenAI 兼容的 embedding 接口编码一批文本。

    Args:
        texts: 非空文本列表（调用方已保证元素是 str）。

    Returns:
        与 ``texts`` 顺序一致的向量列表。

    Raises:
        RuntimeError: 客户端未初始化或接口返回数量不一致时抛出。
        Exception: 网络 / 鉴权 / 限流等底层异常由 SDK 抛出，由调用方降级处理。
    """
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        raise RuntimeError("OpenAI 兼容 embedding 客户端尚未初始化")
    model_name = str(getattr(config, "OPENAI_EMBEDDING_MODEL", "") or "").strip()
    response = _OPENAI_CLIENT.embeddings.create(model=model_name, input=texts)
    # 兼容接口不保证 data 的顺序，按 index 排序后再取向量
    items = sorted(response.data, key=lambda item: int(getattr(item, "index", 0) or 0))
    vectors = [[float(value) for value in item.embedding] for item in items]
    if len(vectors) != len(texts):
        raise RuntimeError(f"OpenAI 嵌入数量不一致：期望 {len(texts)} 条，实际 {len(vectors)} 条")
    actual_dim = len(vectors[0]) if vectors else 0
    with _STATE_LOCK:
        state_dim = int(_STATE.get("dim") or 0)
        if actual_dim > 0 and state_dim and actual_dim != state_dim:
            logger.warning("OpenAI 嵌入实际维度 %d 与配置维度 %d 不一致，按实际维度记录", actual_dim, state_dim)
        _STATE["dim"] = actual_dim
    return vectors


def _resolve_backend() -> None:
    """解析并缓存当前生效的嵌入后端（调用方需持有 _STATE_LOCK）。"""
    global _OPENAI_CLIENT
    provider = _configured_provider()
    _STATE["configured_provider"] = provider
    if provider == "hash":
        _activate_hash("按配置使用确定性 hash 嵌入后端（无模型、无网络依赖）")
        return
    if provider == "bge":
        try:
            model = _load_bge_model()
            dim_raw = model.get_sentence_embedding_dimension()
            dim = int(dim_raw) if dim_raw else _configured_dim()
            if dim <= 0:
                dim = _configured_dim()
            _STATE.update(
                {
                    "provider": "bge",
                    "model": str(getattr(config, "EMBEDDING_MODEL", "") or ""),
                    "dim": dim,
                    "backend": "sentence-transformers",
                    "ready": True,
                    "detail": "BGE 句向量模型已加载",
                }
            )
            logger.info("嵌入后端就绪：bge，模型=%s，dim=%d", _STATE["model"], dim)
        except Exception as exc:  # 依赖缺失 / 下载失败 / 设备异常等都要降级而不是崩溃
            logger.warning("BGE 嵌入模型加载失败，自动降级 hash 后端：%s", exc)
            _activate_hash(f"bge 加载失败，已自动降级 hash：{exc}")
        return
    # provider == "openai"
    api_key = str(getattr(config, "OPENAI_EMBEDDING_API_KEY", "") or "").strip()
    if not api_key:
        _activate_hash("EMBEDDING_PROVIDER=openai 但未配置 OPENAI_EMBEDDING_API_KEY，已降级 hash")
        return
    try:
        from openai import OpenAI  # noqa: PLC0415 - 延迟导入可选依赖

        model_name = str(getattr(config, "OPENAI_EMBEDDING_MODEL", "") or "").strip()
        if not model_name:
            raise RuntimeError("config.OPENAI_EMBEDDING_MODEL 未配置")
        base_url = str(getattr(config, "OPENAI_EMBEDDING_BASE_URL", "") or "").strip() or None
        timeout = float(getattr(config, "LLM_TIMEOUT", 60.0) or 60.0)
        _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    except Exception as exc:
        logger.warning("OpenAI 兼容 embedding 客户端初始化失败，自动降级 hash 后端：%s", exc)
        _activate_hash(f"openai 客户端初始化失败，已自动降级 hash：{exc}")
        return
    _STATE.update(
        {
            "provider": "openai",
            "model": str(getattr(config, "OPENAI_EMBEDDING_MODEL", "") or ""),
            "dim": _configured_dim(),
            "backend": "openai",
            "ready": True,
            "detail": "使用 OpenAI 兼容 embedding 接口",
        }
    )
    logger.info("嵌入后端就绪：openai，模型=%s", _STATE["model"])


def _ensure_backend() -> None:
    """确保后端已解析（未解析时触发一次解析，结果缓存）。"""
    with _STATE_LOCK:
        if _STATE.get("provider"):
            return
        _resolve_backend()


def _peek_state() -> dict[str, Any]:
    """读取当前后端状态，但不触发 BGE 模型下载等昂贵操作。

    对无需模型的 hash 后端会立即确定状态；对 bge 后端在未加载时返回
    ``ready=False`` 的预览状态，避免 /health 或启动日志意外拉起模型下载。

    Returns:
        后端状态字典（副本）。
    """
    with _STATE_LOCK:
        if _STATE.get("provider"):
            return dict(_STATE)
        configured = _configured_provider()
        if configured == "hash":
            _activate_hash("按配置使用确定性 hash 嵌入后端（无模型、无网络依赖）")
            return dict(_STATE)
        if configured == "openai" and not str(getattr(config, "OPENAI_EMBEDDING_API_KEY", "") or "").strip():
            _activate_hash("EMBEDDING_PROVIDER=openai 但未配置 OPENAI_EMBEDDING_API_KEY，已降级 hash")
            return dict(_STATE)
        if configured == "bge":
            loaded = _BGE_MODEL is not None
            return {
                "configured_provider": configured,
                "provider": "bge",
                "model": str(getattr(config, "EMBEDDING_MODEL", "") or ""),
                "dim": _configured_dim(),
                "backend": "sentence-transformers",
                "ready": loaded,
                "detail": "BGE 模型已加载" if loaded else "BGE 模型尚未加载：首次嵌入时懒加载，失败会自动降级 hash",
            }
        return {
            "configured_provider": configured,
            "provider": "openai",
            "model": str(getattr(config, "OPENAI_EMBEDDING_MODEL", "") or ""),
            "dim": _configured_dim(),
            "backend": "openai",
            "ready": True,
            "detail": "使用 OpenAI 兼容 embedding 接口（首次调用时创建客户端）",
        }


def _validate_texts(texts: list[str]) -> list[str]:
    """校验嵌入输入，返回元素列表。

    Args:
        texts: 待校验的文本集合，按契约必须是字符串列表（允许空列表）。

    Returns:
        校验通过后的文本列表。

    Raises:
        ValueError: 输入不是列表 / 可迭代对象，或其中存在非字符串元素时抛出。
    """
    if texts is None:
        raise ValueError("texts 不能为 None；空列表表示不嵌入任何文本")
    if isinstance(texts, str):
        raise ValueError("texts 必须是字符串列表，单条文本请使用 embed_query()")
    try:
        items = list(texts)
    except TypeError as exc:
        raise ValueError("texts 必须是可迭代的字符串列表") from exc
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ValueError(f"texts 第 {index} 条不是字符串（实际类型：{type(item).__name__}）")
    return items


def _hash_vector(text: str, dim: int) -> list[float]:
    """用确定性特征哈希把单条文本编码成 dim 维归一化向量。

    特征来源：小写去空白后的字符 1~3-gram + 简单英文切词；每个特征经 sha1
    映射到某个维度并按符号累加，最后做 L2 归一化。完全基于 hashlib，
    不受 Python 进程级字符串哈希随机化影响，因此多进程结果一致。

    Args:
        text: 待编码文本。
        dim: 目标向量维度（正整数）。

    Returns:
        dim 维 float 列表，L2 范数为 1（文本无任何特征时使用占位特征，仍非零）。
    """
    normalized = text.strip().lower()
    compact = _WHITESPACE_RE.sub("", normalized)
    features: dict[str, float] = {}

    def _add_feature(feature: str, weight: float) -> None:
        """累加一个特征权重（同一特征出现多次则权重叠加）。"""
        features[feature] = features.get(feature, 0.0) + weight

    length = len(compact)
    for index in range(length):
        _add_feature(f"c1:{compact[index]}", _WEIGHT_CHAR1)
        if index + 2 <= length:
            _add_feature(f"c2:{compact[index:index + 2]}", _WEIGHT_CHAR2)
        if index + 3 <= length:
            _add_feature(f"c3:{compact[index:index + 3]}", _WEIGHT_CHAR3)

    for word in _ENGLISH_WORD_RE.findall(normalized):
        _add_feature(f"w:{word}", _WEIGHT_WORD)

    if not features:
        _add_feature(f"c1:{_EMPTY_TEXT_FEATURE}", _WEIGHT_CHAR1)

    vector = [0.0] * dim
    # 按特征名排序后累加，保证浮点加法顺序固定，同文本得到逐位一致的向量
    for feature in sorted(features):
        weight = features[feature]
        digest = hashlib.sha1(feature.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        bucket = value % dim
        sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
        vector[bucket] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:  # 理论上不会发生，兜底避免除零
        vector[0] = 1.0
        norm = 1.0
    return [value / norm for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本编码为向量列表。

    按配置选择 bge / openai / hash 后端；bge 加载失败或 openai 调用失败时
    自动降级为 hash 后端（记录日志，不向调用方抛异常）。
    空列表直接返回 ``[]``，单条文本返回 ``[vec]``。

    Args:
        texts: 文本列表（元素必须为字符串）。

    Returns:
        与 ``texts`` 顺序一致的向量列表，每个向量为 float 列表。

    Raises:
        ValueError: 入参不是字符串列表（属于调用方编程错误，需要显式暴露）。
    """
    items = _validate_texts(texts)
    if not items:
        return []
    _ensure_backend()
    with _STATE_LOCK:
        provider = str(_STATE.get("provider") or "hash")
    if provider == "bge":
        try:
            vectors = _encode_with_bge(items)
            logger.debug("bge 嵌入完成：%d 条文本", len(vectors))
            return vectors
        except Exception as exc:
            logger.warning("BGE 编码失败，自动降级 hash 后端：%s", exc)
            with _STATE_LOCK:
                _activate_hash(f"bge 编码失败，已自动降级 hash：{exc}")
    elif provider == "openai":
        try:
            vectors = _encode_with_openai(items)
            logger.debug("openai 嵌入完成：%d 条文本", len(vectors))
            return vectors
        except Exception as exc:
            logger.warning("OpenAI 兼容 embedding 调用失败，自动降级 hash 后端：%s", exc)
            with _STATE_LOCK:
                _activate_hash(f"openai 调用失败，已自动降级 hash：{exc}")
    with _STATE_LOCK:
        dim = int(_STATE.get("dim") or _configured_dim())
    return [_hash_vector(text, dim) for text in items]


def embed_query(text: str) -> list[float]:
    """把单条查询文本编码为向量。

    Args:
        text: 查询文本（字符串）。

    Returns:
        文本对应的向量（float 列表）。

    Raises:
        ValueError: text 不是字符串时抛出（空字符串仅告警并返回占位向量）。
    """
    if not isinstance(text, str):
        raise ValueError(f"embed_query 的 text 必须是字符串（实际类型：{type(text).__name__}）")
    if not text.strip():
        logger.warning("embed_query 收到空字符串，将返回占位向量，检索结果可能不理想")
    return embed_texts([text])[0]


def get_embedding_dim() -> int:
    """获取当前嵌入后端使用的向量维度。

    Returns:
        正整数维度；后端尚未解析时返回配置维度（默认 512），不触发模型加载。
    """
    state = _peek_state()
    dim = int(state.get("dim") or 0)
    return dim if dim > 0 else _configured_dim()


def get_embedding_info() -> dict:
    """获取嵌入后端信息（用于 /health 与启动日志，不会触发模型下载/网络调用）。

    Returns:
        字典，包含以下字段：

        - configured_provider: 配置声明的后端；
        - provider: 实际生效的后端（bge/hash/openai）；
        - model: 模型名称或 hash 描述；
        - dim: 向量维度；
        - backend: 具体实现（sentence-transformers/hash/openai）；
        - ready: 后端是否已就绪（bge 懒加载未触发时为 False）；
        - detail: 人类可读的说明（含降级原因）。
    """
    state = _peek_state()
    return {
        "configured_provider": str(state.get("configured_provider") or _configured_provider()),
        "provider": str(state.get("provider") or "hash"),
        "model": str(state.get("model") or ""),
        "dim": int(state.get("dim") or _configured_dim()),
        "backend": str(state.get("backend") or "hash"),
        "ready": bool(state.get("ready")),
        "detail": str(state.get("detail") or ""),
    }


def reset_embedding_backend() -> None:
    """重置嵌入后端缓存（主要供测试与运行期切换配置使用）。

    清空后端状态并释放懒加载的模型/客户端引用；不会删除任何文件或数据。

    Returns:
        None。
    """
    global _BGE_MODEL
    global _OPENAI_CLIENT
    with _STATE_LOCK:
        _STATE.clear()
        _BGE_MODEL = None
        _OPENAI_CLIENT = None
    logger.info("嵌入后端缓存已重置，下次调用将按当前配置重新选择后端")
