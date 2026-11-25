"""ChromaDB 持久化向量库封装（集合管理 + 文档增删改查 + 语义检索）。

本模块是向量层对外的唯一存储入口，职责包括：

- 使用 ``chromadb.PersistentClient`` 把集合真正持久化到 ``config.CHROMA_DIR``；
- 写入 / 更新 / 检索时显式传入 embeddings，绝不使用 ChromaDB 默认的 ONNX
  embedding function（默认函数会触发联网下载模型，离线环境会卡死）；
- 集合名严格限制为 ``config.COLLECTIONS``（market / course），非法集合抛 ``ValueError``；
- 首次初始化时给空集合灌入与 SQLite 课程/职业数据对齐的中文示例文档；
- 所有后端异常都记录日志；检索类接口尽量降级返回空结果，写操作抛
  ``VectorStoreError`` 由上层转成 500，参数错误则抛 ``ValueError`` 由上层转成 400。
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

from config import config, get_logger
from vectorstore.embeddings import embed_query, embed_texts

logger: logging.Logger = get_logger("crm.vectorstore")

# 集合 metadata 的 key 与距离度量；cosine distance = 1 - 余弦相似度，越小越相似
_DISTANCE_SPACE_KEY = "hnsw:space"
_COSINE_SPACE = "cosine"

# 保护 chroma 客户端与集合缓存；嵌入计算不在此锁内进行，避免阻塞其它请求
_LOCK = threading.RLock()
# 缓存的 PersistentClient 实例（reset_vectorstore 会置空以便测试隔离）
_CLIENT: Any | None = None
# 缓存的 Collection 对象：{集合名: Collection}
_COLLECTIONS: dict[str, Any] = {}


class VectorStoreError(Exception):
    """向量库后端异常（初始化、写入、更新等失败时抛出）。

    上层（app.py）会把它和其它非 ValueError 异常一起转成 HTTP 500；
    而参数错误统一抛 ValueError，转成 HTTP 400。
    """


# 预置示例文档：内容与 database/schema.sql 中的课程、职业示例数据保持一致，
# 便于「RAG 检索课程/岗位文档」与「SQL 查询课程/职业表」得到互相印证的结果。
_SAMPLE_DOCUMENTS: dict[str, list[dict[str, Any]]] = {
    config.COLLECTION_MARKET: [
        {
            "id": "market-001",
            "document": (
                "AI 应用开发工程师岗位属于互联网·人工智能行业，月薪 15000 至 28000 元。"
                "主要职责是基于大模型搭建 RAG 知识库应用、设计 Agent 与工具调用流程、"
                "完成大模型 API 接入与效果评测，并配合产品经理把 AI 能力落地到业务场景。"
                "技能要求：Python、LangChain/LlamaIndex、向量数据库、Prompt 工程、基础后端开发。"
                "行业趋势：企业级 RAG 与 Agent 应用落地加速，既懂大模型 API 又懂工程实现的复合型人才缺口持续扩大。"
            ),
            "metadata": {
                "career_id": 1,
                "title": "AI 应用开发工程师",
                "industry": "互联网·人工智能",
                "salary_min": 15000,
                "salary_max": 28000,
                "skills": "Python、LangChain/LlamaIndex、向量数据库、Prompt 工程、后端开发",
                "trend": "企业级 RAG 与 Agent 应用落地加速，复合型人才缺口持续扩大",
            },
        },
        {
            "id": "market-002",
            "document": (
                "大模型算法工程师岗位属于人工智能行业，月薪 25000 至 50000 元。"
                "主要职责是参与大模型预训练、指令微调与对齐（SFT/RLHF）、推理性能优化和模型评测，"
                "跟踪前沿论文并复现关键算法。"
                "技能要求：PyTorch、分布式训练、Transformer 原理、LoRA/QLoRA 微调、CUDA 与推理优化。"
                "行业趋势：大模型从通用能力竞争转向行业垂直模型与推理成本优化，高端算法人才薪资持续走高。"
            ),
            "metadata": {
                "career_id": 2,
                "title": "大模型算法工程师",
                "industry": "人工智能",
                "salary_min": 25000,
                "salary_max": 50000,
                "skills": "PyTorch、分布式训练、Transformer、LoRA 微调、推理优化",
                "trend": "垂直行业大模型与推理成本优化成为主要方向，高端算法人才稀缺",
            },
        },
        {
            "id": "market-003",
            "document": (
                "数据分析师岗位属于互联网·金融行业，月薪 12000 至 25000 元。"
                "主要职责是搭建业务指标体系、编写 SQL 取数与报表、使用 Python 做数据清洗与可视化，"
                "通过数据分析为运营和产品决策提供支持。"
                "技能要求：SQL、Python（pandas/matplotlib）、Excel、BI 工具、业务理解与数据表达能力。"
                "行业趋势：数据驱动决策成为企业标配，懂业务、会写 SQL 且能做可视化的复合型分析师需求稳定。"
            ),
            "metadata": {
                "career_id": 3,
                "title": "数据分析师",
                "industry": "互联网·金融",
                "salary_min": 12000,
                "salary_max": 25000,
                "skills": "SQL、Python、pandas、可视化、BI 工具、业务分析",
                "trend": "数据驱动决策普及，业务与数据能力兼备的分析师需求稳定",
            },
        },
        {
            "id": "market-004",
            "document": (
                "Python 后端开发工程师岗位属于互联网行业，月薪 13000 至 26000 元。"
                "主要职责是使用 Python 框架开发与维护后端服务、设计数据库表结构与接口、"
                "保障服务性能与稳定性，并参与需求评审和代码评审。"
                "技能要求：Python、FastAPI/Django/Flask、MySQL、Redis、Linux、Git、基础微服务与容器化知识。"
                "行业趋势：AI 应用与业务系统都需要后端支撑，掌握 Python Web 框架和数据库的工程师长期有稳定需求。"
            ),
            "metadata": {
                "career_id": 4,
                "title": "Python 后端开发工程师",
                "industry": "互联网",
                "salary_min": 13000,
                "salary_max": 26000,
                "skills": "Python、FastAPI/Django、MySQL、Redis、Linux、Git",
                "trend": "AI 应用与业务系统持续扩产后端岗位，Web 框架与数据库能力是刚需",
            },
        },
        {
            "id": "market-005",
            "document": (
                "算法工程师岗位属于人工智能·互联网行业，月薪 20000 至 40000 元。"
                "主要职责是面向推荐、搜索或自然语言处理场景设计机器学习方案，"
                "完成特征工程、模型训练调优与线上部署，并持续跟踪业务指标。"
                "技能要求：机器学习与深度学习基础、Python、PyTorch/TensorFlow、特征工程、A/B 实验与模型评估。"
                "行业趋势：推荐与搜索场景仍在精细化迭代，具备完整建模与落地经验的算法工程师议价能力较强。"
            ),
            "metadata": {
                "career_id": 5,
                "title": "算法工程师",
                "industry": "人工智能·互联网",
                "salary_min": 20000,
                "salary_max": 40000,
                "skills": "机器学习、深度学习、Python、PyTorch、特征工程、模型评估",
                "trend": "推荐搜索场景精细化迭代，完整建模与落地经验价值高",
            },
        },
        {
            "id": "market-006",
            "document": (
                "AI 产品经理岗位属于人工智能·互联网行业，月薪 18000 至 35000 元。"
                "主要职责是调研用户需求、理解大模型能力边界、撰写 PRD 并推动 AI 功能落地，"
                "通过数据分析持续评估产品效果并迭代方案。"
                "技能要求：需求分析与产品设计、大模型与 RAG 基础认知、数据分析、跨团队沟通与项目管理。"
                "行业趋势：AI 产品从概念验证走向规模化落地，既懂业务又理解大模型边界的 AI 产品经理供不应求。"
            ),
            "metadata": {
                "career_id": 6,
                "title": "AI 产品经理",
                "industry": "人工智能·互联网",
                "salary_min": 18000,
                "salary_max": 35000,
                "skills": "需求分析、产品设计、大模型与 RAG 认知、数据分析、项目管理",
                "trend": "AI 产品规模化落地，理解大模型边界的复合型产品经理供不应求",
            },
        },
        {
            "id": "market-007",
            "document": (
                "Python 初级开发工程师岗位属于互联网·软件服务行业，月薪 8000 至 15000 元。"
                "主要职责是在指导下完成 Python 业务功能开发、编写单元测试、修复缺陷并维护项目文档。"
                "技能要求：Python 语法基础、至少一种 Web 框架、SQL、Linux 常用命令与 Git 协作。"
                "行业趋势：大量企业级项目使用 Python 开发，初级岗位是转行进入开发行业的主要入口之一。"
            ),
            "metadata": {
                "career_id": 7,
                "title": "Python 初级开发工程师",
                "industry": "互联网·软件服务",
                "salary_min": 8000,
                "salary_max": 15000,
                "skills": "Python 基础、Web 框架、SQL、Linux、Git",
                "trend": "Python 项目需求广泛，初级岗位是转行开发的主要入口",
            },
        },
        {
            "id": "market-008",
            "document": (
                "数据运营专员岗位属于互联网·电商行业，月薪 9000 至 16000 元。"
                "主要职责是维护日常数据看板、统计活动效果并输出复盘报告，结合用户分层与增长目标提出运营建议。"
                "技能要求：Excel、SQL 取数、数据看板搭建、活动复盘、用户增长与沟通协调能力。"
                "行业趋势：电商与内容平台重视精细化运营，懂数据、能落地的数据运营岗位需求稳定。"
            ),
            "metadata": {
                "career_id": 8,
                "title": "数据运营专员",
                "industry": "互联网·电商",
                "salary_min": 9000,
                "salary_max": 16000,
                "skills": "Excel、SQL、数据看板、活动复盘、用户增长",
                "trend": "精细化运营依赖数据支撑，数据运营岗位需求稳定",
            },
        },
    ],
    config.COLLECTION_COURSE: [
        {
            "id": "course-001",
            "document": (
                "《Python 全栈开发实战》是一门面向零基础的编程开发课程，学费 12800 元，学习周期 24 周。"
                "课程从 Python 语法基础、数据结构与算法讲起，逐步覆盖 Web 后端开发、数据库设计与前后端联调，"
                "并通过企业级项目实战巩固所学。适合零基础转行、在校学生系统学习，学完可胜任 Python 后端开发相关岗位。"
            ),
            "metadata": {
                "course_id": 1,
                "name": "Python 全栈开发实战",
                "category": "编程开发",
                "price": 12800,
                "duration": "24 周",
                "level": "零基础",
                "suitable_for": "零基础转行、在校学生",
            },
        },
        {
            "id": "course-002",
            "document": (
                "《Python 数据分析与可视化》是一门面向有 Python 基础学员的数据分析课程，学费 9800 元，学习周期 16 周。"
                "课程覆盖 NumPy、pandas 数据处理、数据清洗、指标体系与 matplotlib/BI 可视化，帮助学员独立完成业务分析报告。"
                "适合想转行数据分析、业务运营的学员，学完可胜任数据分析师相关岗位。"
            ),
            "metadata": {
                "course_id": 2,
                "name": "Python 数据分析与可视化",
                "category": "数据分析",
                "price": 9800,
                "duration": "16 周",
                "level": "有 Python 基础",
                "suitable_for": "转行数据分析、业务运营",
            },
        },
        {
            "id": "course-003",
            "document": (
                "《AI 应用开发工程师（大模型 RAG）》是一门人工智能方向的进阶课程，学费 15800 元，学习周期 20 周。"
                "课程讲解大模型 API 调用、Prompt 工程、向量数据库、RAG 知识库与 Agent 工具编排，"
                "并通过企业级问答应用项目实战打通工程链路。适合会 Python 基础、想转型 AI 应用开发的后端开发人员，"
                "学完可胜任 AI 应用开发工程师岗位。"
            ),
            "metadata": {
                "course_id": 3,
                "name": "AI 应用开发工程师（大模型 RAG）",
                "category": "人工智能",
                "price": 15800,
                "duration": "20 周",
                "level": "会 Python 基础",
                "suitable_for": "想转型 AI 应用开发的后端开发人员",
            },
        },
        {
            "id": "course-004",
            "document": (
                "《大模型工程与微调实战》是一门人工智能进阶课程，学费 19800 元，学习周期 24 周。"
                "课程覆盖 Transformer 原理、LoRA/QLoRA 微调、训练数据构造、模型评测与推理部署，"
                "帮助学员掌握从微调到上线的完整大模型工程能力。适合有后端或算法基础的算法工程师、AI 方向进阶者，"
                "学完可胜任大模型算法工程师等高薪岗位。"
            ),
            "metadata": {
                "course_id": 4,
                "name": "大模型工程与微调实战",
                "category": "人工智能",
                "price": 19800,
                "duration": "24 周",
                "level": "有后端或算法基础",
                "suitable_for": "算法工程师、AI 方向进阶者",
            },
        },
        {
            "id": "course-005",
            "document": (
                "《前端开发实战（Vue3 + TypeScript）》是一门前端开发课程，学费 8800 元，学习周期 16 周。"
                "课程从 HTML/CSS/JavaScript 基础讲起，深入 Vue3 组件化开发、TypeScript 类型系统、"
                "前后端联调与项目部署，适合零基础想快速就业的学员，学完可胜任前端开发相关岗位。"
            ),
            "metadata": {
                "course_id": 5,
                "name": "前端开发实战（Vue3 + TypeScript）",
                "category": "前端开发",
                "price": 8800,
                "duration": "16 周",
                "level": "零基础",
                "suitable_for": "零基础想快速就业的学员",
            },
        },
        {
            "id": "course-006",
            "document": (
                "《算法工程师进阶（机器学习方向）》是一门算法方向课程，学费 16800 元，学习周期 20 周。"
                "课程系统讲解机器学习与深度学习原理、特征工程、模型调优、A/B 实验与模型评估，"
                "并结合推荐/搜索场景项目实战。适合有数学与 Python 基础的算法岗求职者、计算机专业学生，"
                "学完可胜任算法工程师岗位。"
            ),
            "metadata": {
                "course_id": 6,
                "name": "算法工程师进阶（机器学习方向）",
                "category": "算法",
                "price": 16800,
                "duration": "20 周",
                "level": "有数学与 Python 基础",
                "suitable_for": "算法岗求职者、计算机专业学生",
            },
        },
        {
            "id": "course-007",
            "document": (
                "《Python 自动化办公训练营》是一门办公自动化课程，学费 3980 元，学习周期 8 周。"
                "课程讲解 Python 基础语法、Excel 批量处理、文件与邮件自动化、报表自动生成等实用技能，"
                "适合零基础的职场白领、行政财务人员，学完可用 Python 替代大量重复性办公操作。"
            ),
            "metadata": {
                "course_id": 7,
                "name": "Python 自动化办公训练营",
                "category": "办公自动化",
                "price": 3980,
                "duration": "8 周",
                "level": "零基础",
                "suitable_for": "职场白领、行政财务",
            },
        },
    ],
}


def _normalize_collection_name(collection_name: str) -> str:
    """校验并归一化集合名。

    Args:
        collection_name: 待校验的集合名。

    Returns:
        去掉首尾空白后的合法集合名。

    Raises:
        ValueError: 集合名不是字符串、为空或不在 ``config.COLLECTIONS`` 白名单内时抛出。
    """
    if not isinstance(collection_name, str):
        raise ValueError(f"collection 必须是字符串（实际类型：{type(collection_name).__name__}）")
    name = collection_name.strip()
    if not name:
        raise ValueError("collection 不能为空")
    allowed = tuple(str(item) for item in getattr(config, "COLLECTIONS", ()))
    if name not in allowed:
        raise ValueError(f"不支持的集合名：{collection_name!r}，可选值：{', '.join(allowed)}")
    return name


def _ensure_client() -> Any:
    """获取（必要时创建）ChromaDB 持久化客户端。

    Returns:
        ``chromadb.PersistentClient`` 实例（模块级缓存）。

    Raises:
        VectorStoreError: 数据目录创建或客户端初始化失败时抛出。
    """
    global _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            return _CLIENT
        try:
            # Chroma 需要数据目录存在；目录由 config 统一管理，这里不硬编码路径
            config.ensure_dirs()
            client = chromadb.PersistentClient(
                path=str(config.CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception as exc:
            logger.exception("初始化 ChromaDB 持久化客户端失败（path=%s）：%s", getattr(config, "CHROMA_DIR", ""), exc)
            raise VectorStoreError(f"初始化向量库失败：{exc}") from exc
        _CLIENT = client
        logger.info("ChromaDB 持久化客户端已就绪：path=%s", config.CHROMA_DIR)
        return client


def _get_collection(collection_name: str) -> Any:
    """获取（必要时创建）集合对象并缓存。

    创建时显式传 ``embedding_function=None``，避免 ChromaDB 使用默认 ONNX
    向量函数（该函数会联网下载模型，离线环境可能长时间卡住）。

    Args:
        collection_name: 集合名（已通过白名单校验）。

    Returns:
        ChromaDB Collection 对象。

    Raises:
        ValueError: 集合名非法时抛出。
        VectorStoreError: 客户端或集合创建失败时抛出。
    """
    name = _normalize_collection_name(collection_name)
    with _LOCK:
        cached = _COLLECTIONS.get(name)
        if cached is not None:
            return cached
        client = _ensure_client()
        try:
            collection = client.get_or_create_collection(
                name=name,
                metadata={_DISTANCE_SPACE_KEY: _COSINE_SPACE},
                embedding_function=None,
            )
        except Exception as exc:
            logger.exception("获取向量集合 %s 失败：%s", name, exc)
            raise VectorStoreError(f"获取向量集合 {name} 失败：{exc}") from exc
        _COLLECTIONS[name] = collection
        return collection


def _clean_metadata(metadata: dict) -> dict[str, Any]:
    """清洗单条 metadata，使其满足 ChromaDB 只接受标量值的要求。

    处理规则：过滤 ``None`` 值（Chroma 会抛 TypeError）、丢弃 NaN/Inf 浮点数、
    保留 bool/int/str、float、list/tuple/set 转成「、」拼接的字符串、其它类型转 str。

    Args:
        metadata: 原始 metadata 字典。

    Returns:
        只包含标量值的 metadata 字典。
    """
    cleaned: dict[str, Any] = {}
    for raw_key, value in metadata.items():
        if raw_key is None:
            logger.debug("忽略 metadata 中的 None 键")
            continue
        key = str(raw_key)
        if value is None:
            # Chroma 只接受标量，None 会直接抛 TypeError，这里静默过滤
            continue
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, int):
            cleaned[key] = int(value)
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                logger.warning("忽略 metadata 字段 %s 的非法浮点数：%r", key, value)
                continue
            cleaned[key] = float(value)
        elif isinstance(value, str):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            parts = [str(item) for item in value if item is not None]
            if parts:
                cleaned[key] = "、".join(parts)
        elif isinstance(value, (set, frozenset)):
            # set 无序，排序后再拼接保证同输入得到同 metadata
            parts = sorted(str(item) for item in value if item is not None)
            if parts:
                cleaned[key] = "、".join(parts)
        else:
            cleaned[key] = str(value)
    return cleaned


def _validate_documents(documents: list[str]) -> list[str]:
    """校验文档列表，返回元素列表。

    Args:
        documents: 待写入的文档文本列表。

    Returns:
        校验通过的文档列表（保持原始顺序，不做文本改写）。

    Raises:
        ValueError: 不是列表 / 为空 / 元素不是非空字符串时抛出。
    """
    if documents is None:
        raise ValueError("documents 不能为 None")
    if isinstance(documents, str) or not isinstance(documents, (list, tuple)):
        raise ValueError("documents 必须是字符串列表")
    items = list(documents)
    if not items:
        raise ValueError("documents 不能为空，至少需要一条文档")
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ValueError(f"documents 第 {index} 条必须是字符串（实际类型：{type(item).__name__}）")
        if not item.strip():
            raise ValueError(f"documents 第 {index} 条不能为空字符串")
    return items


def _prepare_ids(ids: list[str] | None, collection_name: str, expected: int) -> list[str]:
    """校验或自动生成文档 id。

    Args:
        ids: 调用方传入的 id 列表；None 表示自动生成。
        collection_name: 集合名，用于生成 ``{collection_name}-{uuid4().hex}`` 形式的 id。
        expected: 期望的 id 数量（等于文档数量）。

    Returns:
        与文档一一对应的 id 列表。

    Raises:
        ValueError: ids 不是列表、数量不匹配、元素非法或存在重复时抛出。
    """
    if ids is None:
        return [f"{collection_name}-{uuid.uuid4().hex}" for _ in range(expected)]
    if isinstance(ids, str) or not isinstance(ids, (list, tuple)):
        raise ValueError("ids 必须是字符串列表")
    items = list(ids)
    if len(items) != expected:
        raise ValueError(f"ids 数量（{len(items)}）必须与 documents 数量（{expected}）一致")
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ValueError(f"ids 第 {index} 个必须是字符串（实际类型：{type(item).__name__}）")
        if not item.strip():
            raise ValueError(f"ids 第 {index} 个不能为空字符串")
    if len(set(items)) != len(items):
        raise ValueError("ids 存在重复值，无法作为文档唯一主键")
    return items


def _validate_ids(ids: list[str]) -> list[str]:
    """校验更新 / 删除操作传入的 id 列表。

    Args:
        ids: 待校验的 id 列表。

    Returns:
        校验通过的 id 列表。

    Raises:
        ValueError: 不是列表、为空、元素非法或存在重复时抛出。
    """
    if ids is None:
        raise ValueError("ids 不能为 None")
    if isinstance(ids, str) or not isinstance(ids, (list, tuple)):
        raise ValueError("ids 必须是字符串列表")
    items = list(ids)
    if not items:
        raise ValueError("ids 不能为空，至少需要一个文档 id")
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ValueError(f"ids 第 {index} 个必须是字符串（实际类型：{type(item).__name__}）")
        if not item.strip():
            raise ValueError(f"ids 第 {index} 个不能为空字符串")
    if len(set(items)) != len(items):
        raise ValueError("ids 存在重复值，同一文档不能被重复更新或删除")
    return items


def _prepare_metadatas(metadatas: list[dict] | None, expected: int) -> list[dict[str, Any]] | None:
    """校验并清洗 metadata 列表。

    Args:
        metadatas: 调用方传入的 metadata 列表；None 表示不设置 metadata。
        expected: 期望的数量（等于文档数量）。

    Returns:
        清洗后的 metadata 列表；入参为 None 原样返回 None；
        所有条目的 metadata 在清洗后都为空时也返回 None（Chroma 不接受空 metadata 字典）。

    Raises:
        ValueError: 不是字典列表、数量不匹配、元素不是字典，
            或部分条目的 metadata 清洗后为空（无法与 Chroma 的批量写入语义对齐）时抛出。
    """
    if metadatas is None:
        return None
    if isinstance(metadatas, dict) or not isinstance(metadatas, (list, tuple)):
        raise ValueError("metadatas 必须是字典列表，每个文档对应一个字典")
    items = list(metadatas)
    if len(items) != expected:
        raise ValueError(f"metadatas 数量（{len(items)}）必须与文档数量（{expected}）一致")
    cleaned: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if item is None:
            cleaned.append({})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"metadatas 第 {index} 个必须是字典（实际类型：{type(item).__name__}）")
        cleaned.append(_clean_metadata(item))
    # ChromaDB 1.5.x 拒绝空 metadata 字典：全部为空等价于「不设置 metadata」，整体不传即可
    if not any(cleaned):
        logger.debug("metadata 清洗后全部为空，本次写入不携带 metadatas")
        return None
    empty_indexes = [index for index, metadata in enumerate(cleaned) if not metadata]
    if empty_indexes:
        # 批量 metadatas 中混有空字典时 Chroma 无法逐条省略，只能拒绝并为调用方指出位置
        raise ValueError(
            "metadatas 中部分条目过滤 None 值后为空，Chroma 无法只省略个别条目的 metadata；"
            f"请为这些文档提供至少一个有效字段或整体不传 metadatas（空条目下标：{empty_indexes[:10]}）"
        )
    return cleaned


def _extract_first_batch(payload: dict[str, Any], key: str) -> list[Any]:
    """从 Chroma query 返回的「列表的列表」中取出第一个查询批次的列表。

    Args:
        payload: ``collection.query`` 的返回字典。
        key: 字段名（ids / documents / metadatas / distances）。

    Returns:
        第一个查询批次的扁平列表；字段缺失或为空时返回空列表。
    """
    value = payload.get(key)
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        first = value[0]
        if isinstance(first, (list, tuple)):
            return list(first)
        return list(value)
    return []


def _result_metadata(raw_metadata: Any) -> dict[str, Any]:
    """把 Chroma 返回的 metadata 转成干净的结果字典（过滤 None 值）。

    Args:
        raw_metadata: Chroma 返回的单条 metadata（可能是 None 或字典）。

    Returns:
        不含 None 值的 metadata 字典；非法输入返回空字典。
    """
    if not isinstance(raw_metadata, dict):
        return {}
    return {str(key): value for key, value in raw_metadata.items() if value is not None}


def add_documents(
    collection_name: str,
    documents: list[str],
    metadatas: list[dict] | None = None,
    ids: list[str] | None = None,
) -> list[str]:
    """向指定集合新增文档（自动生成向量）。

    Args:
        collection_name: 集合名，必须是 ``config.COLLECTIONS`` 中的值。
        documents: 文档文本列表（非空）。
        metadatas: 与文档等长的 metadata 字典列表；None 表示不带 metadata。
        ids: 与文档等长的 id 列表；None 时自动生成 ``{集合名}-{uuid4().hex}``。

    Returns:
        实际写入的文档 id 列表（与 documents 顺序一致）。

    Raises:
        ValueError: 集合名或参数非法时抛出。
        VectorStoreError: 向量生成或写入 ChromaDB 失败时抛出。
    """
    name = _normalize_collection_name(collection_name)
    docs = _validate_documents(documents)
    doc_count = len(docs)
    id_list = _prepare_ids(ids, name, doc_count)
    metadata_list = _prepare_metadatas(metadatas, doc_count)

    try:
        vectors = embed_texts(docs)
    except Exception as exc:
        logger.exception("生成文档向量失败：collection=%s，文档数=%d", name, doc_count)
        raise VectorStoreError(f"生成文档向量失败：{exc}") from exc
    if len(vectors) != doc_count:
        raise VectorStoreError(f"向量数量（{len(vectors)}）与文档数量（{doc_count}）不一致")

    collection = _get_collection(name)
    add_kwargs: dict[str, Any] = {"ids": id_list, "embeddings": vectors, "documents": docs}
    if metadata_list is not None:
        add_kwargs["metadatas"] = metadata_list
    try:
        collection.add(**add_kwargs)
    except Exception as exc:
        logger.exception("写入向量集合失败：collection=%s，ids=%s", name, id_list[:3])
        raise VectorStoreError(f"写入向量集合 {name} 失败：{exc}") from exc

    logger.info("向量集合 %s 新增 %d 条文档", name, doc_count)
    return id_list


def update_documents(
    collection_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    """更新指定文档的文本（并重新生成向量），可选更新 metadata。

    Args:
        collection_name: 集合名，必须是 ``config.COLLECTIONS`` 中的值。
        ids: 与 documents 等长的文档 id 列表。
        documents: 新的文档文本列表。
        metadatas: 与 ids 等长的 metadata 列表；None 表示保留原有 metadata。

    Returns:
        None。

    Raises:
        ValueError: 集合名或参数非法时抛出。
        VectorStoreError: 向量生成或更新 ChromaDB 失败时抛出。
    """
    name = _normalize_collection_name(collection_name)
    id_list = _validate_ids(ids)
    docs = _validate_documents(documents)
    if len(id_list) != len(docs):
        raise ValueError(f"ids 数量（{len(id_list)}）必须与 documents 数量（{len(docs)}）一致")
    metadata_list = _prepare_metadatas(metadatas, len(id_list))

    collection = _get_collection(name)
    # 先确认哪些 id 真实存在：Chroma 对不存在的 id 是静默 no-op，不提示调用方
    try:
        existing_payload = collection.get(ids=id_list)
        existing_ids = set(existing_payload.get("ids") or [])
    except Exception as exc:
        logger.exception("更新前查询集合 %s 已有文档失败：%s", name, exc)
        raise VectorStoreError(f"更新前查询向量集合 {name} 失败：{exc}") from exc

    missing_ids = [doc_id for doc_id in id_list if doc_id not in existing_ids]
    if missing_ids:
        logger.warning("更新向量集合 %s 时忽略 %d 个不存在的文档 id：%s", name, len(missing_ids), missing_ids[:5])
        keep_indexes = [index for index, doc_id in enumerate(id_list) if doc_id in existing_ids]
        id_list = [id_list[index] for index in keep_indexes]
        docs = [docs[index] for index in keep_indexes]
        if metadata_list is not None:
            metadata_list = [metadata_list[index] for index in keep_indexes]
    if not id_list:
        logger.info("向量集合 %s 没有可更新的文档，更新操作已跳过", name)
        return

    try:
        vectors = embed_texts(docs)
    except Exception as exc:
        logger.exception("生成更新文档向量失败：collection=%s，文档数=%d", name, len(docs))
        raise VectorStoreError(f"生成更新文档向量失败：{exc}") from exc
    if len(vectors) != len(docs):
        raise VectorStoreError(f"向量数量（{len(vectors)}）与文档数量（{len(docs)}）不一致")

    update_kwargs: dict[str, Any] = {"ids": id_list, "embeddings": vectors, "documents": docs}
    if metadata_list is not None:
        update_kwargs["metadatas"] = metadata_list
    try:
        collection.update(**update_kwargs)
    except Exception as exc:
        logger.exception("更新向量集合失败：collection=%s，ids=%s", name, id_list[:3])
        raise VectorStoreError(f"更新向量集合 {name} 失败：{exc}") from exc

    logger.info("向量集合 %s 更新 %d 条文档", name, len(id_list))


def delete_documents(collection_name: str, ids: list[str]) -> None:
    """从指定集合删除文档（不存在的 id 会被 ChromaDB 忽略）。

    Args:
        collection_name: 集合名，必须是 ``config.COLLECTIONS`` 中的值。
        ids: 待删除的文档 id 列表。

    Returns:
        None。

    Raises:
        ValueError: 集合名或参数非法时抛出。
        VectorStoreError: 删除操作失败时抛出。
    """
    name = _normalize_collection_name(collection_name)
    id_list = _validate_ids(ids)
    collection = _get_collection(name)
    try:
        collection.delete(ids=id_list)
    except Exception as exc:
        logger.exception("删除向量集合文档失败：collection=%s，ids=%s", name, id_list[:3])
        raise VectorStoreError(f"删除向量集合 {name} 文档失败：{exc}") from exc
    logger.info("向量集合 %s 删除 %d 条文档（不存在的 id 会被忽略）", name, len(id_list))


def search(collection_name: str, query: str, top_k: int = 3) -> list[dict]:
    """在指定集合中做语义检索，返回按距离升序排列的结果。

    Args:
        collection_name: 集合名，必须是 ``config.COLLECTIONS`` 中的值。
        query: 查询文本（非空字符串）。
        top_k: 返回条数上限（正整数，实际不超过集合总数）。

    Returns:
        ``[{"id", "document", "metadata", "distance"}]``，distance 为余弦距离
        （越小越相似），按升序排列；集合为空或后端异常时返回 ``[]``。

    Raises:
        ValueError: 集合名非法、query 非字符串或为空、top_k 非正整数时抛出。
    """
    name = _normalize_collection_name(collection_name)
    if not isinstance(query, str):
        raise ValueError(f"query 必须是字符串（实际类型：{type(query).__name__}）")
    query_text = query.strip()
    if not query_text:
        raise ValueError("query 不能为空")
    if isinstance(top_k, bool):
        raise ValueError("top_k 必须是正整数")
    try:
        top_k_value = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"top_k 必须是正整数，实际为 {top_k!r}") from exc
    if top_k_value <= 0:
        raise ValueError("top_k 必须是正整数")

    try:
        collection = _get_collection(name)
        total = int(collection.count())
    except Exception as exc:
        logger.exception("检索前获取向量集合失败：collection=%s：%s", name, exc)
        return []
    if total <= 0:
        logger.info("向量集合 %s 为空，检索返回空列表", name)
        return []

    try:
        query_vector = embed_query(query_text)
        response = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k_value, total),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # 检索是读路径，失败时降级为空结果，避免让对话整体报错
        logger.exception("向量检索失败：collection=%s，query=%r", name, query_text[:50])
        return []

    ids_batch = _extract_first_batch(response, "ids")
    documents_batch = _extract_first_batch(response, "documents")
    metadatas_batch = _extract_first_batch(response, "metadatas")
    distances_batch = _extract_first_batch(response, "distances")

    results: list[dict] = []
    for position, doc_id in enumerate(ids_batch):
        document = documents_batch[position] if position < len(documents_batch) else ""
        metadata = _result_metadata(metadatas_batch[position] if position < len(metadatas_batch) else None)
        try:
            distance = float(distances_batch[position]) if position < len(distances_batch) else 0.0
        except (TypeError, ValueError):
            distance = 0.0
        results.append(
            {
                "id": str(doc_id),
                "document": "" if document is None else str(document),
                "metadata": metadata,
                "distance": distance,
            }
        )
    # Chroma 已按距离升序返回，这里再排一次以显式满足契约（distance 升序）
    results.sort(key=lambda item: item["distance"])
    logger.info("向量检索：collection=%s，query=%r，命中 %d 条", name, query_text[:50], len(results))
    return results


def list_collections() -> list[dict]:
    """列出所有向量集合及其文档数。

    返回结果按集合名排序，便于前端 / 日志稳定展示。

    Returns:
        ``[{"name": str, "count": int}]``；后端异常时返回 ``[]``。
    """
    try:
        client = _ensure_client()
        raw_collections = client.list_collections()
    except Exception as exc:
        logger.exception("查询向量集合列表失败：%s", exc)
        return []
    items: list[dict] = []
    for collection in raw_collections:
        name = str(getattr(collection, "name", "") or "")
        try:
            count = int(collection.count())
        except Exception as exc:
            logger.warning("读取向量集合 %s 文档数失败：%s", name, exc)
            count = 0
        items.append({"name": name, "count": count})
    items.sort(key=lambda item: item["name"])
    return items


def get_collection_stats() -> list[dict]:
    """获取各集合的文档数统计（与 :func:`list_collections` 结构一致）。

    Returns:
        ``[{"name": str, "count": int}]``，按集合名排序。
    """
    return list_collections()


def count_documents(collection_name: str) -> int:
    """统计指定集合的文档数量。

    Args:
        collection_name: 集合名，必须是 ``config.COLLECTIONS`` 中的值。

    Returns:
        文档数量；后端异常时记录日志并返回 0。

    Raises:
        ValueError: 集合名非法时抛出。
    """
    name = _normalize_collection_name(collection_name)
    try:
        collection = _get_collection(name)
        return int(collection.count())
    except Exception as exc:
        logger.exception("统计向量集合 %s 文档数失败：%s", name, exc)
        return 0


def get_documents(collection_name: str, limit: int = 100) -> list[dict]:
    """读取集合中的文档（调试与测试用，不做向量检索）。

    Args:
        collection_name: 集合名，必须是 ``config.COLLECTIONS`` 中的值。
        limit: 最多读取的条数（正整数）。

    Returns:
        ``[{"id", "document", "metadata"}]``；后端异常时返回 ``[]``。

    Raises:
        ValueError: 集合名、limit 非法时抛出。
    """
    name = _normalize_collection_name(collection_name)
    if isinstance(limit, bool):
        raise ValueError("limit 必须是正整数")
    try:
        limit_value = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"limit 必须是正整数，实际为 {limit!r}") from exc
    if limit_value <= 0:
        raise ValueError("limit 必须是正整数")
    try:
        collection = _get_collection(name)
        payload = collection.get(limit=limit_value)
    except Exception as exc:
        logger.exception("读取向量集合 %s 文档失败：%s", name, exc)
        return []
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    items: list[dict] = []
    for position, doc_id in enumerate(ids):
        items.append(
            {
                "id": str(doc_id),
                "document": str(documents[position]) if position < len(documents) and documents[position] is not None else "",
                "metadata": _result_metadata(metadatas[position] if position < len(metadatas) else None),
            }
        )
    return items


def clear_collection(collection_name: str) -> None:
    """清空指定集合（测试用；删除集合后下次访问会重新创建）。

    Args:
        collection_name: 集合名，必须是 ``config.COLLECTIONS`` 中的值。

    Returns:
        None。

    Raises:
        ValueError: 集合名非法时抛出。
    """
    name = _normalize_collection_name(collection_name)
    with _LOCK:
        # 先清缓存，避免后续继续使用已删除集合的旧对象
        _COLLECTIONS.pop(name, None)
        try:
            client = _ensure_client()
            client.delete_collection(name)
            logger.info("已清空向量集合 %s", name)
        except Exception as exc:
            message = str(exc)
            if "does not exist" in message.lower():
                logger.info("向量集合 %s 原本不存在，无需清空", name)
            else:
                logger.warning("清空向量集合 %s 失败（数据可能仍存在）：%s", name, exc)


def reset_vectorstore() -> None:
    """重置缓存的 ChromaDB 客户端与集合对象（测试用）。

    只关闭客户端并清空缓存，不删除磁盘数据；下次调用会重新打开
    ``config.CHROMA_DIR``，因此可用于验证持久化。

    Returns:
        None。
    """
    global _CLIENT
    with _LOCK:
        client = _CLIENT
        _CLIENT = None
        _COLLECTIONS.clear()
    if client is not None:
        try:
            client.close()
        except Exception as exc:
            logger.warning("关闭 ChromaDB 客户端失败：%s", exc)
    logger.info("向量库客户端缓存已重置（数据仍持久化在磁盘上）")


def init_vectorstores() -> None:
    """初始化向量集合：确保 market/course 集合存在，并在为空时灌入示例文档。

    该函数幂等：集合已有文档时绝不重复写入；单个集合初始化失败只记录错误，
    不影响其它集合，也不阻塞应用启动。

    Returns:
        None。
    """
    logger.debug("开始初始化向量集合：%s", ", ".join(str(name) for name in config.COLLECTIONS))
    for collection_name in config.COLLECTIONS:
        name = str(collection_name)
        try:
            collection = _get_collection(name)
            count = int(collection.count())
            if count > 0:
                logger.debug("向量集合 %s 已有 %d 条文档，跳过示例数据初始化", name, count)
                continue
            samples = _SAMPLE_DOCUMENTS.get(name) or []
            if not samples:
                logger.warning("向量集合 %s 没有预置示例文档，保持为空", name)
                continue
            add_documents(
                name,
                [str(sample["document"]) for sample in samples],
                [dict(sample["metadata"]) for sample in samples],
                [str(sample["id"]) for sample in samples],
            )
            logger.info("向量集合 %s 首次初始化：已灌入 %d 条示例文档", name, len(samples))
        except Exception as exc:
            # 初始化阶段保持健壮：记录完整堆栈但继续处理下一个集合
            logger.error("初始化向量集合 %s 失败：%s", name, exc, exc_info=True)
    logger.info("向量集合初始化完成：%s", list_collections())
