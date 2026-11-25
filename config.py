"""全局配置模块。

职责：
1. 加载项目根目录的 ``.env``（python-dotenv），环境变量优先于 .env；
2. 集中管理 DeepSeek / 嵌入模型 / 路径 / 集合名 / 检索参数 / 日志参数；
3. 提供日志初始化、目录创建与配置自检（脱敏）能力。

设计原因：
- 所有可变配置集中在此，禁止在业务代码里硬编码模型名、路径、Key；
- DeepSeek 官方没有 embedding 接口，所以「对话模型」与「嵌入模型」必须分开配置；
- 支持 ``CRM_DATA_DIR`` 覆盖数据目录，方便测试与部署时隔离数据。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# 项目根目录：config.py 所在目录
BASE_DIR: Path = Path(__file__).resolve().parent
ENV_FILE: Path = BASE_DIR / ".env"

# override=False：已经存在的真实环境变量优先级更高（部署时常用）
load_dotenv(dotenv_path=ENV_FILE, override=False)


# ---------------------------------------------------------------------------
# 环境变量解析工具：统一处理类型转换与默认值，避免散落的 int()/bool() 调用
# ---------------------------------------------------------------------------
def _env_str(key: str, default: str = "") -> str:
    """读取字符串环境变量，自动去掉首尾空白。"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip()


def _env_int(key: str, default: int) -> int:
    """读取整数环境变量，非法值回退到默认值并记录警告。"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning("环境变量 %s=%r 不是合法整数，使用默认值 %s", key, raw, default)
        return default


def _env_float(key: str, default: float) -> float:
    """读取浮点数环境变量，非法值回退到默认值。"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning("环境变量 %s=%r 不是合法浮点数，使用默认值 %s", key, raw, default)
        return default


def _env_bool(key: str, default: bool) -> bool:
    """读取布尔环境变量，支持 true/false/1/0/yes/no/on/off。"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _env_path(key: str, default: Path) -> Path:
    """读取路径环境变量；相对路径按项目根目录解析。"""
    raw = _env_str(key)
    if not raw:
        return default
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


class Settings:
    """项目全局配置对象（单例 ``config``）。

    所有业务模块统一通过 ``from config import config`` 读取配置，
    测试可以通过修改 ``config`` 的属性来隔离数据目录。
    """

    def __init__(self) -> None:
        # ---------------- 基础路径 ----------------
        self.BASE_DIR: Path = BASE_DIR
        # 数据根目录：默认 ./data，可用 CRM_DATA_DIR 覆盖（测试时指向临时目录）
        self.DATA_DIR: Path = _env_path("CRM_DATA_DIR", BASE_DIR / "data")
        self.SQLITE_PATH: Path = self.DATA_DIR / "crm.db"
        self.CHROMA_DIR: Path = self.DATA_DIR / "chroma"
        self.LOG_DIR: Path = self.DATA_DIR / "logs"
        self.LOG_FILE: Path = self.LOG_DIR / "app.log"

        # ---------------- DeepSeek 对话模型 ----------------
        self.DEEPSEEK_API_KEY: str = _env_str("DEEPSEEK_API_KEY")
        self.DEEPSEEK_BASE_URL: str = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        # 模型名一律走配置，代码里不写死
        self.DEEPSEEK_MODEL: str = _env_str("DEEPSEEK_MODEL", "deepseek-flash")
        self.LLM_TEMPERATURE: float = _env_float("LLM_TEMPERATURE", 0.7)
        self.LLM_MAX_TOKENS: int = _env_int("LLM_MAX_TOKENS", 1500)
        self.LLM_TIMEOUT: float = _env_float("LLM_TIMEOUT", 60.0)
        self.LLM_MAX_TOOL_ROUNDS: int = _env_int("LLM_MAX_TOOL_ROUNDS", 6)
        # 无 Key 或调用失败时是否降级为规则引擎（保证系统可运行、可验收）
        self.LLM_OFFLINE_FALLBACK: bool = _env_bool("LLM_OFFLINE_FALLBACK", True)

        # ---------------- 嵌入模型 ----------------
        # bge = 本地 sentence-transformers；hash = 纯本地哈希向量（无模型依赖）；openai = 兼容接口
        self.EMBEDDING_PROVIDER: str = _env_str("EMBEDDING_PROVIDER", "bge").lower()
        self.EMBEDDING_MODEL: str = _env_str("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        self.EMBEDDING_DIM: int = _env_int("EMBEDDING_DIM", 512)
        self.EMBEDDING_DEVICE: str = _env_str("EMBEDDING_DEVICE", "cpu")
        self.EMBEDDING_BATCH_SIZE: int = _env_int("EMBEDDING_BATCH_SIZE", 16)
        self.HF_ENDPOINT: str = _env_str("HF_ENDPOINT", "https://hf-mirror.com")
        self.OPENAI_EMBEDDING_BASE_URL: str = _env_str("OPENAI_EMBEDDING_BASE_URL")
        self.OPENAI_EMBEDDING_API_KEY: str = _env_str("OPENAI_EMBEDDING_API_KEY")
        self.OPENAI_EMBEDDING_MODEL: str = _env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        # ---------------- 向量库 ----------------
        self.COLLECTION_MARKET: str = _env_str("VECTOR_COLLECTION_MARKET", "market")
        self.COLLECTION_COURSE: str = _env_str("VECTOR_COLLECTION_COURSE", "course")
        # 所有允许操作的集合名（接口参数校验白名单）
        self.COLLECTIONS: tuple[str, ...] = (self.COLLECTION_MARKET, self.COLLECTION_COURSE)
        self.RAG_TOP_K: int = _env_int("RAG_TOP_K", 3)

        # ---------------- 网络搜索插件 ----------------
        self.WEB_SEARCH_ENABLED: bool = _env_bool("WEB_SEARCH_ENABLED", True)
        # 搜索提供方：bing（cn.bing.com，国内网络通常可直连）/ duckduckgo / serpapi / none
        self.WEB_SEARCH_PROVIDER: str = _env_str("WEB_SEARCH_PROVIDER", "bing").lower()
        self.WEB_SEARCH_TOP_K: int = _env_int("WEB_SEARCH_TOP_K", 5)
        self.WEB_SEARCH_TIMEOUT: float = _env_float("WEB_SEARCH_TIMEOUT", 3.0)
        self.SERPAPI_API_KEY: str = _env_str("SERPAPI_API_KEY")

        # ---------------- SQL Agent 安全 ----------------
        self.SQL_READONLY: bool = _env_bool("SQL_READONLY", True)
        self.SQL_MAX_ROWS: int = _env_int("SQL_MAX_ROWS", 100)

        # ---------------- 聊天 ----------------
        self.CHAT_HISTORY_LIMIT: int = _env_int("CHAT_HISTORY_LIMIT", 20)
        self.MAX_QUERY_LENGTH: int = _env_int("MAX_QUERY_LENGTH", 2000)

        # ---------------- 日志 / Flask ----------------
        self.LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO").upper()
        self.HOST: str = _env_str("FLASK_HOST", "0.0.0.0")
        self.PORT: int = _env_int("FLASK_PORT", 5000)
        self.DEBUG: bool = _env_bool("FLASK_DEBUG", False)

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------
    @property
    def llm_configured(self) -> bool:
        """是否配置了 DeepSeek API Key（决定走真实大模型还是降级规则引擎）。"""
        return bool(self.DEEPSEEK_API_KEY)

    @property
    def is_production_llm(self) -> bool:
        """是否使用真实大模型（未配置 Key 时整个系统进入离线降级模式）。"""
        return self.llm_configured

    # ------------------------------------------------------------------
    # 生命周期辅助
    # ------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """创建运行所需的全部目录（幂等）。

        在 init_app() 启动阶段以及测试初始化时调用，
        保证 SQLite、ChromaDB、日志目录存在。
        """
        for directory in (self.DATA_DIR, self.CHROMA_DIR, self.LOG_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    def override_data_dir(self, data_dir: str | Path) -> None:
        """运行时切换数据目录（测试隔离用）。

        Args:
            data_dir: 新的数据根目录，SQLite / Chroma / 日志会随之切换。
        """
        new_dir = Path(data_dir).expanduser().resolve()
        self.DATA_DIR = new_dir
        self.SQLITE_PATH = new_dir / "crm.db"
        self.CHROMA_DIR = new_dir / "chroma"
        self.LOG_DIR = new_dir / "logs"
        self.LOG_FILE = self.LOG_DIR / "app.log"
        self.ensure_dirs()

    def masked(self) -> dict[str, Any]:
        """返回脱敏后的配置快照，供 /health、/stats 与启动日志使用。"""
        key = self.DEEPSEEK_API_KEY
        masked_key = f"{key[:5]}***{key[-4:]}" if len(key) > 12 else ("已配置" if key else "未配置")
        return {
            "model": self.DEEPSEEK_MODEL,
            "base_url": self.DEEPSEEK_BASE_URL,
            "api_key": masked_key,
            "llm_ready": self.llm_configured,
            "embedding_provider": self.EMBEDDING_PROVIDER,
            "embedding_model": self.EMBEDDING_MODEL,
            "sqlite_path": str(self.SQLITE_PATH),
            "chroma_dir": str(self.CHROMA_DIR),
            "collections": list(self.COLLECTIONS),
            "web_search": {"enabled": self.WEB_SEARCH_ENABLED, "provider": self.WEB_SEARCH_PROVIDER},
            "sql_readonly": self.SQL_READONLY,
        }

    def validate(self) -> list[str]:
        """配置自检，返回问题列表（空列表表示没有发现问题）。

        注意：未配置 API Key 不是错误（系统会降级为规则引擎），
        只有明显不合理的取值才提示。
        """
        problems: list[str] = []
        if self.EMBEDDING_PROVIDER not in {"bge", "hash", "openai"}:
            problems.append(f"EMBEDDING_PROVIDER={self.EMBEDDING_PROVIDER} 不受支持，可选 bge/hash/openai")
        if self.EMBEDDING_PROVIDER == "openai" and not self.OPENAI_EMBEDDING_API_KEY:
            problems.append("EMBEDDING_PROVIDER=openai 但未配置 OPENAI_EMBEDDING_API_KEY")
        if self.RAG_TOP_K <= 0:
            problems.append("RAG_TOP_K 必须大于 0")
        if self.LLM_MAX_TOOL_ROUNDS <= 0:
            problems.append("LLM_MAX_TOOL_ROUNDS 必须大于 0")
        if not self.llm_configured and not self.LLM_OFFLINE_FALLBACK:
            problems.append("未配置 DEEPSEEK_API_KEY 且关闭了离线降级，/chat 将无法回答")
        return problems


# 全局唯一配置实例：其它模块统一 ``from config import config``
config = Settings()

# 让 sentence-transformers / huggingface_hub 走镜像（未显式设置时）
if config.HF_ENDPOINT and not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def setup_logging(level: str | None = None, log_file: Path | None = None) -> logging.Logger:
    """初始化全局日志（幂等，可重复调用）。

    Args:
        level: 日志级别，默认取 config.LOG_LEVEL。
        log_file: 日志文件路径，默认 config.LOG_FILE；传 None 表示仅输出到控制台。

    Returns:
        根 logger。
    """
    log_level = getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO)
    root = logging.getLogger()

    # 已经配置过就直接返回，避免 Flask reloader 下重复输出
    if getattr(root, "_crm_configured", False):
        root.setLevel(log_level)
        return root

    root.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # 文件输出（失败不影响启动，例如只读文件系统）
    target = config.LOG_FILE if log_file is None else log_file
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # pragma: no cover - 依赖文件系统状态
        root.warning("日志文件初始化失败（仅控制台输出）：%s", exc)

    root._crm_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.Logger:
    """获取带项目前缀的 logger，业务模块统一使用该函数。"""
    return logging.getLogger(name)
