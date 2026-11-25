"""pytest 全局配置：测试环境隔离（数据目录、离线模式、日志级别）。

设计原因：
- 测试必须与开发数据隔离：这里在**导入任何项目模块之前**把 ``CRM_DATA_DIR`` 指向临时目录；
- 测试不依赖外网与 API Key：强制关闭 DeepSeek、网络搜索，嵌入层使用 hash 后端；
- 通过 session 级自动 fixture 只初始化一次数据库与向量库，避免每个用例重复建表。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# 项目根目录加入 sys.path（保证 `import app` / `import config` 可用）
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 必须在导入 config 之前设置环境变量（config 在导入时读取）
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="crm_test_")
os.environ["CRM_DATA_DIR"] = _TEST_DATA_DIR
os.environ["DEEPSEEK_API_KEY"] = ""          # 测试不调用真实大模型
os.environ["LLM_OFFLINE_FALLBACK"] = "true"  # 走规则引擎分支
os.environ["EMBEDDING_PROVIDER"] = "hash"    # 不下载 BGE 模型，测试更快
os.environ["WEB_SEARCH_ENABLED"] = "false"   # 不访问外网
os.environ["WEB_SEARCH_PROVIDER"] = "none"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["FLASK_DEBUG"] = "false"

import pytest  # noqa: E402

from config import config  # noqa: E402


_initialized = {"done": False}


@pytest.fixture(scope="session", autouse=True)
def prepared_environment() -> dict:
    """会话级初始化：创建目录、建表、灌入示例数据、初始化向量库（只执行一次）。"""
    if not _initialized["done"]:
        config.ensure_dirs()
        from database.db import init_db
        from vectorstore.store import init_vectorstores

        init_db()
        init_vectorstores()
        _initialized["done"] = True
    return {"data_dir": str(config.DATA_DIR)}


@pytest.fixture(scope="session")
def flask_app():
    """Flask 应用实例（session 级，避免重复初始化 ChromaDB）。"""
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture
def client(flask_app):
    """Flask 测试客户端。"""
    return flask_app.test_client()
