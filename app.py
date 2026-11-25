"""SeaChatInterview — AI 客服咨询系统（Flask 入口）。

系统能力：
- ``POST /chat``：手机号发起咨询，SSE 流式返回（含意图识别、工具调用、画像更新）；
- ``POST /vectorstore/add|update|delete|search``、``GET /vectorstore/list``：向量库对外维护接口；
- ``POST /database/query``：SQL Agent（自然语言 → 只读 SQL → 中文总结）；
- ``GET /health``：健康检查（LLM / 嵌入模型 / 向量库 / 数据库 / 搜索状态）；
- ``GET /``：内置网页版聊天界面。

启动方式：``python app.py``（或 ``python -m flask --app app run``）。
"""

from __future__ import annotations

import datetime
import json
from typing import Any, Iterator

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from config import config, get_logger, setup_logging
from database import db
from database.sql_agent import answer_with_sql_agent
from llm.client import get_llm_status
from llm.web_search import get_search_status
from services.chat import process_chat, stream_chat
from vectorstore import embeddings, store

logger = get_logger("crm.app")

# 常见错误码 → 中文提示
HTTP_ERROR_MESSAGES = {
    400: "请求参数有误，请检查后重试",
    404: "接口不存在，请检查访问地址",
    405: "请求方法不被支持",
    413: "请求体过大",
    500: "服务内部错误，请稍后重试",
}


def _json(payload: dict[str, Any], status: int = 200) -> tuple[Response, int]:
    """返回 UTF-8 JSON 响应（中文不转义）。"""
    return jsonify(payload), status


def _error(message: str, status: int = 400) -> tuple[Response, int]:
    """返回统一格式的错误 JSON。"""
    return jsonify({"ok": False, "error": message}), status


def _sse(event: dict[str, Any]) -> str:
    """把事件字典序列化成 SSE 数据帧。"""
    payload = json.dumps(event, ensure_ascii=False, default=str)
    return f"data: {payload}\n\n"


def _vectorstore_update_wrapper(action: str, handler) -> tuple[Response, int]:
    """统一包一层异常处理，把 ValueError 转成 400，其他异常转 500。"""
    try:
        return handler()
    except ValueError as exc:
        logger.warning("向量库 %s 参数错误：%s", action, exc)
        return _error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001 - 向量库接口不允许向外抛异常
        logger.exception("向量库 %s 失败：%s", action, exc)
        return _error(f"向量库操作失败：{exc}", 500)


def create_app() -> Flask:
    """创建并初始化 Flask 应用（日志 / SQLite / 向量库 / 嵌入模型 / LLM 状态）。

    Returns:
        初始化完成的 Flask 应用实例。
    """
    setup_logging()
    config.ensure_dirs()
    config.validate()

    logger.info("=" * 72)
    logger.info("SeaChatInterview AI CRM 正在启动……")
    logger.info("数据目录：%s", config.DATA_DIR)

    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config["JSON_AS_ASCII"] = False
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

    # ---------- 1. SQLite 初始化 ----------
    try:
        db.init_db()
        stats = db.get_database_stats()
        logger.info(
            "[初始化] SQLite 完成：%s（users=%d，chat_history=%d，courses=%d，career_info=%d）",
            config.SQLITE_PATH,
            stats.get("users", 0),
            stats.get("chat_history", 0),
            stats.get("courses", 0),
            stats.get("career_info", 0),
        )
    except Exception as exc:  # noqa: BLE001 - 初始化失败仍允许服务启动以便排查
        logger.exception("[初始化] SQLite 失败：%s", exc)

    # ---------- 2. 向量库初始化（含示例文档） ----------
    try:
        store.init_vectorstores()
        collections = store.get_collection_stats()
        summary = "，".join(f"{item['name']}={item['count']}" for item in collections) or "空"
        logger.info("[初始化] 向量库完成：%s（%s）", config.CHROMA_DIR, summary)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[初始化] 向量库失败：%s", exc)

    # ---------- 3. 嵌入模型状态 ----------
    try:
        embedding_info = embeddings.get_embedding_info()
        logger.info(
            "[初始化] 嵌入模型完成：provider=%s，model=%s，dim=%s",
            embedding_info.get("provider"),
            embedding_info.get("model"),
            embedding_info.get("dim"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[初始化] 嵌入模型失败：%s", exc)

    # ---------- 4. LLM 状态 ----------
    try:
        llm_status = get_llm_status()
        if llm_status.get("available"):
            logger.info("[初始化] DeepSeek 完成：model=%s，base_url=%s", llm_status.get("model"), llm_status.get("base_url"))
        else:
            logger.warning("[初始化] DeepSeek 未配置 API Key，将使用离线规则模式（功能可演示，质量受限）")
    except Exception as exc:  # noqa: BLE001
        logger.exception("[初始化] LLM 状态检查失败：%s", exc)

    logger.info("启动完成，监听地址：http://%s:%s", config.HOST, config.PORT)
    logger.info("=" * 72)

    register_routes(app)
    register_error_handlers(app)
    register_cors(app)
    return app


def register_cors(app: Flask) -> None:
    """允许浏览器跨域调用接口（演示用，生产应按域名收紧）。"""

    @app.after_request
    def _add_cors_headers(response: Response) -> Response:
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.setdefault("Access-Control-Max-Age", "86400")
        return response


def register_error_handlers(app: Flask) -> None:
    """统一 JSON 错误响应，避免浏览器/调用方收到 HTML 错误页。"""

    @app.errorhandler(404)
    def _not_found(error):  # noqa: ANN001, ARG001 - Flask 约定签名
        return _error(HTTP_ERROR_MESSAGES[404], 404)

    @app.errorhandler(405)
    def _method_not_allowed(error):  # noqa: ANN001, ARG001
        return _error(HTTP_ERROR_MESSAGES[405], 405)

    @app.errorhandler(500)
    def _internal_error(error):  # noqa: ANN001
        logger.exception("服务内部错误：%s", error)
        return _error(HTTP_ERROR_MESSAGES[500], 500)

    @app.errorhandler(Exception)
    def _unhandled_exception(error):  # noqa: ANN001
        from werkzeug.exceptions import HTTPException

        if isinstance(error, HTTPException):
            return _error(HTTP_ERROR_MESSAGES.get(error.code, error.description), error.code or 500)
        logger.exception("未捕获异常：%s", error)
        return _error(HTTP_ERROR_MESSAGES[500], 500)


def register_routes(app: Flask) -> None:
    """注册全部 HTTP 路由。"""

    # ------------------------------ 页面 ------------------------------
    @app.get("/")
    def index():
        """内置聊天页面。"""
        return render_template("index.html")

    # ------------------------------ 健康检查 ------------------------------
    @app.get("/health")
    def health():
        """健康检查：汇总各组件的初始化状态。"""
        payload: dict[str, Any] = {
            "ok": True,
            "status": "ok",
            "service": "SeaChatInterview AI CRM",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "data_dir": str(config.DATA_DIR),
        }
        try:
            payload["llm"] = get_llm_status()
        except Exception as exc:  # noqa: BLE001
            payload["llm"] = {"available": False, "error": str(exc)}
        try:
            payload["embedding"] = embeddings.get_embedding_info()
        except Exception as exc:  # noqa: BLE001
            payload["embedding"] = {"ready": False, "error": str(exc)}
        try:
            payload["vectorstore"] = store.get_collection_stats()
        except Exception as exc:  # noqa: BLE001
            payload["vectorstore"] = {"error": str(exc)}
        try:
            payload["database"] = db.get_database_stats()
        except Exception as exc:  # noqa: BLE001
            payload["database"] = {"error": str(exc)}
        try:
            payload["search"] = get_search_status()
        except Exception as exc:  # noqa: BLE001
            payload["search"] = {"error": str(exc)}
        return jsonify(payload)

    # ------------------------------ 对话 ------------------------------
    @app.post("/chat")
    def chat():
        """发起咨询。默认 SSE 流式返回；``stream=false`` 时返回普通 JSON。"""
        body = request.get_json(silent=True) or {}
        phone_number = body.get("phone_number") or body.get("phone") or request.form.get("phone_number")
        query = body.get("query") or body.get("message") or request.form.get("query")
        want_stream = body.get("stream", True)
        if isinstance(want_stream, str):
            want_stream = want_stream.lower() not in {"false", "0", "no", "off"}

        if not str(query or "").strip():
            return _error("请提供 query 参数（用户问题）", 400)
        if not str(phone_number or "").strip():
            return _error("请提供 phone_number 参数（用户手机号）", 400)

        if not want_stream:
            result = process_chat(str(phone_number), str(query))
            if not result.get("ok"):
                return _error(str(result.get("error") or "处理失败"), 400)
            return jsonify(result)

        def generate() -> Iterator[str]:
            try:
                for event in stream_chat(str(phone_number), str(query)):
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001 - 流式响应内部异常兜底
                logger.exception("流式对话异常：%s", exc)
                yield _sse({"type": "error", "message": "服务内部错误，请稍后重试"})
            finally:
                yield "data: [DONE]\n\n"

        response = Response(stream_with_context(generate()), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Connection"] = "keep-alive"
        return response

    # ------------------------------ 向量库维护接口 ------------------------------
    @app.post("/vectorstore/add")
    def vectorstore_add():
        """新增向量文档：``{collection, documents, metadatas?, ids?}``。"""

        def handler():
            body = request.get_json(silent=True) or {}
            collection = str(body.get("collection") or "").strip()
            documents = body.get("documents")
            if not collection:
                raise ValueError("缺少 collection 参数：" + "、".join(config.COLLECTIONS))
            if not isinstance(documents, list) or not documents:
                raise ValueError("documents 必须是非空字符串数组")
            ids = store.add_documents(
                collection,
                [str(item) for item in documents],
                body.get("metadatas"),
                body.get("ids"),
            )
            logger.info("[向量库] 新增文档：collection=%s，count=%d", collection, len(ids))
            return jsonify(
                {
                    "ok": True,
                    "collection": collection,
                    "ids": ids,
                    "count": len(ids),
                    "stats": store.count_documents(collection),
                }
            )

        return _vectorstore_update_wrapper("add", handler)

    @app.post("/vectorstore/update")
    def vectorstore_update():
        """更新向量文档：``{collection, ids, documents, metadatas?}``。"""

        def handler():
            body = request.get_json(silent=True) or {}
            collection = str(body.get("collection") or "").strip()
            ids = body.get("ids")
            documents = body.get("documents")
            if not collection:
                raise ValueError("缺少 collection 参数：" + "、".join(config.COLLECTIONS))
            if not isinstance(ids, list) or not ids:
                raise ValueError("ids 必须是非空数组")
            if not isinstance(documents, list) or not documents:
                raise ValueError("documents 必须是非空字符串数组")
            if len(ids) != len(documents):
                raise ValueError("ids 与 documents 长度必须一致")
            store.update_documents(
                collection,
                [str(item) for item in ids],
                [str(item) for item in documents],
                body.get("metadatas"),
            )
            logger.info("[向量库] 更新文档：collection=%s，count=%d", collection, len(ids))
            return jsonify({"ok": True, "collection": collection, "ids": [str(item) for item in ids], "updated": len(ids)})

        return _vectorstore_update_wrapper("update", handler)

    @app.post("/vectorstore/delete")
    def vectorstore_delete():
        """删除向量文档：``{collection, ids}``。"""

        def handler():
            body = request.get_json(silent=True) or {}
            collection = str(body.get("collection") or "").strip()
            ids = body.get("ids")
            if not collection:
                raise ValueError("缺少 collection 参数：" + "、".join(config.COLLECTIONS))
            if not isinstance(ids, list) or not ids:
                raise ValueError("ids 必须是非空数组")
            store.delete_documents(collection, [str(item) for item in ids])
            logger.info("[向量库] 删除文档：collection=%s，ids=%s", collection, ids)
            return jsonify(
                {
                    "ok": True,
                    "collection": collection,
                    "deleted": [str(item) for item in ids],
                    "stats": store.count_documents(collection),
                }
            )

        return _vectorstore_update_wrapper("delete", handler)

    @app.get("/vectorstore/list")
    def vectorstore_list():
        """列出全部向量集合及文档数量。"""
        try:
            collections = store.list_collections()
            return jsonify({"ok": True, "collections": collections, "count": len(collections)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("向量库列表查询失败：%s", exc)
            return _error(f"向量库列表查询失败：{exc}", 500)

    @app.post("/vectorstore/search")
    def vectorstore_search():
        """向量检索：``{collection, query, top_k?}``（便于验收「更新后 /chat 可检索」）。"""
        try:
            body = request.get_json(silent=True) or {}
            collection = str(body.get("collection") or "").strip()
            query = str(body.get("query") or "").strip()
            if not collection:
                raise ValueError("缺少 collection 参数")
            if not query:
                raise ValueError("缺少 query 参数")
            top_k = int(body.get("top_k") or config.RAG_TOP_K)
            results = store.search(collection, query, top_k=top_k)
            return jsonify({"ok": True, "collection": collection, "query": query, "results": results, "count": len(results)})
        except ValueError as exc:
            return _error(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("向量检索失败：%s", exc)
            return _error(f"向量检索失败：{exc}", 500)

    @app.post("/vectorstore/init")
    def vectorstore_init():
        """重建/补齐示例向量数据（幂等）。"""
        try:
            store.init_vectorstores()
            return jsonify({"ok": True, "collections": store.get_collection_stats()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("向量库初始化失败：%s", exc)
            return _error(f"向量库初始化失败：{exc}", 500)

    # ------------------------------ SQL Agent ------------------------------
    @app.post("/database/query")
    def database_query():
        """SQL Agent：自然语言 → 只读 SQL → 执行 → 中文总结。"""
        try:
            body = request.get_json(silent=True) or {}
            question = str(body.get("question") or body.get("query") or "").strip()
            if not question:
                return _error("请提供 question 参数（自然语言问题）", 400)
            result = answer_with_sql_agent(question)
            status = 200 if result.get("ok") else 400
            return jsonify(result), status
        except Exception as exc:  # noqa: BLE001 - SQL Agent 不允许抛异常给 Flask
            logger.exception("SQL Agent 处理失败：%s", exc)
            return _error(f"SQL Agent 处理失败：{exc}", 500)

    @app.get("/database/schema")
    def database_schema():
        """返回数据库表结构说明（调试用）。"""
        try:
            return jsonify({"ok": True, "schema": db.get_schema_description(), "tables": db.list_tables()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("读取数据库结构失败：%s", exc)
            return _error(f"读取数据库结构失败：{exc}", 500)


if __name__ == "__main__":
    application = create_app()
    logger.info("开发服务器启动：http://%s:%s（Ctrl+C 停止）", config.HOST, config.PORT)
    application.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
