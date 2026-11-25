# AI CRM 接口契约（冻结版 v1）

> 本文件是**并行开发的唯一依据**：所有模块必须严格按照这里定义的函数名、签名、返回结构实现。
> 开发环境：Windows，Python 3.14（代码需兼容 3.10+ 语法）。工作目录：`C:\Users\IKUN\Desktop\crm项目`。
> 已完成（**不要修改**）：`config.py`、`llm/prompts.py`、`tests/conftest.py`、`requirements.txt`、`.env.example`。

## 0. 全局约定

1. **无 API Key 开发**：当前环境没有 `DEEPSEEK_API_KEY`。任何 LLM 调用必须允许失败：
   - 更低层（`llm/client.py`）在无 Key 时抛 `LLMUnavailableError`，不要把假答案伪装成模型输出；
   - 更上层（意图、SQL Agent、画像抽取）捕获异常后走**规则/关键词兜底**，并让返回结构里的 `mode` 标记为 `"rule"`。
2. **只允许标准库 + `requirements.txt` 里的依赖**（Flask、openai、chromadb、sentence-transformers、numpy、requests、python-dotenv、pytest）。禁止引入 bs4/duckduckgo_search 等新依赖。
3. **禁止** `pass` / `TODO` / `NotImplementedError` 占位；每个函数写中文 docstring（用途、参数、返回值），关键逻辑写中文注释。
4. **异常处理**：数据库、向量库、LLM、网络搜索都要 try/except，返回友好结构或空列表，不能把异常抛给 Flask 导致 500。
5. **日志**：`from config import get_logger`，例如 `logger = get_logger("crm.database")`；关键节点（初始化、工具调用、SQL 执行、向量库更新、降级）都要 `logger.info/warning`。
6. **配置**：只从 `config.config` 读取路径/模型/阈值，禁止硬编码 `data/crm.db`、模型名等。
7. **文件归属**：只写自己 scope 的文件，**绝不修改**别人 scope 的文件；`llm/prompts.py`、`config.py`、`tests/conftest.py` 归 Lead，任何人不改。
8. **测试**：`tests/conftest.py` 已把数据目录隔离到临时目录，并强制离线模式（无 Key、hash 嵌入、无网络搜索）。你的测试文件直接 import 模块即可，不要删改 conftest。
9. 提交前自检：`python -m py_compile <你写的所有 .py>`；依赖装好后跑自己的测试 `python -m pytest tests/你的测试.py -q`。

## 1. `config.py`（已就绪，只读使用）

```python
from config import config, setup_logging, get_logger
```

可用属性（都有默认值，测试已通过环境变量覆盖）：

| 属性 | 说明 |
|---|---|
| `config.DATA_DIR` / `config.SQLITE_PATH` / `config.CHROMA_DIR` / `config.LOG_DIR` / `config.LOG_FILE` | `Path` 对象 |
| `config.DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | 对话模型配置 |
| `config.LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT` / `LLM_MAX_TOOL_ROUNDS` / `LLM_OFFLINE_FALLBACK` | 对话参数 |
| `config.EMBEDDING_PROVIDER`（bge/hash/openai）/ `EMBEDDING_MODEL` / `EMBEDDING_DIM` / `EMBEDDING_DEVICE` / `EMBEDDING_BATCH_SIZE` / `HF_ENDPOINT` / `OPENAI_EMBEDDING_*` | 嵌入配置 |
| `config.COLLECTION_MARKET`（"market"）/ `COLLECTION_COURSE`（"course"）/ `COLLECTIONS`（tuple）/ `RAG_TOP_K` | 向量库配置 |
| `config.WEB_SEARCH_ENABLED` / `WEB_SEARCH_PROVIDER` / `WEB_SEARCH_TOP_K` / `WEB_SEARCH_TIMEOUT` / `SERPAPI_API_KEY` | 搜索配置 |
| `config.SQL_READONLY` / `SQL_MAX_ROWS` | SQL 安全 |
| `config.CHAT_HISTORY_LIMIT` / `MAX_QUERY_LENGTH` | 聊天配置 |
| `config.llm_configured`（bool）/ `config.ensure_dirs()` / `config.masked()` / `config.validate()` / `config.override_data_dir(path)` | 工具方法 |

## 2. `database/db.py`（负责人：data-agent）

```python
ALLOWED_USER_FIELDS = ("name", "intention_direction", "budget", "stage", "notes")

def get_connection() -> sqlite3.Connection:
    """返回 SQLite 连接：row_factory=sqlite3.Row，timeout=10，foreign_keys=ON，自动建父目录。"""

def init_db() -> None:
    """执行 database/schema.sql 建表（幂等），并在 courses / career_info 为空时插入示例数据。"""

def create_user(phone_number: str, **fields) -> dict:
    """用户不存在则创建，存在则更新传入字段，返回完整用户行 dict（含 created_at/updated_at）。"""

def get_user_info(phone_number: str) -> dict | None:
    """按手机号查询用户画像，不存在返回 None。"""

def update_user_info(phone_number: str, **fields) -> dict:
    """更新画像（不存在则创建）；只接受 ALLOWED_USER_FIELDS；budget 转 float；stage 只允许 新咨询/已了解/已报名，非法值忽略并记日志；返回最新行 dict。"""

def fetch_chat_history(phone_number: str, limit: int | None = None) -> list[dict]:
    """按时间正序返回聊天历史；limit 表示只取最近 N 条（仍按正序）。元素含 id/phone_number/role/content/intent_code/created_at。"""

def insert_message(phone_number: str, role: str, content: str, intent_code: int | None = None) -> int:
    """插入一条聊天消息，返回自增 id；role 只允许 user/assistant，非法抛 ValueError。"""

def update_chat_history(phone_number: str, query: str, response: str, intent_code: int | None = None) -> None:
    """保存一轮完整对话（先 user 后 assistant），供伪代码 update_chat_history 使用。"""

def is_readonly_sql(sql: str) -> bool:
    """判断是否单条只读 SELECT/WITH 语句（去掉注释与首尾空白后判断）。"""

def execute_sql(sql: str) -> list[dict]:
    """只读执行 SQL：非法/多语句/写操作抛 ValueError(FORBIDDEN_SQL_MESSAGE)；最多返回 config.SQL_MAX_ROWS 行 dict。"""

def get_schema_description() -> str:
    """返回注入 SQL Agent 提示词的中文表结构说明（表名、字段、类型、含义 + 每表示例数据）。"""

def list_tables() -> list[str]: ...
def get_database_stats() -> dict[str, int]:
    """返回 {"users": n, "chat_history": n, "courses": n, "career_info": n}。"""

FORBIDDEN_SQL_MESSAGE = "出于安全限制，只能执行只读的 SELECT 查询。"
```

要点：`database/schema.sql` 包含提示词要求的 4 张表 + 索引；示例数据：≥6 门课程（Python 全栈/数据分析/AI 应用开发/大模型工程/前端/算法，含价格、时长、简介）、≥6 条职业信息（AI 应用开发/大模型算法/数据分析/Python 后端/算法工程师/产品经理，含 salary_min/max 与行业）。中文课程名要能被语义检索与 SQL 查询命中。

## 3. `database/sql_agent.py`（负责人：data-agent）

```python
SQL_TOOL_SCHEMA: dict
# OpenAI function calling schema：
# {"type": "function", "function": {"name": "query_database", "description": "...", "parameters": {"type": "object", "properties": {"sql": {"type": "string", "description": "只读 SELECT 语句"}}, "required": ["sql"]}}}

def answer_with_sql_agent(question: str, history: list[dict] | None = None) -> dict:
    """自然语言 → SQL → 执行 → 中文总结，返回：
    {"ok": bool, "answer": str, "sql": list[str], "rows": list[dict],
     "mode": "llm" | "rule" | "error", "error": str | None}
    实现：注入 get_schema_description() 后走 llm.client.run_tool_calling_loop（executor 用 db.execute_sql）；
    LLM 不可用或异常时走 rule_based_answer()；任何情况下都不抛异常。"""

def rule_based_answer(question: str) -> dict:
    """离线规则兜底：关键词 → 固定 SQL（课程列表/最贵课程/课程价格/薪资阈值/职业列表/用户统计），
    返回结构与 answer_with_sql_agent 相同，mode="rule"。"""

def generate_sql(question: str) -> str | None:
    """让模型只输出一条 SQL（非 tool 调用）；LLM 不可用返回 None。"""
```

规则必须覆盖验收用例：「有哪些课程」→ 课程列表；「薪资 2 万以上的职业有哪些」→ 解析 `2万=20000`，按 `salary_max >= 20000` 查询；「课程多少钱/最贵的课程」→ 价格查询；无法映射时返回友好说明且 `ok=False`。依赖 `llm/client.py` 的 `run_tool_calling_loop`（见第 5 节）。

## 4. `vectorstore/embeddings.py` / `vectorstore/store.py`（负责人：vector-agent）

```python
# embeddings.py
def embed_texts(texts: list[str]) -> list[list[float]]
def embed_query(text: str) -> list[float]
def get_embedding_dim() -> int
def get_embedding_info() -> dict
    # {"configured_provider","provider","model","dim","backend","ready","detail"}
def reset_embedding_backend() -> None
```

- `EMBEDDING_PROVIDER=bge`：懒加载 `sentence_transformers.SentenceTransformer(config.EMBEDDING_MODEL, device=...)`，`encode(..., normalize_embeddings=True)`；首次加载失败（无依赖/无网络/模型下载失败）→ 记 warning 并**自动降级为 hash 后端**，不能崩溃。下载走 `HF_ENDPOINT` 镜像（config 已设置环境变量）。
- `hash`：纯标准库确定性向量（不用 Python 内置 `hash()`，用 `hashlib.sha1`）：小写化后提取字符 1~3-gram 特征 → 哈希到 `EMBEDDING_DIM` 维 → 符号哈希累加 → L2 归一化。相同文本必须得到相同向量，相近文本余弦相似度更高。
- `openai`：仅在配置了 `OPENAI_EMBEDDING_API_KEY` 时调用兼容接口，否则降级 hash。
- 空列表返回 `[]`；单条文本返回 `[vec]`。全程有 logger。

```python
# store.py
class VectorStoreError(Exception): ...
def init_vectorstores() -> None          # 幂等：创建 market/course 两个集合；集合为空时灌入示例文档
def add_documents(collection_name: str, documents: list[str],
                  metadatas: list[dict] | None = None,
                  ids: list[str] | None = None) -> list[str]
def update_documents(collection_name: str, ids: list[str],
                     documents: list[str],
                     metadatas: list[dict] | None = None) -> None
def delete_documents(collection_name: str, ids: list[str]) -> None
def search(collection_name: str, query: str, top_k: int = 3) -> list[dict]
    # [{"id": str, "document": str, "metadata": dict, "distance": float}]，按 distance 升序
def list_collections() -> list[dict]     # [{"name": str, "count": int}]
def get_collection_stats() -> list[dict] # 同 list_collections
def count_documents(collection_name: str) -> int
def get_documents(collection_name: str, limit: int = 100) -> list[dict]  # [{"id","document","metadata"}]
def clear_collection(collection_name: str) -> None   # 测试用
def reset_vectorstore() -> None                       # 测试用：重置缓存的 chroma client
```

- 使用 `chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False, allow_reset=True))`，**必须真正持久化到 config.CHROMA_DIR**。
- 集合名校验：只允许 `config.COLLECTIONS`，非法抛 `ValueError`。
- 写入/检索**显式传 embeddings**（`embed_texts` / `embed_query`），不要依赖 chroma 默认 ONNX 向量函数（本机无网络模型）。
- 集合用余弦距离（创建时 metadata `{"hnsw:space": "cosine"}`）。
- `search` 后端异常时记日志并返回 `[]`；`ids` 不传时自动生成 `f"{collection_name}-{uuid4().hex}"`；`add_documents` 参数非法抛 `ValueError`。
- 示例文档：course 集合 ≥6 条（与 SQLite 课程对应，含价格/时长/适合人群），market 集合 ≥6 条（岗位职责/薪资/技能要求/行业趋势）。用固定 id，如 `course-001`、`market-001`。
- `init_vectorstores()` 只灌入空集合，避免重复插入；可重复调用。

## 5. `llm/client.py` / `llm/intent.py` / `llm/web_search.py`（负责人：llm-agent）

```python
# client.py
class LLMUnavailableError(RuntimeError): """无 Key / 客户端不可用。"""
class LLMResponseError(RuntimeError):    """模型调用失败/返回非法。"""

def get_client() -> openai.OpenAI | None
def is_llm_available() -> bool
def get_llm_status() -> dict     # {"available": bool, "model": str, "base_url": str}
def reset_client() -> None       # 测试用
def chat_completion(messages: list[dict], tools: list[dict] | None = None,
                    tool_choice=None, temperature: float | None = None,
                    max_tokens: int | None = None) -> dict:
    """非流式调用，返回 {"role": "assistant", "content": str, "tool_calls": list[dict]}；
    tool_calls 元素为 {"id": str, "name": str, "arguments": dict}。无 Key 抛 LLMUnavailableError。"""

def run_tool_calling_loop(messages: list[dict], tools: list[dict],
                          tool_executor: Callable[[str, dict], Any],
                          max_rounds: int | None = None,
                          temperature: float | None = None,
                          on_event: Callable[[dict], None] | None = None) -> dict:
    """完整 function calling 循环（对应提示词里的 5 步），返回
    {"content": str, "tool_calls": [{"id","name","arguments","result"}], "rounds": int, "mode": "llm"}。
    tool_executor 返回可 JSON 序列化对象；tool 消息 role="tool" 并带 tool_call_id/name。
    超轮数时最后强制不带 tools 再请求一次；工具执行异常转成 {"ok": False, "error": ...} 结果继续。"""

def stream_completion(messages: list[dict], temperature: float | None = None,
                      max_tokens: int | None = None) -> Iterator[str]:
    """流式返回文本增量；无 Key 抛 LLMUnavailableError。"""

def run_streaming_tool_loop(messages: list[dict], tools: list[dict] | None,
                            tool_executor: Callable[[str, dict], Any],
                            max_rounds: int | None = None,
                            temperature: float | None = None,
                            on_event: Callable[[dict], None] | None = None) -> Iterator[dict]:
    """流式版 function calling 循环。依次 yield：
    {"type": "tool_call", "name": str, "arguments": dict, "result": Any}（每执行一个工具）
    {"type": "delta", "content": str}（最终回答的文本增量）
    {"type": "final", "content": str, "tool_calls": list[dict], "rounds": int}
    无 Key 时在开始阶段抛 LLMUnavailableError；流中断抛 LLMResponseError。"""

def simple_chat(messages: list[dict], **kwargs) -> str
def extract_json(text: str) -> dict | None
```

```python
# intent.py
INTENT_CAREER = 1; INTENT_COURSE = 2; INTENT_OTHER = 0
DETECT_INTENT_TOOL_SCHEMA: dict   # detect_intent 工具 schema（intent_code enum [1,2,0], reason）
def keyword_intent(query: str) -> int
def detect_intent_code(query: str, history: list[dict] | None = None) -> int
def detect_intent_with_reason(query: str, history: list[dict] | None = None) -> dict
    # {"intent_code": int, "reason": str, "mode": "llm"|"rule"}
```

- 意图 LLM 路径：用 `llm/prompts.py::build_intent_messages(query, history)` + `chat_completion(tools=[DETECT_INTENT_TOOL_SCHEMA], tool_choice={"type":"function","function":{"name":"detect_intent"}})`；解析失败/异常/不可用 → `keyword_intent`。
- 关键词规则必须通过验收：「AI 职业规划」→1，「Python 课程多少钱」→2，「今天天气」→0；课程/价格/报名/试听/上课/培训/学习/证书 偏向 2；职业/岗位/就业/薪资/工资/转行/前景/招聘/面试/行业/规划 偏向 1；两者都命中时按加权分数取高者，无法判断 → 0。

```python
# web_search.py
def web_search(query: str, top_k: int = 5) -> list[dict]
    # [{"title": str, "snippet": str, "url": str, "source": str, "provider": str}]
def is_search_available() -> bool
def get_search_status() -> dict   # {"enabled","provider","last_error","fallback_used","calls"}
def reset_search_status() -> None
```

- provider=`serpapi`（有 Key 时）：`https://serpapi.com/search.json?engine=google&q=...&api_key=...`（requests，超时 `config.WEB_SEARCH_TIMEOUT`）。
- provider=`duckduckgo`：依次尝试 `https://html.duckduckgo.com/html/?q=...`、`https://lite.duckduckgo.com/lite/?q=...`，用标准库 `html.parser`/正则解析标题、摘要、链接（不要引入 bs4）。
- 全部失败或 `WEB_SEARCH_ENABLED=false`：**降级**读 SQLite `career_info` 返回本地职业信息（`source="本地知识库兜底"`, `provider="local"`），并 `logger.warning`；仍无结果返回 `[]`。任何情况都不抛异常。
- 独立模块：不得 import `tools/`，避免循环依赖。

## 6. `tools/functions.py`（负责人：llm-agent）

```python
TOOL_SCHEMAS: list[dict]          # 恰好 6 个工具（OpenAI tools 格式，type=function）
def get_openai_tools(names: list[str] | None = None) -> list[dict]
def get_tool_names() -> list[str]
def execute_tool(name: str, arguments: dict | None = None,
                 context: dict | None = None) -> dict
    # {"ok": bool, "tool": str, "result": Any, "error": str | None}；永不抛异常
```

6 个工具：`detect_intent`、`web_search`、`search_market_docs`、`search_course_docs`、`query_database`、`save_chat_record`。
- `detect_intent`：schema 复用 `llm.intent.DETECT_INTENT_TOOL_SCHEMA`；执行时校验 intent_code ∈ {1,2,0}，返回 `{"intent_code","reason"}`。
- `web_search`：参数 `query`、`top_k`；调 `llm.web_search.web_search`，返回 `{"query","results","count"}`。
- `search_market_docs` / `search_course_docs`：参数 `query`、`top_k`；调 `vectorstore.store.search(config.COLLECTION_MARKET/COLLECTION_COURSE)`，返回 `{"query","collection","results","count"}`。
- `query_database`：schema 复用 `database.sql_agent.SQL_TOOL_SCHEMA`；执行 `db.execute_sql`，返回 `{"sql","rows","row_count"}`。
- `save_chat_record`：参数 `phone_number?`、`role`、`content`；`context` 可带 `phone_number`；调 `db.insert_message`，返回 `{"id","saved": True}`。

## 7. 集成说明（Lead 负责，agent 不用写）

`services/`、`app.py`、`tests/test_chat.py`、`tests/test_api.py`、`tests/mock_llm_server.py`、`README.md` 由 Lead 编写。你们只需保证上面的接口可被调用。

## 8. 交付时请报告

- 实际写入的文件清单；
- 自检命令与结果（py_compile / pytest 输出摘要）；
- 与契约不一致之处（如 chromadb 1.5.9 API 差异、返回结构变化），以及你做的处理。
