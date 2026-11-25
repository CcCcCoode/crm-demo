# SeaChatInterview — AI 智能客服咨询系统（AI CRM）

基于 **Flask + DeepSeek-Flash（OpenAI 兼容 SDK）+ 本地 ChromaDB + 本地 BGE 嵌入 + SQLite** 的智能客服咨询系统。
用户以手机号发起咨询，系统读取历史对话与用户画像，自动完成 **意图识别（1=AI 职业咨询 / 2=课程咨询 / 0=其他）→ 工具调用（联网搜索 / 知识库检索 / SQL 查询）→ 画像驱动回答 → 写回数据库并更新画像 → 流式返回** 的完整闭环。

---

## 1. 功能特性

| 能力 | 说明 |
| --- | --- |
| 意图识别 | LLM function calling（`detect_intent`）+ 关键词规则双通道，返回 1 / 2 / 0 |
| Function Calling | 6 个工具：`detect_intent`、`web_search`、`search_market_docs`、`search_course_docs`、`query_database`、`save_chat_record` |
| 本地向量库 | ChromaDB 本地持久化（`data/chroma`），两个集合：`market`（AI 就业市场资料）、`course`（课程资料）；支持增删改查 |
| 本地嵌入 | 默认 `BAAI/bge-small-zh-v1.5`（sentence-transformers，CPU 可用）；模型不可用时自动降级为确定性 hash 嵌入，保证系统可用 |
| 联网搜索 | Bing（cn.bing.com，默认）/ DuckDuckGo / SerpAPI；多源自动互相兜底，全部失败时降级到本地 `career_info` 数据并进入 120 秒熔断冷却（期间不再联网，避免每轮都等待超时），绝不打断对话 |
| SQL Agent | 自然语言 → 只读 SQL（仅 SELECT/WITH）→ 执行 → 中文总结；带规则兜底 |
| 用户画像 | 自动提取姓名 / 求职方向 / 预算 / 阶段 / 备注并写回 `users` 表，拼入 System Prompt 决定回答策略 |
| 流式对话 | `POST /chat` 默认 SSE 流式（`text/event-stream`），也支持 `stream=false` 返回整包 JSON |
| 向量库运维接口 | 对外提供 `/vectorstore/add|update|delete|list|search|init`，更新后 `/chat` 立即可检索 |
| 离线降级 | 未配置 `DEEPSEEK_API_KEY` 时，工具调用与 RAG 照常执行，回答由规则模板生成，系统仍可完整演示与验收 |

---

## 2. 技术架构

```
用户 (手机号 + 问题)
        │
        ▼
┌────────────────────────── Flask app.py ──────────────────────────┐
│  POST /chat                                                      │
│     │                                                            │
│     ▼                                                            │
│  services/chat.py  流式对话主流程                                 │
│     ├─ 1. 查 SQLite：users 画像 + chat_history 历史               │
│     ├─ 2. 无历史 → 欢迎语（并建 user / 落库）                     │
│     ├─ 3. 有历史 → 意图识别（llm/intent.py，1/2/0）               │
│     ├─ 4. 工具调用（tools/functions.py）                          │
│     │      ├─ 意图 1 → web_search + search_market_docs（market）  │
│     │      ├─ 意图 2 → search_course_docs（course）               │
│     │      └─ query_database（SQL Agent，只读）                   │
│     ├─ 5. 组装 System Prompt（画像 + 策略 + 检索证据）            │
│     ├─ 6. DeepSeek-Flash 流式生成（无 Key 则规则模板降级）        │
│     ├─ 7. 写回对话记录，并按回答更新用户画像                      │
│     └─ 8. SSE 推送 error / start / intent / tool_call / delta /   │
│           done 事件                                               │
└──────────────────────────────────────────────────────────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
   SQLite (data/crm.db)     ChromaDB (data/chroma)     DeepSeek API
   users / chat_history     market / course 集合        api.deepseek.com
   courses / career_info    BGE 本地嵌入（离线）          （OpenAI SDK）
```

---

## 3. 目录结构

```
crm项目/
├── app.py                      # Flask 入口：create_app()、路由、错误处理、CORS
├── config.py                   # 配置单例（.env）、目录/日志初始化
├── requirements.txt            # 锁定版本依赖
├── .env.example                # 环境变量示例（复制为 .env）
├── README.md
├── data/                       # 运行时自动生成（SQLite + Chroma 持久化）
│   ├── crm.db
│   └── chroma/
├── database/
│   ├── db.py                   # SQLite 连接/建表/CRUD/只读 SQL 执行
│   ├── schema.sql              # 4 张表 + 索引 + 示例数据
│   ├── sql_agent.py            # SQL Agent（自然语言 → 只读 SQL → 汇总）
│   └── __init__.py
├── vectorstore/
│   ├── embeddings.py           # 本地 BGE 嵌入 + hash 兜底
│   ├── store.py                # ChromaDB 集合管理/增删改查/检索
│   └── __init__.py
├── llm/
│   ├── client.py               # OpenAI SDK 封装：非流式/流式 + 工具循环
│   ├── intent.py               # 意图识别（LLM + 关键词规则）
│   ├── web_search.py           # 联网搜索（DuckDuckGo / SerpAPI + 本地兜底）
│   ├── prompts.py              # 全部提示词模板与消息组装
│   └── __init__.py
├── tools/
│   ├── functions.py            # 6 个 Function Calling 工具 schema + 统一分发
│   └── __init__.py
├── services/
│   ├── chat.py                 # 对话主流程（流式/非流式）
│   ├── rag.py                  # 意图路由检索、证据上下文组装
│   ├── user_info.py            # 用户画像提取与更新
│   └── __init__.py
├── templates/index.html        # 内置聊天页面（SSE 流式渲染）
├── tests/
│   ├── conftest.py             # 测试环境隔离（临时数据目录 / hash 嵌入 / 无 Key）
│   ├── test_chat.py            # 对话主流程
│   ├── test_api.py             # HTTP 接口
│   ├── test_database.py        # SQLite + SQL Agent
│   ├── test_vectorstore.py     # 向量库增删改查
│   ├── test_llm_tools.py       # 意图 / 工具 / 搜索降级
│   ├── test_integration_llm.py # 本地 mock OpenAI 服务 → 真实 function calling 路径
│   └── mock_llm_server.py      # 本地 OpenAI 兼容 mock 服务（无需真实 Key）
└── docs/CONTRACT.md            # 模块接口契约（开发用）
```

---

## 4. 快速开始（从零跑通）

### 4.1 环境要求

- Python **3.10+**（本项目在 Windows + Python 3.14 实测通过）
- 可访问 PyPI；如需下载 BGE 模型，需要可访问 HuggingFace 或镜像站
- 无需 Docker、无需外部数据库：SQLite 与 ChromaDB 均本地文件化

### 4.2 安装依赖

```bash
cd crm项目
python -m pip install -r requirements.txt
```

> 依赖包含 `chromadb`、`sentence-transformers`（会带 torch）等，首次安装体积较大，请耐心等待。

### 4.3 配置环境变量

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

按需编辑 `.env`（所有配置均有默认值，可只填 Key）：

```ini
# DeepSeek（OpenAI 兼容协议）；不填则自动进入「离线规则模式」
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash

# 本地嵌入模型（首次运行自动下载）；国内网络建议启用镜像
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu
HF_ENDPOINT=https://hf-mirror.com

# 网页搜索（默认 Bing，国内网络通常可直连；可选 duckduckgo / serpapi / none）
# 单一来源失败时会自动尝试另一个真实搜索源，再失败才降级本地知识库
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=bing
# 单次联网搜索超时（秒）；网络不可达时首次最多等这么久，之后进入 120 秒熔断直接用本地知识库
WEB_SEARCH_TIMEOUT=3
# SERPAPI_API_KEY=

# Flask
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
```

> **没有 DeepSeek Key 也能跑通全流程**：系统会照常执行意图识别、工具调用（搜索 / RAG / SQL）、画像更新与流式输出，只是最终回答由规则模板生成（日志与 `/health` 会显示 `mode: rule` / 离线状态）。填入 Key 后无需改代码，自动切换为真实大模型。

### 4.4 启动

```bash
python app.py
```

启动日志应包含四项初始化完成信息：**日志系统、SQLite、向量库（ChromaDB + 嵌入）、DeepSeek**；并自动生成 `data/crm.db` 与 `data/chroma/`。

浏览器访问 <http://127.0.0.1:5000/> 即可使用内置聊天页面；健康检查：<http://127.0.0.1:5000/health>。

### 4.5 四类输入体验

| 场景 | 示例问题 | 预期行为 |
| --- | --- | --- |
| 首次咨询 | 任意内容（新手机号） | 返回欢迎语，创建用户，记录 2 条历史 |
| 意图 1（AI 职业咨询） | 「AI 应用开发工程师现在薪资多少？」 | 调用 `web_search` + `search_market_docs`，结合市场资料回答 |
| 意图 2（课程咨询） | 「你们有哪些 Python 课程？多少钱？」 | 调用 `search_course_docs`，结合课程资料回答 |
| 意图 0（其他） | 「你好」「今天天气怎么样」 | 常规回复，不强制检索 |

---

## 5. 接口文档

### 5.1 `POST /chat` — 智能咨询（核心接口）

请求体：

```json
{
  "phone_number": "13900000001",
  "query": "AI 应用开发工程师薪资怎么样？",
  "stream": true
}
```

- `stream` 默认 `true`：返回 `text/event-stream`，每帧 `data: {json}\n\n`，以 `data: [DONE]\n\n` 结束。
- `stream=false`：返回整包 JSON。

SSE 事件类型：

| type | 说明 |
| --- | --- |
| `start` | 会话开始（是否新用户、回答模式） |
| `intent` | 意图识别结果（`1/2/0` + 原因） |
| `tool_call` | 每次工具调用（名称、参数、是否成功） |
| `delta` | 回答增量文本 |
| `done` | 结束（完整回复、意图、模式、工具调用、用户画像、耗时） |
| `error` | 错误信息 |

非流式响应示例：

```json
{
  "ok": true,
  "reply": "……",
  "phone_number": "13900000001",
  "intent_code": 2,
  "intent_name": "课程咨询",
  "mode": "llm",
  "tool_calls": [{"name": "search_course_docs", "arguments": {"query": "Python 课程 价格"}}],
  "user_info": {"phone_number": "13900000001", "stage": "已了解", "...": "..."},
  "is_new": false,
  "rounds": 2,
  "elapsed_ms": 1234
}
```

### 5.2 `GET /health` — 健康检查

返回各组件状态：`llm`（是否配置 Key / 模型）、`embedding`（提供者与维度）、`vectorstore`（各集合文档数）、`database`（各表行数）、`search`（搜索提供者）。

### 5.3 向量库运维接口（硬性需求：对外开放向量库更新接口）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/vectorstore/add` | 新增文档 `{collection, documents, metadatas?, ids?}` |
| POST | `/vectorstore/update` | 更新文档 `{collection, ids, documents, metadatas?}` |
| POST | `/vectorstore/delete` | 删除文档 `{collection, ids}` |
| GET | `/vectorstore/list` | 列出集合与文档数 |
| POST | `/vectorstore/search` | 检索 `{collection, query, top_k?}` |
| POST | `/vectorstore/init` | 重建/补齐示例数据（幂等） |

`collection` 仅允许 `market`（AI 就业市场）与 `course`（课程资料）。

示例：

```bash
curl -X POST http://127.0.0.1:5000/vectorstore/add \
  -H "Content-Type: application/json" \
  -d "{\"collection\":\"course\",\"documents\":[\"AI 大模型应用开发实战班：价格 9999 元，学制 12 周，含 RAG 与 Agent 项目\"]}"

curl -X POST http://127.0.0.1:5000/vectorstore/search \
  -H "Content-Type: application/json" \
  -d "{\"collection\":\"course\",\"query\":\"大模型课程价格\",\"top_k\":3}"
```

新增后直接向 `/chat` 提问「大模型课程多少钱」，即可在回答中看到该资料被检索命中（工具调用 `search_course_docs`）。

### 5.4 SQL Agent 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/database/query` | 自然语言查询 `{question}`，返回 `{ok, answer, sql, rows, mode}`（`mode` 为 `llm` 或 `rule`） |
| GET | `/database/schema` | 返回数据库表结构说明 |

示例：

```bash
curl -X POST http://127.0.0.1:5000/database/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"有哪些课程？\"}"
curl -X POST http://127.0.0.1:5000/database/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"薪资 2 万以上、在北上广深方向的职业有哪些？\"}"
```

仅允许只读 `SELECT` / `WITH` 查询（拦截 `INSERT/UPDATE/DELETE/DROP` 等），杜绝误删数据。

---

## 6. 数据库

`data/crm.db` 自动创建，共 4 张表（含示例数据）：

| 表 | 用途 |
| --- | --- |
| `users` | 用户画像：手机号（主键）、姓名、求职方向、预算、阶段、备注、时间戳 |
| `chat_history` | 对话记录：手机号、角色（`user` / `assistant`）、内容、意图码、时间 |
| `courses` | 课程库：名称、类别、价格、周期、描述 |
| `career_info` | AI 职业市场资料：岗位、行业、薪资区间、描述（也是联网搜索失败时的兜底数据） |

---

## 7. 测试

```bash
python -m pytest -q
```

测试环境由 `tests/conftest.py` 自动隔离（临时数据目录、`hash` 嵌入、无 Key 离线模式），不会污染正式 `data/` 目录，也不需要下载 BGE 模型或联网。

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_chat.py` | 欢迎语与建用户、意图 1/2/0、工具调用、画像提取、SQL Agent 规则、流式事件序列 |
| `tests/test_api.py` | `/health`、`/chat` SSE 与整包、参数校验、向量库 CRUD、SQL Agent 接口、404/500 JSON |
| `tests/test_database.py` | 建表与示例数据、CRUD、历史读写、只读 SQL 拦截、SQL Agent 规则 |
| `tests/test_vectorstore.py` | 集合初始化、add→search 命中、update 生效、delete 生效、幂等 |
| `tests/test_llm_tools.py` | 意图关键词、6 个工具 schema 与分发、`extract_json`、搜索降级 |
| `tests/test_integration_llm.py` | 本地 mock OpenAI 服务 → 真实 `run_tool_calling_loop` / 流式工具循环 / `/chat` 端到端 |
| `tests/ui_smoke.js` | 浏览器级 UI 回归（jsdom，Node.js，可选）：模板转义、手机号校验、SSE 渲染、多轮对话 |

> `tests/test_integration_llm.py` 会在本机启动一个 OpenAI 兼容的 mock 服务来验证真实 function calling 与流式代码路径，**无需真实 API Key**；有 Key 时也可手动改为真实服务地址联调。

### 7.1 浏览器级 UI 回归（可选，需 Node.js）

单元测试与接口测试只覆盖 Flask 侧；为了能发现「前端内联 JS 被模板转义破坏」这类只在浏览器里才暴露的问题（历史 bug：`\d` 被写成 `\\d` 导致合法手机号被拦截、SSE 帧拆不开），项目附带一个 jsdom 冒烟脚本：加载 `GET /` 返回的真实页面、执行内联 JS、模拟点击发送与 Enter。

```bash
cd tests
npm install        # 安装 jsdom（仅开发自测依赖，不属于 Python 运行依赖）
node ui_smoke.js   # 需先 python app.py 启动服务（默认 http://127.0.0.1:5000）
```

脚本共 29 项检查：服务端 HTML 转义、顶部技术栈徽标已移除、工具调用明细独立成块、非法手机号拦截、首轮欢迎语、第二轮课程咨询（意图 2 + 工具调用）、快捷按钮与 Enter 发送、职业咨询（意图 1 + 工具调用）、SSE 增量渲染、健康检查状态灯。真实 LLM 模式下单轮可能需要十几秒，脚本会等待本轮 `done` 完成后再发下一条。

---

## 8. 验收对照

| 需求 | 落地位置 |
| --- | --- |
| 整体系统可正式运行 | `python app.py` 一键启动，日志显示四项初始化完成，页面与 `/health` 可用 |
| ≥1 个 Function Calling 工具 | `tools/functions.py` 定义 6 个工具，主流程与 SQL Agent 均走真实 tool calling |
| 本地向量数据库创建 + 更新 | `vectorstore/store.py`（ChromaDB 持久化）+ `/vectorstore/add|update|delete` |
| 本地 SQLite + SQL Agent | `database/db.py`、`database/sql_agent.py`、`POST /database/query` |
| 对外开放向量库更新接口 | `/vectorstore/add`、`/vectorstore/update`、`/vectorstore/delete`、`/vectorstore/list`、`/vectorstore/search` |
| 对话历史 + 用户画像 | `services/user_info.py` + `users` / `chat_history` 表，画像拼入 System Prompt |
| 流式返回 | `POST /chat` 默认 SSE，逐 `delta` 推送 |
| 浏览器端可正常对话 | `templates/index.html`（前端手机号校验 + SSE 解析）+ `tests/ui_smoke.js` 浏览器级回归（jsdom 29 项检查） |

---

## 9. 常见问题

**Q1：没有 DeepSeek API Key 能验收吗？**
可以。系统会进入「离线规则模式」：意图识别、RAG 检索、SQL Agent、画像更新、SSE 流式全部照常执行，回答由规则模板生成。启动日志与 `/health` 会明确提示当前模式；填入 Key 后自动切换真实模型，无需改代码。

**Q2：首次启动没有网络 / 下载 BGE 模型很慢？**
默认嵌入模型为 `BAAI/bge-small-zh-v1.5`，首次启动自动下载并缓存。国内建议在 `.env` 设置 `HF_ENDPOINT=https://hf-mirror.com`。若模型加载失败，系统自动降级为 hash 嵌入（可用但语义精度下降），不会阻塞启动。

**Q3：想清空数据重新验收？**
停止服务后删除 `data/` 目录（`crm.db` 与 `chroma/`），重新 `python app.py` 即可自动重建表与示例数据。

**Q4：端口被占用？**
在 `.env` 设置 `FLASK_PORT=5001`（或其它端口）后重启。

**Q5：Windows 控制台中文乱码？**
执行 `chcp 65001` 切换 UTF-8 代码页，或使用 Windows Terminal。

**Q6：如何确认工具调用真的发生了？**
两种方式：① `/chat` 的 SSE 中会逐条推送 `tool_call` 事件；② 查看 `data/logs/app.log` 中的 `[tool_call]` 日志行（启动日志、检索、SQL、工具调用均记录在此）。离线模式同样会记录，因为工具调用与回答生成是两个独立环节。

**Q7：点击发送提示「请输入正确的 11 位手机号」，或发送后没有任何回复？**
先强制刷新页面（Ctrl+F5）清除旧模板缓存。该现象源于早期版本模板内联 JS 的反斜杠被二次转义；现已修复，并在 `tests/test_api.py::test_index_html_inline_javascript_not_double_escaped` 与 `tests/ui_smoke.js` 中加了回归检查。

**Q8：有 Key 时职业咨询偶尔要等十几秒？**
真实模型本身需要几秒；另外若某个搜索源在本机不可达，一次联网搜索会等满 `WEB_SEARCH_TIMEOUT`（默认 3 秒）后再换源/降级。系统有 120 秒熔断：全部真实搜索源都失败后，冷却期内直接使用本地 `career_info` 知识库，不会每轮都等待超时。`/health` 的 `search.cooldown_remaining` 可看到剩余冷却时间。

**Q9：怎么确认真实联网搜索生效了？**
`data/logs/app.log` 里会出现 `远程搜索成功：source=bing provider=bing 条数=N` 或 `远程搜索源不可用（duckduckgo）...` 后接 `source=bing`；`/health` 的 `search.provider` 显示当前主用来源。默认 `WEB_SEARCH_PROVIDER=bing`（cn.bing.com），DuckDuckGo 在部分网络不可达时会自动切换到 Bing，无需手工改配置。

---

## 10. 安全与约束

- SQL 仅允许只读查询，所有执行前经过白名单与关键字校验，并限制返回行数（`SQL_MAX_ROWS`）。
- API Key 只从 `.env` / 环境变量读取，日志输出时做掩码处理；`.env` 与 `data/` 已加入 `.gitignore`。
- 用户输入长度受限（`MAX_QUERY_LENGTH`），手机号做格式校验。
- 所有对外接口异常均转换为友好 JSON，不泄露堆栈。
- Web 搜索、向量库、SQLite、LLM 任一依赖异常均被捕获并降级，不会导致服务崩溃。
