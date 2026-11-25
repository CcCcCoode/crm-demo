# -*- coding: utf-8 -*-
"""生成 SeaChat 项目报告 DOCX（第二部分：第 4~8 章 + 附录，导入第一部分后继续写）。"""
import make_report  # noqa: F401  - 先执行第一部分，构建文档与样式
from make_report import (  # noqa: F401
    DARK, GRAY, MID, OUT, ROOT, SHOT, bullets, cap, code, doc, figure, h1, h2, h3,
    numbered, p, table,
)

# ---------------------------------------------------------------------------
# 第 4 章
# ---------------------------------------------------------------------------
h1("4　RAG 设计与优化")

h2("4.1　本地向量库部署形态")
p("向量库选用 ChromaDB 1.5.9 的本地持久化模式（`PersistentClient`），数据落在 `data/chroma/chroma.sqlite3`，"
  "不需要额外进程或容器；备份时直接拷贝目录即可。两个集合都用余弦距离（`metadata={\"hnsw:space\": \"cosine\"}`），"
  "返回的 `distance` 等于 `1 - 余弦相似度`，越小越相关。")
p("实现上有两个关键点。第一，创建集合时显式传入 `embedding_function=None`，写入与查询都由项目自己计算向量再传给 Chroma，"
  "这样可以避免 Chroma 首次使用时自动联网下载默认 ONNX 嵌入模型，保证断网可启动。第二，集合对象做了进程内缓存并加 "
  "`RLock` 保护，`init_vectorstores()` 只在集合为空时灌入示例数据，重复调用不会产生重复文档。")
coll_rows = [
    ("market", "8", "512", "cosine", "AI 岗位的职责、薪资区间、技能要求、行业趋势"),
    ("course", "7", "512", "cosine", "课程大纲、价格、周期、难度、适合人群"),
]
table(["集合", "文档数", "向量维度", "距离度量", "存放内容"], coll_rows, [2.4, 2.0, 2.2, 2.4, 7.6],
      caption="表 4-1　向量库集合", size=8.5)

h2("4.2　嵌入模型与降级策略")
p("默认使用 `BAAI/bge-small-zh-v1.5`（512 维，中文小模型），`encode(normalize_embeddings=True)` 输出单位向量，"
  "因此余弦距离与点积等价、数值稳定。模型懒加载：第一次真正需要嵌入时才初始化，之后常驻内存；"
  "加载前会把 `HF_ENDPOINT` 指向 `hf-mirror.com`，避免国内网络下载失败。实测首次加载 10.2 秒"
  "（另一次冷启动因与压测任务争抢 CPU 用了 39.1 秒），单条 query 嵌入 9.8~10.5 毫秒。")
p("为了在“没有模型文件、没有网络”的环境里仍能跑通测试与演示，项目内置了 `hash` 兜底后端：文本小写去空白后取字符 1~3-gram"
  "（权重 0.5/1.0/1.0）与英文整词（权重 2.0），每个特征用 sha1 散列到 512 个桶并按哈希符号累加，最后 L2 归一化。"
  "它没有语义能力，但完全确定、跨进程一致，适合做单元测试的稳定基线。")
emb_rows = [
    ("bge（默认）", "sentence-transformers + 模型权重", "强，中文语义可用", "首载 10~40 s，单条约 10 ms", "生产与演示"),
    ("hash（兜底）", "仅标准库 hashlib", "弱，仅字面重合", "微秒级", "单元测试、无网/无模型环境"),
    ("openai（可选）", "外部 Embedding API", "强", "取决于网络", "需要多语言或更高质量时"),
]
table(["后端", "依赖", "语义质量", "速度", "适用场景"], emb_rows, [2.8, 4.6, 4.0, 3.4, 3.4],
      caption="表 4-2　三种嵌入后端", size=8.5)
p("语义能力做过对照实验：`相似(\"学 AI 要花多少钱\", \"AI 课程的价格和学费是多少\") = 0.8770`，"
  "而 `相似(\"学 AI 要花多少钱\", \"今天天气很好适合出去玩\") = 0.2576`，区分度明显，说明检索走的是真实语义而不是字面匹配。")

h2("4.3　知识库内容")
p("知识库由 SQLite 种子数据生成，与 `courses`、`career_info` 两张表一一对应（course-001~007 对应 7 门课程，"
  "market-001~008 对应 8 个职业），保证“向量库回答”和“SQL 查询回答”口径一致。每条文档是一段完整的中文描述，"
  "metadata 保留结构化字段，便于前端展示与后续做元数据过滤。")
doc_rows = [
    ("market-001 ~ 008", "8", "岗位名称、所属行业、月薪下限/上限、核心技能、岗位描述、发展路径",
     "title / industry / salary_min / salary_max / skills", "market-003"),
    ("course-001 ~ 007", "7", "课程名称、类别、学费、学习周期、难度、适合人群、课程内容",
     "name / category / price / duration / level / suitable_for", "course-007"),
]
table(["文档 ID", "数量", "正文内容", "metadata 字段", "示例"], doc_rows, [3.0, 1.4, 6.4, 4.4, 2.2],
      caption="表 4-3　知识库文档结构", size=7.8)
p("以 course-007 为例，入库文本是：")
code("""《Python 自动化办公训练营》是一门办公自动化课程，学费 3980 元，学习周期 8 周。课程讲解 Python 基础语法、
Excel 批量处理、文件与邮件自动化、报表自动生成等实用技能，适合零基础的职场白领、行政财务人员，学完可用
Python 替代大量重复性办公操作。""", caption="代码 4-1　一条课程文档的正文")
p("检索时返回的 metadata 里还带有 `price`、`duration`、`suitable_for` 等字段，模型可以据此直接报价、比较周期，"
  "不需要再从 SQL 里查一遍。通过 `/vectorstore/add` 写入的自定义文档可以携带任意标量 metadata（数字、字符串、布尔），"
  "列表与空值会被清洗掉，避免 Chroma 写入失败。")

h2("4.4　在线检索链路")
p("检索不是“每次都用向量库”，而是先按意图路由：意图 1（职业）同时调 `web_search` 与 `search_market_docs`，"
  "意图 2（课程）只调 `search_course_docs`，意图 0（其他）不检索。这本身就是一种低成本的 RAG 优化——"
  "减少无关召回，也减少 token 消耗。核心逻辑在 `services/rag.py:105-135`：")
code("""try:
    if int(intent_code) == INTENT_CAREER:
        web_payload = _run_tool("web_search",
                                {"query": query, "top_k": config.WEB_SEARCH_TOP_K}, context, tool_calls, errors)
        if isinstance(web_payload, dict):
            web_results = list(web_payload.get("results") or [])
        market_payload = _run_tool("search_market_docs",
                                   {"query": query, "top_k": config.RAG_TOP_K}, context, tool_calls, errors)
        if isinstance(market_payload, dict):
            market_docs = list(market_payload.get("results") or [])
    elif int(intent_code) == INTENT_COURSE:
        course_payload = _run_tool("search_course_docs",
                                   {"query": query, "top_k": config.RAG_TOP_K}, context, tool_calls, errors)
        if isinstance(course_payload, dict):
            course_docs = list(course_payload.get("results") or [])
except Exception as exc:      # 检索失败降级为空资料，不阻断主流程
    errors.append(f"retrieve: {exc}")""", caption="代码 4-2　按意图调度检索工具（节选）")
p("检索结果经 `build_evidence_block` 转成带编号、标题、来源的文本块，再拼进 system prompt，并在提示词里明确要求"
  "“优先依据资料回答、引用来源、资料不足时说明”。每次工具调用都会写一行 `[tool_call]` 日志，"
  "验收时可以直接在日志里确认“检索真的发生了”。图 4 是课程咨询的完整链路截图：")
figure("03-课程咨询意图2.png", "图 4　课程咨询（意图 2）：向量检索 + SQL 核对，4 次工具调用全部成功", width=14.6)

h2("4.5　RAG 优化方式调研")
p("RAG 的优化没有“一招通吃”，比较实用的做法是按失败环节归因：答案所在的块没被召回，问题在切分与查询表达；"
  "块召回了但排在后面，问题在排序；块排得靠前仍答错，问题在上下文拼装与提示词。基于这个框架，"
  "我把业界常见手段整理如下，并逐项评估与本项目的适配度。")
opt_rows = [
    ("分块策略", "递归 512 token、20~30% 重叠；语义分块", "公开基准中递归切分准确率约 69%，语义分块约 54%", "低",
     "当前文档本身就是一条完整业务描述，按实体整条入库即可；将来接 PDF 再用"),
    ("上下文前缀（Contextual Retrieval）", "给每个块补一段文档级上下文再嵌入", "检索失败率相对下降约 35%，叠加 BM25 达约 49%", "低~中",
     "**优先做**：把课程名、类别、价格拼到正文首部，几乎零成本"),
    ("混合检索", "BM25 稀疏检索 + 向量检索，RRF 融合", "对编号、专有名词、数字等精确匹配显著更稳", "中",
     "课程名、价格、薪资数字场景收益明显，需引入分词与 BM25 索引"),
    ("重排序（Rerank）", "cross-encoder 对召回结果精排，top-20 → top-5", "与上下文检索叠加后失败率降幅可达约 67%", "中（CPU 20~40 ms/条）",
     "候选只有 3 条时收益有限；把召回扩到 10 条再上重排更划算"),
    ("查询改写", "LLM 改写口语问题、多查询、HyDE", "提升口语化、指代型问题的召回", "中",
     "可与意图识别共用一次模型调用，摊薄延迟与成本"),
    ("元数据过滤", "先按类别/价格/难度过滤，再向量检索", "缩小候选集、降低噪声", "低",
     "metadata 已齐全，加一个 where 条件即可"),
    ("上下文压缩与排序", "去重、截断、按相关性排序，规避“中间遗忘”", "提升长上下文下的答案忠实度", "低",
     "当前只放 3 条证据，收益有限，可作为后续优化"),
    ("评估闭环", "构建 golden set，度量 recall@k、MRR、忠实度", "让每次优化可量化、可回归", "低",
     "建议先做 20~30 条问题—期望文档对，纳入 pytest"),
    ("Agentic RAG / 路由", "由模型决定检索几次、查哪个库、是否再查", "复杂多跳问题提升明显", "高",
     "当前的意图路由是其轻量版本，可演进为多轮检索 Agent"),
    ("缓存", "query 向量 LRU、常见问题答案缓存", "直接降本提速，客服场景命中率高", "低",
     "**性价比最高**：同一问题在售前场景会反复出现"),
]
table(["优化方向", "做法", "业界效果 / 结论", "成本", "对本项目的适配判断"], opt_rows, [3.2, 4.6, 4.6, 2.0, 6.2],
      caption="表 4-4　RAG 常见优化方式与本项目适配度", size=7.6)
p("参考资料：Anthropic《Contextual Retrieval》（上下文检索与 BM25 组合的失败率数据）、"
  "阿里云开发者社区《检索质量优化——查询改写、混合检索与重排》（三层归因法）、"
  "腾讯云开发者社区《RAG 原理：分块、检索、重排的全链路拆解》、"
  "GitHub `youngyangyang04/llm-master` 的 RAG 优化笔记，以及公开的 chunking 基准结果。", indent=False, size=8.5, color=GRAY)

h2("4.6　与本项目的融合路线")
p("结合上面的评估，我把可落地的优化分成三档。P0 是本次已经实现的；P1 是不改架构、几天内能做完且收益明确的；"
  "P2 需要引入新依赖或改造索引结构。")
road_rows = [
    ("P0 已完成", "意图路由（0/1/2 分流）、top_k=3、证据块带来源、hash 降级、增量更新接口、检索失败不阻断",
     "当前形态", "—"),
    ("P1 近期（3~5 人日）", "文档首部拼接上下文前缀、BM25+向量混合（RRF）、metadata 过滤（类别/价格区间）、"
     "查询改写、query 向量缓存、20~30 条 golden set 写入 pytest",
     "预期 recall@3 提升 10~20%，重复问题延迟下降", "低，全部为纯 Python 实现"),
    ("P2 中期（2~3 周）", "bge-reranker-base 重排（召回扩到 10）、父子块检索、多路召回、答案缓存、"
     "索引版本化与灰度切换、评估看板",
     "长尾与多跳问题提升明显，可量化回归", "中，需引入 reranker 与额外存储"),
]
table(["阶段", "措施", "预期收益", "成本"], road_rows, [3.0, 7.6, 4.2, 4.0],
      caption="表 4-5　RAG 优化融合路线", size=7.8)

h2("4.7　数据更新与一致性")
p("向量库更新通过四个接口完成，语义如下：`add` 未指定 id 时自动生成 `{collection}-{uuid}`，返回写入的 id 与集合总数；"
  "`update` 先用 `get(ids)` 检查存在性，不存在的 id 会被忽略并在响应里体现，存在的重新计算向量后覆盖；"
  "`delete` 幂等，删除不存在的 id 不报错；`init` 只在集合为空时灌入示例，重复调用安全。")
p("一致性方面，单进程内用 `RLock` 串行化写操作；多进程部署时 Chroma 的持久化客户端不保证跨进程写安全，"
  "需要改为 Chroma Server 模式或加外部锁。建议后续补充三项能力：按内容哈希去重（避免重复灌入）、"
  "批量 upsert（减少往返）、软删除与更新审计（保留可回溯的版本记录）。")

# ---------------------------------------------------------------------------
# 第 5 章
# ---------------------------------------------------------------------------
h1("5　接口文档")

h2("5.1　通用约定")
bullets([
    "**Base URL**：`http://127.0.0.1:5000`（`FLASK_HOST` / `FLASK_PORT` 可覆盖）。",
    "**编码**：请求与响应均为 UTF-8；普通接口 `Content-Type: application/json`，`/chat` 流式返回 `text/event-stream`。",
    "**统一结构**：成功为 `{\"ok\": true, ...业务字段}`，失败为 `{\"ok\": false, \"error\": \"中文错误说明\"}`。",
    "**跨域**：响应头带 `Access-Control-Allow-Origin: *`，允许前端独立部署。",
    "**请求体上限**：4 MB（Flask `MAX_CONTENT_LENGTH`），超出返回 413。",
    "**时间格式**：`YYYY-MM-DD HH:MM:SS`（数据库）与 ISO 8601（`/health` 的 timestamp）。",
])

h2("5.2　接口总览")
api_rows = [
    ("GET", "/", "返回聊天页面（templates/index.html）", "—"),
    ("GET", "/health", "健康检查：模型、嵌入、向量库、数据库、搜索状态", "—"),
    ("POST", "/chat", "核心对话接口，支持 SSE 流式与非流式", "phone_number、query、stream"),
    ("POST", "/vectorstore/add", "向指定集合新增文档", "collection、documents、metadatas、ids"),
    ("POST", "/vectorstore/update", "按 id 更新文档（重新计算向量）", "collection、ids、documents、metadatas"),
    ("POST", "/vectorstore/delete", "按 id 删除文档", "collection、ids"),
    ("GET", "/vectorstore/list", "列出集合与文档数", "—"),
    ("POST", "/vectorstore/search", "语义检索，返回相似度距离", "collection、query、top_k"),
    ("POST", "/vectorstore/init", "幂等初始化示例知识库", "—"),
    ("POST", "/database/query", "SQL Agent：自然语言转只读 SQL 并解释结果", "question"),
    ("GET", "/database/schema", "返回 4 张表的 schema 与示例数据", "—"),
]
table(["方法", "路径", "功能", "主要参数"], api_rows, [1.8, 4.6, 7.4, 5.0], caption="表 5-1　接口总览（11 个）", size=8.2)

h2("5.3　POST /chat")
p("核心接口。请求体三个字段：")
param_rows = [
    ("phone_number", "string", "是", "手机号，正则 `^1[3-9]\\d{9}$`，允许中间带空格或短横线（服务端会清洗）"),
    ("query", "string", "是", "用户问题，去空白后 1~2000 字"),
    ("stream", "bool", "否", "默认 true；显式传 false 时返回一次性 JSON"),
]
table(["字段", "类型", "必填", "说明"], param_rows, [3.0, 2.0, 1.6, 11.0], caption="表 5-2　/chat 请求字段", size=8.5)
p("流式响应由若干 SSE 帧组成，每帧格式为 `data: {JSON}\\n\\n`，以 `data: [DONE]` 结束。帧类型如下：")
event_rows = [
    ("start", "phone_number、is_new、intent_code、intent_name、mode", "会话开始；老用户在第 4 步之后发出，新用户直接以 welcome 模式返回"),
    ("intent", "intent_code、intent_name、reason、mode", "意图识别结果与理由；mode 为意图识别方式（llm / rule）"),
    ("tool_call", "name、arguments、ok、error", "一次工具调用；前端折叠展示，便于验收"),
    ("delta", "content", "回答增量文本，前端追加到气泡"),
    ("done", "reply、intent_code、intent_name、mode、tool_calls、user_info、rounds、elapsed_ms", "本轮结束；mode 为回答生成方式（llm / rule / fallback / welcome）"),
    ("error", "message", "异常提示，随后仍会发 [DONE]"),
]
table(["帧类型", "主要字段", "含义"], event_rows, [2.2, 7.6, 7.0], caption="表 5-3　SSE 帧类型", size=7.8)
p("下面是真实抓取的帧（课程咨询，已省略 arguments 细节）：")
code("""data: {"type": "start", "phone_number": "13700002222", "is_new": false, "mode": "rule",
       "intent_code": 2, "intent_name": "课程咨询"}

data: {"type": "intent", "intent_code": 2, "intent_name": "课程咨询",
       "reason": "命中课程类关键词「课程、多少钱」，判断为课程咨询", "mode": "rule"}

data: {"type": "tool_call", "name": "search_course_docs",
       "arguments": {"query": "AI 应用开发课程多少钱？", "top_k": 3}, "ok": true, "error": null}

data: {"type": "delta", "content": "## AI 应用开发工程师（大模型 RAG）· 费用\\n\\n**学费：15,800 元…"}

data: {"type": "done", "reply": "…", "phone_number": "13700002222", "is_new": false,
       "intent_code": 2, "intent_name": "课程咨询", "mode": "llm",
       "tool_calls": [{"name": "search_course_docs", "ok": true, ...}], "rounds": 2, "elapsed_ms": 6880}

data: [DONE]""", caption="代码 5-1　真实 SSE 帧（截断）")
p("需要注意：`start` 与 `intent` 帧里的 `mode` 表示**意图识别**方式（rule 表示走了关键词兜底），"
  "`done` 帧里的 `mode` 表示**回答生成**方式（llm 表示由大模型生成），两者可能不同——上例即“规则识别意图 + 大模型回答”。")
p("非流式请求返回聚合后的 JSON，字段与 done 帧一致：")
code("""{
  "ok": true,
  "reply": "## Python 数据分析与可视化（课程编号 course-002）…",
  "phone_number": "13700002222",
  "is_new": false,
  "intent_code": 2,
  "intent_name": "课程咨询",
  "mode": "llm",
  "rounds": 2,
  "elapsed_ms": 6880,
  "tool_calls": [
    {"name": "search_course_docs", "arguments": {"query": "…", "top_k": 3}, "ok": true, "error": null}
  ],
  "user_info": {"phone_number": "13700002222", "name": null, "intention_direction": "Python 数据分析",
                "budget": null, "stage": "已了解", "notes": "…"}
}""", caption="代码 5-2　非流式响应（真实返回，reply 截断）")
p("新用户首轮不会走意图识别：直接返回欢迎语，`is_new=true`、`mode=welcome`、`intent_code=0`，"
  "并把欢迎语写入历史，保证第二轮开始有上下文。")

h2("5.4　向量库接口")
p("六个接口围绕两个集合工作，`collection` 只接受 `market` 与 `course`，其他值返回 400。所有写操作都会写日志，"
  "并在响应里带上写入后的集合总数（`stats`），方便前端确认。")
h3("5.4.1　POST /vectorstore/add")
code("""// 请求
{"collection": "course",
 "documents": ["Python 自动化办公训练营：3980 元，8 周，适合职场白领用脚本替代重复劳动。"],
 "metadatas": [{"name": "Python 自动化办公训练营", "price": 3980, "source": "report-demo"}],
 "ids": ["report-demo-001"]}

// 响应（HTTP 200）
{"ok": true, "collection": "course", "count": 1, "ids": ["report-demo-001"], "stats": 8}""")
h3("5.4.2　POST /vectorstore/update")
code("""// 请求（ids 与 documents 必须等长）
{"collection": "course", "ids": ["report-demo-001"], "documents": ["更新后的课程描述…"]}

// 响应：updated 为实际更新条数，missing 为不存在的 id
{"ok": true, "collection": "course", "ids": ["report-demo-001"], "updated": 1}""")
h3("5.4.3　POST /vectorstore/delete")
code("""{"ok": true, "collection": "course", "deleted": ["report-demo-001"], "stats": 7}""")
h3("5.4.4　GET /vectorstore/list")
code("""{"ok": true, "count": 2,
 "collections": [{"name": "course", "count": 7}, {"name": "market", "count": 8}]}""")
h3("5.4.5　POST /vectorstore/search")
code("""// 请求
{"collection": "course", "query": "办公自动化 多少钱", "top_k": 2}

// 响应：distance 为余弦距离（0 表示完全一致，越小越相关）
{"ok": true, "collection": "course", "count": 2, "query": "办公自动化 多少钱",
 "results": [
   {"id": "report-demo-001", "distance": 0.3167, "document": "Python 自动化办公训练营：3980 元…",
    "metadata": {"name": "Python 自动化办公训练营", "price": 3980, "source": "report-demo"}},
   {"id": "course-007", "distance": 0.364,
    "document": "《Python 自动化办公训练营》是一门办公自动化课程，学费 3980 元…",
    "metadata": {"category": "办公自动化", "course_id": 7, "price": 3980, "duration": "8 周"}}
 ]}""")
h3("5.4.6　POST /vectorstore/init")
code("""{"ok": true, "collections": [{"name": "course", "count": 7}, {"name": "market", "count": 8}]}""")

h2("5.5　数据库接口")
p("`POST /database/query` 接收自然语言问题，内部先尝试大模型生成 SQL（只读校验通过才执行），失败则用规则映射；"
  "返回 SQL 语句、结果行与中文解释。`GET /database/schema` 返回 4 张表的结构说明与每表 2 行示例，"
  "既供调试，也作为 SQL Agent 的提示词素材。")
code("""// POST /database/query
{"question": "薪资最高的三个职业是什么"}

// 响应（HTTP 200，rows 只展示前 3 行）
{"ok": true, "mode": "llm",
 "sql": ["SELECT title, industry, salary_min, salary_max, skills FROM career_info
          ORDER BY salary_max DESC LIMIT 3"],
 "rows": [{"title": "大模型算法工程师", "industry": "人工智能", "salary_min": 25000,
           "salary_max": 50000, "skills": "深度学习、Transformer、PyTorch、微调、推理优化"}, …],
 "answer": "按**月薪上限**从高到低，薪资最高的三个职业是：…（Markdown 表格）…",
 "error": null}""", caption="代码 5-3　数据库查询接口真实返回（截断）")
p("非法或无法映射的问题返回 400，例如 `DROP TABLE users` 得到 `{\"ok\": false, \"error\": \"无法将问题映射为数据库查询\"}`；"
  "模型生成的非法 SQL 会被 `is_readonly_sql` 拦截，错误回灌给模型后可以重试。")

h2("5.6　GET /health")
p("用于部署自检与前端状态灯，真实返回如下（字段有删减）：")
code("""{"ok": true, "service": "SeaChatInterview AI CRM", "status": "ok",
 "timestamp": "2026-11-18T13:00:39",
 "llm": {"available": true, "base_url": "https://api.deepseek.com", "model": "deepseek-flash"},
 "embedding": {"configured_provider": "bge", "provider": "bge", "model": "BAAI/bge-small-zh-v1.5",
               "dim": 512, "backend": "sentence-transformers", "ready": true, "detail": "BGE 句向量模型已加载"},
 "vectorstore": [{"name": "course", "count": 7}, {"name": "market", "count": 8}],
 "database": {"users": 21, "chat_history": 139, "courses": 7, "career_info": 8},
 "search": {"enabled": true, "provider": "bing", "calls": 4, "fallback_used": false,
            "last_error": null, "cooldown_remaining": 0.0}}""", caption="代码 5-4　健康检查真实返回")

h2("5.7　错误码与规范差异")
err_rows = [
    ("200", "成功", "业务结果在响应体里，`ok=true`"),
    ("400", "参数错误", "缺 phone_number/query、手机号格式错、集合名非法、SQL 无法映射、ids 与 documents 不等长"),
    ("404 / 405", "路由不存在 / 方法不允许", "统一返回 JSON，不返回 HTML 错误页"),
    ("413", "请求体过大", "超过 4 MB"),
    ("500", "服务内部错误", "兜底异常处理，返回 `{\"ok\": false, \"error\": \"服务器内部错误：…\"}`"),
]
table(["状态码", "含义", "典型场景"], err_rows, [2.4, 4.6, 9.6], caption="表 5-4　错误码", size=8.5)
p("与作业文字要求存在三处差异，均不影响功能，但验收时可能被问到：①作业示例的向量库接口统一返回 "
  "`{\"code\": 0, \"message\": \"ok\", \"data\": ...}`，本项目用的是 `{\"ok\": true, ...}`，语义等价但字段名不同；"
  "②伪代码里 `/chat` 的 `content_type` 写的是 `application/json`，本项目流式分支返回 `text/event-stream`"
  "（按行输出 JSON 分块，SSE 是更标准、浏览器可直接消费的实现）；③伪代码要求名为 `SeaChatInterview` 的函数，"
  "本项目实际入口是 `stream_chat` / `process_chat`，见 8.3 的兼容建议。")

# ---------------------------------------------------------------------------
# 第 6 章
# ---------------------------------------------------------------------------
h1("6　并发能力预估")

h2("6.1　测试环境与方法")
p("压测在项目同机进行：AMD Ryzen 7 4800H（8 核 16 线程）、23.4 GB 内存、Windows 11、Python 3.14.3，"
  "服务以 `python app.py` 启动（Flask 内置服务器，`threaded=True`），客户端为 `报告/bench.py`"
  "（urllib + threading，固定总请求数，统计墙钟、QPS 与 p50/p95/max）。")
p("测试分两组：A 组为“规则模式”——无 API Key、hash 嵌入、关闭联网，测框架与检索本身的开销；"
  "B 组为“大模型模式”——真实 Key、关闭联网（嵌入仍用 hash 以隔离变量），测端到端时延。另用 `报告/bench_embed.py` 单独测量嵌入与检索耗时。"
  "受本机其他任务争抢 CPU 影响，p95 有约 ±15% 的波动。")

h2("6.2　实测结果")
h3("6.2.1　规则模式（无 Key）")
rule_rows = [
    ("1", "4", "0.13", "31.7", "32.1", "34.5", "40.3"),
    ("2", "4", "0.08", "51.5", "20.3", "20.8", "55.5"),
    ("4", "8", "0.15", "52.4", "47.6", "57.8", "129.6"),
    ("8", "16", "0.39", "41.6", "125.8", "274.8", "303.6"),
    ("16", "32", "0.80", "40.3", "131.2", "581.6", "769.5"),
]
table(["并发", "请求数", "墙钟(s)", "QPS", "p50(ms)", "p95(ms)", "max(ms)"], rule_rows,
      [2.0, 2.2, 2.6, 2.4, 2.6, 2.6, 2.6], caption="表 6-1　/chat 规则模式压测", size=8.5)
p("吞吐在 4 并发时达到峰值约 52 QPS，此后基本稳定在 40 QPS 左右，p50 从 20 ms 升到 131 ms——"
  "瓶颈是 GIL 与嵌入/检索的串行化，属于“单进程 CPU 受限”的典型形态。")
h3("6.2.2　健康检查")
health_rows = [
    ("4", "16", "0.08", "206.8", "11.4", "20.7", "43.4"),
    ("16", "64", "0.21", "301.0", "51.3", "62.0", "71.8"),
]
table(["并发", "请求数", "墙钟(s)", "QPS", "p50(ms)", "p95(ms)", "max(ms)"], health_rows,
      [2.0, 2.2, 2.6, 2.4, 2.6, 2.6, 2.6], caption="表 6-2　/health 压测", size=8.5)
h3("6.2.3　大模型模式（真实 Key）")
llm_rows = [
    ("1", "4", "29.97", "0.13", "7322", "8086", "8997"),
    ("2", "4", "16.37", "0.24", "7860", "8637", "9609"),
    ("4", "8", "15.76", "0.51", "7354", "7686", "8195"),
    ("8", "16", "15.63", "1.02", "7384", "8121", "8181"),
    ("16", "32", "15.76", "2.03", "7191", "8348", "8401"),
]
table(["并发", "请求数", "墙钟(s)", "QPS", "p50(ms)", "p95(ms)", "max(ms)"], llm_rows,
      [2.0, 2.2, 2.6, 2.4, 2.6, 2.6, 2.6], caption="表 6-3　/chat 大模型模式压测", size=8.5)
p("关键现象：**每个请求的时延几乎恒定为 7.2~7.9 秒，与并发数无关；吞吐随并发线性增长**。"
  "这说明请求在等上游生成（I/O 等待，线程会释放 GIL），单机并没有被算力压满，"
  "QPS 的上限取决于能开多少并发线程以及上游的配额。")
h3("6.2.4　单请求耗时分解")
seg_rows = [
    ("BGE 模型首次加载", "10.2 s（另一次冷启动 39.1 s）", "一次性，之后常驻内存"),
    ("单条 query 嵌入", "9.8 ~ 10.5 ms", "512 维，CPU 推理"),
    ("向量检索（含嵌入）", "15 ~ 25 ms（首次含集合打开 140 ms）", "15 条文档规模，耗时主要在嵌入"),
    ("规则回答生成", "< 10 ms", "模板拼接，可忽略"),
    ("大模型回答（全链路）", "6.9 ~ 15.6 s", "2~9 次工具调用，上游生成时间主导"),
]
table(["阶段", "实测耗时", "说明"], seg_rows, [4.6, 5.6, 6.4], caption="表 6-4　单请求耗时分解", size=8.5)

h2("6.3　瓶颈分析")
bullets([
    "**上游大模型是绝对瓶颈**：全链路 7 秒里，本地环节合计不到 100 ms，其余都是网络与生成时间。"
    "因此提升吞吐最有效的手段是提高上游并发配额，而不是加 CPU。",
    "**CPU 嵌入是第二瓶颈**：BGE 单条约 10 ms，torch 自身会开多线程，与请求线程争抢；"
    "并发升高后 p95 明显抬升（表 6-1 的 8/16 并发）。",
    "**SQLite 单写者**：每轮对话写 2 行 + 可能更新画像，写锁持有时间极短（毫秒级），"
    "`busy_timeout=10s` 足以吸收尖峰，但高并发写入时仍建议换 WAL 模式或独立写队列。",
    "**Chroma 单进程客户端**：读并发安全，写操作由进程内锁串行；多进程部署需要改用 Server 模式。",
    "**Flask 内置服务器非生产级**：没有进程管理、慢连接防护与超时控制，正式部署应换 gunicorn/uWSGI + Nginx。",
])

h2("6.4　容量模型与部署建议")
p("用 Little 定律估算：并发数 = 目标 QPS × 平均时延。大模型模式平均时延约 7.5 秒，"
  "则 16 并发对应 2.1 QPS、32 并发对应约 4.2 QPS，与实测（2.03 QPS）吻合。规则模式时延约 0.03 秒，"
  "理论上 32 并发可到 1000 QPS，但受 GIL 与嵌入串行限制，实测上限约 40~52 QPS，应按 40 QPS 保守估算。")
cap_rows = [
    ("单进程（Flask threaded，本次实测）", "16", "2.0", "40", "LLM ≈ 1,000 / 规则 ≈ 8,000"),
    ("gunicorn 4 worker × 8 线程（推算）", "32", "4.2", "80 ~ 120", "LLM ≈ 2,000 / 规则 ≈ 20,000"),
    ("8 进程 × 8 线程 + Chroma Server + Redis 缓存（推算）", "64", "8.5（受上游配额约束）", "200+", "LLM ≈ 4,000"),
]
table(["部署形态", "并发", "LLM 模式 QPS", "规则模式 QPS", "估算峰值日活"], cap_rows, [6.2, 1.8, 3.2, 3.2, 2.8],
      caption="表 6-5　容量估算（日活按每用户 10 次/天、峰值集中在 4 小时、峰值系数 3 计算）", size=8.2)
p("测算口径：4 小时 = 14,400 秒；4.2 QPS × 14,400 s ≈ 60,480 次，除以峰值系数 3 ≈ 20,160 次，"
  "再除以每人每天 10 次 ≈ 2,000 日活。若只看规则模式，40 QPS 对应约 19,000 日活。"
  "实际业务量还受上游 TPM/RPM 配额与 token 成本约束，上线前应按真实套餐核对。")
p("部署建议：生产环境用 `gunicorn -w 4 -k gthread --threads 8`（32 并发）+ Nginx 反向代理，"
  "为 `/chat` 设置 60 秒读超时；前置 Redis 做 query 向量与常见问题答案缓存；"
  "对同一手机号做简单限流（如每分钟 6 次），避免单用户占满线程；"
  "监控项至少包括上游 429/5xx 比例、p95 时延、工具失败率与向量库文档数。")

h2("6.5　结论")
p("单机单进程在真实大模型模式下可稳定支撑约 2 QPS、16 个并发对话；规则/降级模式可支撑约 40 QPS。"
  "系统的时延由上游生成主导，因此扩容优先级是“上游配额 > 进程/线程数 > CPU”，"
  "而成本优化的优先级是“缓存 > 少调用工具 > 换更小的模型”。对作业场景（演示与课程验收），"
  "单进程已完全够用；对生产场景，按表 6-5 的 4 worker 配置即可支撑约 2,000 日活的售前咨询。")

# ---------------------------------------------------------------------------
# 第 7 章
# ---------------------------------------------------------------------------
h1("7　测试与质量")

h2("7.1　自动化测试")
p("测试用 pytest 组织，`tests/conftest.py` 在导入项目模块前把数据目录指向临时目录、清空 API Key、"
  "切换 hash 嵌入并关闭联网，因此全部用例可离线、可重复运行，且不会污染 `data/`。")
test_rows = [
    ("tests/test_database.py", "49", "建库与种子、用户/课程/职业 CRUD、预算解析、只读 SQL 拦截、schema 描述"),
    ("tests/test_vectorstore.py", "24", "新增/更新/删除/检索、非法集合、元数据清洗、幂等初始化"),
    ("tests/test_llm_tools.py", "20", "6 个工具 Schema 合法性、参数归一化、执行与异常契约"),
    ("tests/test_chat.py", "15", "主流程、首轮欢迎语、无 Key 降级、画像保存、输入校验"),
    ("tests/test_api.py", "11", "路由、SSE 帧、错误码、模板 JS 转义回归"),
    ("tests/test_integration_llm.py", "4", "mock LLM 服务的全链路集成"),
    ("合计", "123", "pytest 5.4~5.6 秒跑完，1 条 chromadb 遥测的 DeprecationWarning"),
]
table(["测试文件", "用例数", "覆盖点"], test_rows, [6.0, 2.2, 8.4], caption="表 7-1　pytest 用例分布", size=8.5)
p("执行结果：`python -m pytest -q` → `123 passed, 1 warning in 5.43s`。唯一告警来自 chromadb 内部使用"
  "`asyncio.iscoroutinefunction`，与本项目代码无关。")

h2("7.2　前端 UI 冒烟")
p("`tests/ui_smoke.js` 用 jsdom 加载真实的 `templates/index.html`，mock `fetch` 与 SSE 数据流，"
  "断言页面结构、消息渲染、意图/画像/工具标签、错误提示与转义安全，共 30 项。"
  "本轮为截图新增了 Markdown 渲染（`formatRichText`），并补了一条“答案中不再出现 `**` 星号”的回归断言，"
  "防止富文本改造回退。执行方式与结果：")
code("""cd tests
npm install            # 仅首次，安装 jsdom
$env:CRM_BASE_URL="http://127.0.0.1:5099"
node ui_smoke.js       # 30 passed, 0 failed""", caption="代码 7-1　UI 冒烟执行方式")

h2("7.3　端到端验收记录")
p("下表是隔离数据目录（hash 嵌入、无 Key）下通过真实 Flask 路由完成的验收，覆盖五条需求的主路径与异常路径。")
e2e_rows = [
    ("健康检查", "GET /health", "200；course 7 / market 8；courses 7 / career_info 8；llm.available=false", "通过"),
    ("新用户首轮", "phone=13800138000，任意问题", "is_new=true，mode=welcome，intent_code=0，返回欢迎语", "通过"),
    ("职业咨询", "“AI 职业规划怎么做”", "intent=1，mode=rule，工具 web_search + search_market_docs", "通过"),
    ("课程咨询", "“Python 课程多少钱”", "intent=2，工具 search_course_docs", "通过"),
    ("其他咨询", "“今天天气怎么样”", "intent=0，不触发检索", "通过"),
    ("SSE 流式", "stream 缺省", "200，text/event-stream；start/delta/done/[DONE] 共 4 帧", "通过"),
    ("向量库新增", "POST /vectorstore/add", "200，返回 id，集合 7→8", "通过"),
    ("向量库更新", "POST /vectorstore/update", "200，updated=1，重新嵌入生效", "通过"),
    ("向量库检索", "POST /vectorstore/search", "新文档 distance 0.6599 排第一（course-001 为 0.9814）", "通过"),
    ("向量库删除", "POST /vectorstore/delete", "200，集合回到 7", "通过"),
    ("课程 SQL", "“有哪些课程”", "ok=true，sql=SELECT … FROM courses ORDER BY price ASC LIMIT 100，rows=7", "通过"),
    ("高薪 SQL", "“薪资 2 万以上的职业有哪些”", "rows=6（按 salary_max 过滤）", "通过"),
    ("SQL 注入", "“DROP TABLE users”", "400，{\"ok\": false, \"error\": \"无法将问题映射为数据库查询\"}", "通过"),
    ("参数校验", "空 query / 坏手机号 / 非法集合", "均 400，返回中文错误说明", "通过"),
]
table(["场景", "输入", "实际结果", "判定"], e2e_rows, [2.8, 4.6, 7.4, 1.6], caption="表 7-2　端到端验收记录", size=7.8)

h2("7.4　真实环境验证")
p("除离线用例，另在真实 Key 与真实 BGE 下做了三轮验证：")
real_rows = [
    ("语义检索质量", "相关句相似度 0.8770，无关句 0.2576", "BGE 生效，不是字面匹配"),
    ("大模型主流程", "职业咨询 13.5~15.6 s、9 次工具调用；课程咨询 6.9~8.7 s、2~4 次工具调用", "回答含薪资/价格表格，引用来源正确"),
    ("SQL Agent", "生成并执行 ORDER BY price / salary_max 的 SELECT，rows=7/6/3", "结果正确，回答带 Markdown 表格"),
    ("联网搜索", "Bing 0.8 s 返回 3 条结果，来源 ai-bot.cn / ibm.com / developers.google.cn", "搜索链路可用，未触发熔断"),
]
table(["验证项", "实测结果", "结论"], real_rows, [3.2, 8.6, 4.8], caption="表 7-3　真实环境验证", size=8.2)

# ---------------------------------------------------------------------------
# 第 8 章
# ---------------------------------------------------------------------------
h1("8　已知问题与改进建议")
p("以下是审计与真实运行中发现的问题，按影响排序。P1 影响“意图识别是否真的用了大模型”，"
  "P2 影响从零部署，其余为健壮性或规范性问题。")
issue_rows = [
    ("P1", "意图识别使用强制 tool_choice，与 deepseek-flash 的 thinking 模式冲突，接口返回 400："
     "`Thinking mode does not support this tool_choice`", "意图识别永远走关键词兜底，功能不中断但“LLM 识别意图”名存实亡", "8.1"),
    ("P2", "仓库缺少 .env.example，README 第 4 节却让用户 `copy .env.example .env`", "从零部署第一步断链", "8.2"),
    ("P3", "未提供 `SeaChatInterview` 同名函数；响应包为 {ok,...}，与作业建议的 {code,message,data} 不同", "严格按文字验收可能失分", "8.3"),
    ("P4", "大模型写 SQL 时偶发臆造列名（如 target_audience，实际为 suitable_for）", "首轮 SQL 失败，靠下一轮自我修正，多耗 1~2 秒", "8.4"),
    ("P5", "`simple_chat` 在 max_tokens 很小时返回空串（thinking 模式先消耗推理 token）", "辅助调用拿不到结果，需要重试或提高预算", "8.5"),
    ("P6", "前端此前把 Markdown 原样显示（`**`、`#` 直接出现在气泡里）", "答案可读性差（本轮已修复并加回归断言）", "8.6"),
]
table(["编号", "问题", "影响", "详见"], issue_rows, [1.2, 7.6, 6.0, 1.4], caption="表 8-1　问题清单", size=7.8)

h2("8.1　P1：意图识别的 tool_choice 400")
p("`llm/intent.py` 为了让模型必定返回结构化结果，使用了强制工具选择："
  "`tool_choice={\"type\": \"function\", \"function\": {\"name\": \"detect_intent\"}}`。"
  "实测 deepseek-flash 在 thinking 模式下直接拒绝该参数，抛 `Error code: 400 - Thinking mode does not support this tool_choice`，"
  "于是每次意图识别都走异常分支、回退到关键词规则。而不带 tool_choice 时，模型会主动调用 `detect_intent` 且参数正确。"
  "建议改为 `auto` 并兼容“模型没调工具”的情况：")
code("""# llm/intent.py：把强制 tool_choice 改为 auto
response = chat_completion(
    messages, tools=[DETECT_INTENT_TOOL_SCHEMA],
    tool_choice="auto", temperature=0.0, max_tokens=200,
)
calls = response.get("tool_calls") or []
if calls:
    data = extract_json(calls[0]["arguments"]) or {}
    code = _coerce_intent_code(data.get("intent_code"))
    if code is not None:
        return {"intent_code": code, "reason": data.get("reason", ""), "mode": "llm"}
# 没有工具调用或参数非法 → 继续走关键词兜底
return {"intent_code": keyword_intent(query), "reason": "关键词规则兜底", "mode": "rule"}""",
     caption="代码 8-1　P1 的修复方案")

h2("8.2　P2：补充 .env.example")
p("建议新增 `.env.example`，列出全部键名与安全默认值（不含真实密钥），并把 README 的快速开始改为"
  "`copy .env.example .env`。需要覆盖的键包括：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、"
  "`LLM_TEMPERATURE`、`LLM_MAX_TOKENS`、`LLM_MAX_TOOL_ROUNDS`、`LLM_OFFLINE_FALLBACK`、`EMBEDDING_PROVIDER`、"
  "`EMBEDDING_MODEL`、`EMBEDDING_DIM`、`EMBEDDING_DEVICE`、`HF_ENDPOINT`、`RAG_TOP_K`、`WEB_SEARCH_ENABLED`、"
  "`WEB_SEARCH_PROVIDER`、`SERPAPI_API_KEY`、`SQL_READONLY`、`SQL_MAX_ROWS`、`CHAT_HISTORY_LIMIT`、"
  "`MAX_QUERY_LENGTH`、`FLASK_HOST`、`FLASK_PORT`、`LOG_LEVEL`。")

h2("8.3　P3：命名与响应格式兼容")
p("两处可以低成本兼容。其一，增加一个与伪代码同名的薄封装，内部复用现有流程；"
  "其二，在向量库接口上增加一个可选开关，返回 `{code, message, data}` 形式的包体。这样既不破坏现有前端，"
  "也能通过按文字逐条对照的验收。")
code("""# services/chat.py：与作业伪代码同名的入口
def SeaChatInterview(phone_number: str, query: str) -> dict:
    \"\"\"作业伪代码要求的入口名，内部复用 process_chat。\"\"\"
    return process_chat(phone_number, query)""", caption="代码 8-2　兼容入口")

h2("8.4　P4：SQL 列名校验")
p("在把 SQL 交给数据库之前，先用 `PRAGMA table_info` 或项目内已有的表结构信息校验列名，"
  "不合法时不执行，而是把“可用列名”作为工具结果回灌给模型，让它一次改对，避免“执行失败→报错→重试”的往返。"
  "同时在 SQL Agent 的系统提示词里明确列出每张表的列名与含义。")

h2("8.5　P5：短预算调用")
p("`simple_chat` 用于画像抽取等辅助场景，max_tokens 设为 20~120 时实测返回空串，因为 thinking 模式的推理 token"
  "先占满预算。建议为这类调用设置最小 300 token 的预算，或在返回空串时自动放大预算重试一次，"
  "并在日志里记录 `finish_reason`，便于定位。")

h2("8.6　P6：前端 Markdown（已修复）")
p("本轮把气泡渲染从“纯文本”升级为“安全富文本”：`escapeHtml` 先整体转义，再按行解析标题、列表、"
  "行内加粗与行内代码，流式过程中仍显示纯文本，`done` 时一次性替换为格式化结果，避免每帧重排。"
  "同时补了一条 UI 回归断言。改造后 `pytest` 仍为 123 通过，`ui_smoke.js` 由 29 项增加到 30 项且全部通过。")

# ---------------------------------------------------------------------------
# 附录
# ---------------------------------------------------------------------------
h1("附录 A　代码结构与打包")
p("项目按职责分包，根目录只保留入口与配置。完整目录树如下（省略 `data/`、`node_modules/`、缓存目录）：")
code("""crm项目/
├─ app.py                 # Flask 入口：路由、SSE、错误处理、CORS
├─ config.py              # 配置单例与环境变量校验
├─ requirements.txt       # 依赖清单（9 个直接依赖）
├─ .env                   # 本地密钥与参数（不应提交仓库）
├─ database/              # db.py / schema.sql / sql_agent.py
├─ vectorstore/           # store.py（Chroma）/ embeddings.py（BGE、hash）
├─ llm/                   # client.py / intent.py / prompts.py / web_search.py
├─ services/              # chat.py（主流程）/ rag.py / user_info.py
├─ tools/                 # functions.py：6 个工具 Schema 与执行
├─ templates/             # index.html：单文件聊天前端
├─ tests/                 # 123 项 pytest + 30 项 jsdom UI 冒烟
├─ docs/CONTRACT.md       # 冻结的接口契约
├─ 报告/                  # 本报告、截图、压测与截图脚本
└─ data/                  # 运行时生成：crm.db、chroma/、logs/""", caption="代码 A-1　目录结构")
p("交付压缩包 `SeaChat-完整代码.zip` 的打包口径：包含源码、测试、文档、前端与报告脚本；"
  "排除 `.env`（含真实密钥）、`data/`、`node_modules/`、`__pycache__/`、`.pytest_cache/`。"
  "解压后按 1.5 节步骤即可运行，`python -m pytest -q` 应得到 123 passed。")

h1("附录 B　运行截图清单")
shot_rows = [
    ("01-首页.png", "首页与在线状态", "图 1（1.5 节）", "1440×900，右上角“模型在线 · deepseek-flash”"),
    ("02-新用户欢迎语.png", "新用户首轮欢迎语", "图 2（2.2 节）", "mode=welcome、intent_code=0，画像为“新咨询”"),
    ("03-课程咨询意图2.png", "课程咨询（意图 2）", "图 4（4.4 节）", "4 次工具调用、答案带价格表"),
    ("04-职业咨询意图1.png", "职业咨询（意图 1）", "图 3（3.6 节）", "9 次工具调用、答案带薪资表"),
]
table(["文件", "场景", "正文位置", "说明"], shot_rows, [4.6, 4.6, 4.0, 5.0], caption="表 B-1　截图清单", size=8.2)
p("截图由 `报告/shot.js` 通过 Chrome DevTools Protocol 驱动真实 Chrome 无头模式抓取："
  "启动 Chrome → 连接调试端口 → 打开页面 → 输入问题并等待流式结束 → 展开工具折叠 → 截图。"
  "因此图中内容全部是真实运行结果，不是设计稿。")

h1("附录 C　运行日志摘录")
p("以下为 `data/logs/app.log` 的真实片段（略有删减），可作为“程序可正式运行”的佐证。")
code("""2026-11-15 15:18:04 | INFO | crm.app        | SeaChatInterview AI CRM 正在启动……
2026-11-15 15:18:04 | INFO | crm.app        | 数据目录：C:\\Users\\IKUN\\Desktop\\crm项目\\data
2026-11-15 15:18:04 | INFO | crm.database   | 数据库初始化完成，当前数据量：{'users': 0, 'chat_history': 0, 'courses': 7, 'career_info': 8}
2026-11-15 15:18:04 | INFO | crm.vectorstore| ChromaDB 持久化客户端已就绪：path=…\\data\\chroma
2026-11-15 15:18:15 | INFO | crm.vectorstore| 正在懒加载 BGE 嵌入模型：BAAI/bge-small-zh-v1.5（device=cpu）……
2026-11-15 15:21:53 | INFO | crm.services.rag| [tool_call] web_search arguments={"query": "AI 应用开发工程师薪资前景怎么样？", "top_k": 5} ok=True
2026-11-18 12:53:00 | INFO | crm.services.user_info | LLM 抽取到画像增量：{'intention_direction': 'AI 应用开发', 'stage': '已了解', …}
2026-11-18 12:53:00 | INFO | crm.services.chat| 本轮结束：phone=13700002222，intent=1，mode=llm，工具调用=9 次，耗时=15560ms
2026-11-18 12:53:05 | INFO | crm.database   | 执行只读 SQL 成功，返回 2 行：SELECT * FROM users LIMIT 2
2026-11-18 13:00:39 | INFO | werkzeug       | 127.0.0.1 - - "GET /health HTTP/1.1" 200 -""", caption="代码 C-1　启动与运行日志")
code("""$ python -m pytest -q
123 passed, 1 warning in 5.43s

$ node tests/ui_smoke.js
30 passed, 0 failed""", caption="代码 C-2　测试结果")

p("报告完。", indent=False, align=1, size=10, color=GRAY)
doc.save(OUT)
print("saved:", OUT)
