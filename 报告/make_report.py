# -*- coding: utf-8 -*-
"""生成 SeaChat 项目报告 DOCX（第一部分：工具函数 + 封面 + 第 1~3 章）。"""
import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = r"C:\Users\IKUN\Desktop\crm项目"
SHOT = os.path.join(ROOT, "报告", "screenshots")
OUT = os.path.join(ROOT, "报告", "SeaChat项目报告.docx")

DARK = RGBColor(0x1F, 0x38, 0x64)
MID = RGBColor(0x2E, 0x54, 0x96)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0x7F, 0x7F, 0x7F)

doc = Document()
doc.core_properties.title = "SeaChat AI 客服咨询系统 项目报告"
doc.core_properties.author = "项目组"
doc.core_properties.subject = "意图路由 / Function Calling / 本地 RAG / SQL Agent"

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = Cm(2.3)
sec.bottom_margin = Cm(2.1)
sec.different_first_page_header_footer = True

# ---------------------------------------------------------------------------
# 样式
# ---------------------------------------------------------------------------
normal = doc.styles["Normal"]
normal.font.name = "宋体"
normal.font.size = Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.paragraph_format.line_spacing = 1.4
normal.paragraph_format.space_after = Pt(3)

for name, size, color, before, after in (
    ("Heading 1", 17, DARK, 6, 10),
    ("Heading 2", 13.5, MID, 12, 5),
    ("Heading 3", 11.5, RGBColor(0x33, 0x33, 0x33), 8, 3),
):
    st = doc.styles[name]
    st.font.name = "微软雅黑"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = color
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.keep_with_next = True

doc.styles["Heading 1"].paragraph_format.page_break_before = False
doc.styles["Heading 1"].paragraph_format.space_before = Pt(26)


def rich(par, text, size=None, color=None, bold=False):
    """写入支持 **加粗** 与 `等宽` 的富文本。"""
    for piece in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            run = par.add_run(piece[2:-2])
            run.bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            run = par.add_run(piece[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt((size or 10.5) - 1)
            run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "等线")
        else:
            run = par.add_run(piece)
            run.bold = bold
        if size:
            run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color
    return par


def p(text, indent=True, size=10.5, color=None, align=None, after=3, before=0, bold=False):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.first_line_indent = Pt(21) if indent else Pt(0)
    pf.space_after = Pt(after)
    pf.space_before = Pt(before)
    if align is not None:
        par.alignment = align
    rich(par, text, size=size, color=color, bold=bold)
    return par


def h1(text):
    return doc.add_heading(text, level=1)


def h2(text):
    return doc.add_heading(text, level=2)


def h3(text):
    return doc.add_heading(text, level=3)


def bullets(items):
    for item in items:
        par = doc.add_paragraph()
        pf = par.paragraph_format
        pf.left_indent = Cm(0.75)
        pf.first_line_indent = Cm(-0.35)
        pf.space_after = Pt(2)
        rich(par, "· " + item)


def numbered(items):
    for index, item in enumerate(items, 1):
        par = doc.add_paragraph()
        pf = par.paragraph_format
        pf.left_indent = Cm(0.85)
        pf.first_line_indent = Cm(-0.6)
        pf.space_after = Pt(2)
        rich(par, f"{index}. " + item)


def shade_paragraph(par, fill):
    pPr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def code(text, size=8.5, caption=None):
    lines = text.rstrip("\n").split("\n")
    total = len(lines)
    for index, line in enumerate(lines):
        par = doc.add_paragraph()
        pf = par.paragraph_format
        pf.space_before = Pt(6) if index == 0 else Pt(0)
        pf.space_after = Pt(8) if index == total - 1 else Pt(0)
        pf.line_spacing = 1.0
        pf.left_indent = Cm(0.3)
        pf.first_line_indent = Pt(0)
        if total <= 14 and index < total - 1:
            pf.keep_with_next = True
        run = par.add_run(line if line else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(size)
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "等线")
        shade_paragraph(par, "F4F5F7")
    if caption:
        cap(caption)


def cap(text):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(2)
    par.paragraph_format.space_after = Pt(8)
    run = par.add_run(text)
    run.font.size = Pt(9)
    run.font.color.rgb = GRAY


def set_table_style(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "BFBFBF")
        borders.append(el)
    tblPr.append(borders)

    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", 18), ("left", 80), ("bottom", 18), ("right", 80)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def cell_text(cell, text, size=8.5, bold=False, color=None, align=None):
    cell.text = ""
    par = cell.paragraphs[0]
    par.paragraph_format.space_after = Pt(0)
    par.paragraph_format.space_before = Pt(0)
    par.paragraph_format.line_spacing = 1.15
    if align is not None:
        par.alignment = align
    rich(par, str(text), size=size, color=color, bold=bold)


_TBLPR_ORDER = [
    "tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
    "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd",
    "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
]


def _fix_table_layout(tbl, total_twips):
    """把 tblPr 规范化为固定布局：唯一的 tblW / tblLayout，并按 schema 顺序排列。"""
    tblPr = tbl._tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout"):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(total_twips))
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(tblW)
    tblPr.append(layout)
    children = list(tblPr)

    def order_key(el):
        tag = el.tag.split("}")[-1]
        return _TBLPR_ORDER.index(tag) if tag in _TBLPR_ORDER else 99

    children.sort(key=order_key)
    for el in children:
        tblPr.remove(el)
    for el in children:
        tblPr.append(el)


def table(headers, rows, widths, caption=None, size=8.5):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    set_table_style(tbl)

    hdr = tbl.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    trPr.append(repeat)
    for index, head in enumerate(headers):
        cell = hdr.cells[index]
        cell_text(cell, head, size=size, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "DCE6F1")
        tcPr.append(shd)

    for row in rows:
        cells = tbl.add_row().cells
        for index, value in enumerate(row):
            cell_text(cells[index], value, size=size)

    for index, column in enumerate(tbl.columns):
        column.width = Cm(widths[index])
    for row in tbl.rows:
        for index, width in enumerate(widths):
            row.cells[index].width = Cm(width)
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        trPr.append(cant)
    _fix_table_layout(tbl, int(sum(widths) * 567))

    if caption:
        cap(caption)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(name, caption, width=15.6):
    path = os.path.join(SHOT, name)
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(6)
    par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.keep_with_next = True
    par.add_run().add_picture(path, width=Cm(width))
    cap(caption)


def add_field(par, instr, size=9):
    run = par.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr_el, sep, text, end):
        run._r.append(el)
    run.font.size = Pt(size)
    run.font.color.rgb = LIGHT


footer = sec.footer
fp = footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run("SeaChat 项目报告　·　第 ")
run.font.size = Pt(9)
run.font.color.rgb = LIGHT
add_field(fp, "PAGE")
run = fp.add_run(" 页 / 共 ")
run.font.size = Pt(9)
run.font.color.rgb = LIGHT
add_field(fp, "NUMPAGES")
run = fp.add_run(" 页")
run.font.size = Pt(9)
run.font.color.rgb = LIGHT

# ---------------------------------------------------------------------------
# 封面
# ---------------------------------------------------------------------------
for _ in range(4):
    doc.add_paragraph()
par = doc.add_paragraph()
par.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = par.add_run("SeaChat AI 客服咨询系统")
run.font.name = "微软雅黑"
run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
run.font.size = Pt(30)
run.bold = True
run.font.color.rgb = DARK

par = doc.add_paragraph()
par.alignment = WD_ALIGN_PARAGRAPH.CENTER
par.paragraph_format.space_before = Pt(10)
run = par.add_run("项 目 实 现 报 告")
run.font.name = "微软雅黑"
run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
run.font.size = Pt(20)
run.bold = True
run.font.color.rgb = MID

par = doc.add_paragraph()
par.alignment = WD_ALIGN_PARAGRAPH.CENTER
par.paragraph_format.space_before = Pt(16)
run = par.add_run("意图路由　·　Function Calling　·　本地 RAG　·　SQL Agent")
run.font.size = Pt(11.5)
run.font.color.rgb = GRAY

for _ in range(3):
    doc.add_paragraph()

cover_rows = [
    ("技术栈", "Python 3.14 / Flask / DeepSeek-Flash / ChromaDB / BGE / SQLite"),
    ("代码规模", "业务源码约 8,600 行（另有测试约 2,200 行）"),
    ("知识库", "本地向量库 2 个集合、15 条文档；SQLite 4 张业务表"),
    ("质量验证", "pytest 123 项 + 前端 UI 冒烟 30 项，全部通过"),
    ("运行环境", "Windows 11 · AMD Ryzen 7 4800H (8C/16T) · 23.4 GB"),
    ("提交日期", "2026 年 11 月 18 日"),
]
tbl = doc.add_table(rows=0, cols=2)
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
for name, value in cover_rows:
    cells = tbl.add_row().cells
    cell_text(cells[0], name, size=10, bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT)
    cell_text(cells[1], value, size=10)
    cells[0].width = Cm(3.2)
    cells[1].width = Cm(11.5)

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
par = doc.add_paragraph()
par.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = par.add_run("目　　录")
run.font.name = "微软雅黑"
run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
run.font.size = Pt(18)
run.bold = True
run.font.color.rgb = DARK
par.paragraph_format.space_after = Pt(14)

toc_rows = [
    ("1", "项目概述", "背景目标、需求对照、交付物、运行环境、启动自检"),
    ("2", "系统架构", "分层结构、请求时序、文件职责、关键设计决策"),
    ("3", "Function Calling 工具定义", "五步闭环、Schema 组织、六个工具、循环与流式合并、实测链路"),
    ("4", "RAG 设计与优化", "向量库部署、嵌入模型、知识库内容、检索链路、优化调研与融合路线"),
    ("5", "接口文档", "通用约定、接口总览、/chat 事件流、向量库与数据库接口、错误码"),
    ("6", "并发能力预估", "压测方法、实测数据、瓶颈分析、容量模型、部署建议"),
    ("7", "测试与质量", "123 项单测、30 项 UI 冒烟、端到端验收、真实模型验证"),
    ("8", "已知问题与改进", "P1~P6 问题清单、影响评估与修复建议"),
    ("A", "附录 A 代码结构", "目录树、打包口径与交付说明"),
    ("B", "附录 B 运行截图", "新用户欢迎、课程咨询、职业咨询"),
    ("C", "附录 C 运行日志", "启动、工具调用、测试通过记录"),
]
table(["章节", "标题", "主要内容"], toc_rows, [1.6, 5.4, 9.6], size=9)

p("说明：本报告中的所有数字、截图与接口返回均来自本机真实运行结果；"
  "涉及大模型与联网搜索的部分，已在真实 API Key 下完成端到端验证。", indent=False, size=9, color=GRAY)

# ---------------------------------------------------------------------------
# 第 1 章
# ---------------------------------------------------------------------------
h1("1　项目概述").paragraph_format.page_break_before = True

h2("1.1　背景与目标")
p("SeaChat 是面向 AI 职业教育机构的售前咨询系统。用户打开网页，输入手机号与问题，系统完成四件事："
  "识别意图（职业咨询 / 课程咨询 / 其他）、按意图检索素材（联网搜索、本地向量库、本地数据库）、"
  "结合用户画像与会话历史调用 DeepSeek-Flash 生成回答、把本轮对话与画像增量写回数据库。")
p("作业给出的伪代码函数 `SeaChatInterview(phone_number, query)` 规定了主干：连接数据库 → 读取该手机号的历史 → "
  "无历史则初始化并返回欢迎语 → 识别意图 → 按意图取向量库回答或走常规处理 → 更新历史 → 返回响应；"
  "Flask 侧用 `Response(stream_response(data), content_type=...)` 逐块输出。本项目的实现与该主干逐句对应，"
  "并在其之上补齐了用户画像、Function Calling、SQL Agent、向量库管理接口与离线降级能力。")

h2("1.2　需求与实现对照")
p("下表把作业的五条项目要求逐条映射到具体实现与验证证据。结论：五条要求全部达成，"
  "存在三处“字面不一致”（函数命名、响应包装格式、流式 Content-Type），详见 5.7 与 8.3。")
req_rows = [
    ("1", "基于伪代码完成整体流程，程序可正式运行",
     "services/chat.py:stream_chat（入参校验→取历史→首轮欢迎→意图→检索→生成→落库）；app.py 提供 /chat",
     "123 项单测 + 端到端冒烟 + 真实 Key 跑通", "达成"),
    ("2", "至少涉及一个 function calling 工具定义",
     "tools/functions.py 定义 6 个工具 Schema；llm/client.py:run_tool_calling_loop 实现完整循环",
     "真实一轮回答触发 9 次工具调用（图 3、附录 C）", "达成"),
    ("3", "本地部署向量库，完成创建与更新",
     "vectorstore/store.py（Chroma 持久化于 data/chroma）+ embeddings.py（BGE 512 维，hash 兜底）",
     "/vectorstore/add|update|delete 实测 200；相似度 0.877 vs 0.258", "达成"),
    ("4", "本地 SQLite 数据库 + SQL Agent",
     "database/schema.sql（4 表）+ db.py（只读校验）+ sql_agent.py（NL2SQL，规则兜底）",
     "/database/query 生成并执行 SELECT，返回 7 行/6 行/3 行", "达成"),
    ("5", "对外开放向量库更新接口，封装整体系统",
     "app.py 暴露 6 个 /vectorstore/* REST 接口 + CORS + 统一 JSON + 4 MB 体积限制",
     "HTTP 200 实测（add→search→update→delete 全链路）", "达成"),
]
table(["#", "原始要求", "实现位置", "验证证据", "结论"], req_rows, [0.8, 4.0, 5.4, 4.2, 1.4],
      caption="表 1-1　五条项目要求的实现与验证对照", size=8)

h2("1.3　交付物清单")
deliver_rows = [
    ("完整代码", "源码 + 测试 + 文档 + 前端页面（打包为 SeaChat-完整代码.zip）", "项目根目录"),
    ("运行截图", "首页、新用户欢迎语、课程咨询（意图 2）、职业咨询（意图 1）", "报告/screenshots"),
    ("项目报告", "本文件，含架构、Function Calling、RAG、接口与并发预估", "报告/SeaChat项目报告.docx"),
    ("测试与压测脚本", "pytest 用例、jsdom UI 冒烟、bench.py、bench_embed.py", "tests/、报告/"),
    ("数据与日志", "SQLite 库、Chroma 持久化目录、运行日志", "data/"),
]
table(["交付项", "内容", "位置"], deliver_rows, [3.0, 10.4, 3.2], caption="表 1-2　交付物清单", size=8.5)

h2("1.4　运行环境")
env_rows = [
    ("操作系统", "Windows 11 专业工作站版"),
    ("处理器 / 内存", "AMD Ryzen 7 4800H，8 核 16 线程 / 23.4 GB"),
    ("Python", "3.14.3"),
    ("Web 框架", "Flask 3.1.3 + Werkzeug 3.1.3"),
    ("大模型", "deepseek-flash（OpenAI 兼容接口 https://api.deepseek.com）"),
    ("向量数据库", "ChromaDB 1.5.9（本地 PersistentClient，HNSW + cosine）"),
    ("嵌入模型", "BAAI/bge-small-zh-v1.5（512 维，CPU 推理，HF 镜像 hf-mirror.com）"),
    ("关系数据库", "SQLite 3（Python 内置 sqlite3，WAL 未启用，busy_timeout=10s）"),
    ("关键依赖", "openai 3.24.0 / sentence-transformers 6.3.0 / numpy 2.4.4 / python-dotenv 1.2.4 / pytest 9.2.0"),
]
table(["项目", "配置"], env_rows, [3.6, 13.0], caption="表 1-3　运行环境", size=8.5)

h2("1.5　启动与自检")
p("项目根目录执行以下命令即可启动。首次启动会自动建库、灌入示例知识库并加载 BGE 模型（约 10~40 秒），"
  "随后访问 http://127.0.0.1:5000 即可使用。")
code("""python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

# 配置密钥：复制并填写 DEEPSEEK_API_KEY（当前仓库缺少 .env.example，见 8.2）
# 也可以不填：系统自动进入离线规则模式，功能不中断

python app.py              # 默认 0.0.0.0:5000，可用 FLASK_PORT 覆盖
python -m pytest -q        # 123 passed
""")
p("图 1 为启动后的首页：右上角实时显示“模型在线 · deepseek-flash”，中间是引导语，"
  "底部为快捷问题与输入框。该状态来自 /health 接口，模型、嵌入、向量库、数据库任一不可用时都会在页面上明确降级显示。",
  indent=True)
figure("01-首页.png", "图 1　系统首页（真实运行，模型在线）", width=15.2)
p("页面顶部的状态灯不是写死的：页面加载时会请求一次 /health，模型可用时显示“模型在线 · deepseek-flash”，"
  "否则显示“未配置 API Key · 离线规则模式”。下表是本次验收时的真实返回值，可直接作为“系统自检通过”的证据：")
check_rows = [
    ("大模型", "llm.available / llm.model", "true / deepseek-flash", "在线"),
    ("嵌入模型", "embedding.provider / embedding.dim", "bge / 512", "就绪"),
    ("向量库", "vectorstore[].count", "course 7、market 8", "正常"),
    ("数据库", "database", "users 21、chat_history 139、courses 7、career_info 8", "正常"),
    ("联网搜索", "search.enabled / search.provider", "true / bing", "可用"),
]
table(["检查项", "字段", "实测值", "判定"], check_rows, [3.0, 5.6, 6.4, 2.2],
      caption="表 1-4　启动自检（GET /health 真实返回）", size=8.5)
p("若某项显示为降级（例如“离线模式”“hash 嵌入”“搜索不可用”），系统仍可完整对话，只是能力受限；"
  "常见原因是未配置 DEEPSEEK_API_KEY、模型下载超时或端口被占用。")

# ---------------------------------------------------------------------------
# 第 2 章
# ---------------------------------------------------------------------------
h1("2　系统架构")

h2("2.1　分层结构")
p("系统按“接口层—编排层—能力层—数据层”四层组织，配置与日志横切。每层只依赖下一层的公开函数，"
  "工具调用是能力层与编排层之间唯一的横向通道，因此每个工具既可以由流程主动调用（规则模式），"
  "也可以由大模型自主调用（Function Calling 模式）。")
layer_rows = [
    ("接口层", "app.py", "11 个路由、SSE 流式封装、CORS、统一错误处理、请求体大小限制"),
    ("编排层", "services/chat.py、rag.py、user_info.py",
     "主流程状态机、按意图检索、证据拼装、画像抽取与更新"),
    ("能力层", "tools/functions.py、llm/*、database/sql_agent.py",
     "工具注册与执行、模型客户端与工具循环、意图识别、提示词、联网搜索、NL2SQL"),
    ("数据层", "database/db.py、vectorstore/store.py、embeddings.py",
     "SQLite 读写与只读 SQL 校验、Chroma 集合管理、BGE/hash 嵌入"),
    ("横切", "config.py", "环境变量、目录、日志、参数校验与脱敏"),
]
table(["层次", "主要文件", "职责"], layer_rows, [2.2, 6.6, 7.8], caption="表 2-1　分层结构", size=8.5)

h2("2.2　一次咨询的完整时序")
p("以“老用户咨询 AI 岗位薪资”为例，一次请求依次经过 9 个步骤，与伪代码主干一一对应：")
numbered([
    "**校验与建用户**：`normalize_phone` 校验手机号（`^1[3-9]\\d{9}$`），`get_or_create_user` 确保 users 表有记录。",
    "**取历史**：`fetch_chat_history` 按 `CHAT_HISTORY_LIMIT=20` 取最近若干轮，正序返回。",
    "**首轮分支**：无历史时直接返回欢迎语（`mode=welcome`、`intent_code=0`），写入本轮记录后结束——对应伪代码的 `initialize_chat_history()`。",
    "**意图识别**：`detect_intent_with_reason` 先请求大模型（强制调用 detect_intent），失败或非法时回退关键词规则，得到 1/2/0 与理由。",
    "**检索素材**：`retrieve_evidence` 按意图调用工具——意图 1 调 `web_search` + `search_market_docs`，意图 2 调 `search_course_docs`，意图 0 不检索。",
    "**组装上下文**：`build_evidence_block` 把资料转成带编号与来源的文本块；`build_system_prompt` 再拼入用户画像与阶段策略。",
    "**生成回答**：有 Key 时进入 `run_streaming_tool_loop`（模型可继续调用 6 个工具）；无 Key 或上游异常时用模板降级。",
    "**落库与更新画像**：`save_conversation_turn` 写 user + assistant 两行；`update_user_profile` 用 LLM（失败回退规则）抽取预算、方向、阶段、备注。",
    "**返回**：流式逐帧推送 start/intent/tool_call/delta/done，最后 `data: [DONE]`；非流式则聚合为一次 JSON。",
])
p("其中第 3 步“首轮分支”决定了新用户的第一印象：不调用任何工具、不做意图识别，直接返回欢迎语并引导补充画像。"
  "图 2 是真实界面，气泡下方的标签依次显示“意图 0 · 其他/常规咨询”“欢迎语”“画像：新咨询”。")
figure("02-新用户欢迎语.png", "图 2　新用户首轮：直接返回欢迎语（mode=welcome，未调用任何工具）", width=14.6)

h2("2.3　目录结构与文件职责")
p("下表给出全部源码文件的职责。行数为真实统计（含空行与注释），业务源码合计约 8,600 行，测试约 2,200 行。")
files = [
    ("app.py", "415", "Flask 应用工厂：注册 11 个路由、SSE 封装、CORS、404/405/500 统一处理"),
    ("config.py", "288", "Settings 单例：路径、模型、嵌入、检索、搜索、SQL、日志参数与 validate()"),
    ("crm.md", "75", "作业伪代码与需求原文（参考）"),
    ("开发提示词.md", "402", "作业出题文档（参考）"),
    ("README.md", "379", "项目说明、接口文档、验收对照、常见问题"),
    ("database/db.py", "963", "唯一 SQLite 访问层：建库、CRUD、历史、画像、只读 SQL 安全校验"),
    ("database/schema.sql", "122", "4 张表 DDL + 6 个索引 + 课程/职业种子数据"),
    ("database/sql_agent.py", "916", "SQL Agent：NL2SQL、工具 Schema、结果解释、规则兜底"),
    ("database/__init__.py", "65", "导出 db 与 sql_agent 的公开接口"),
    ("vectorstore/store.py", "1034", "Chroma 封装：集合管理、增删改查、示例灌入、幂等初始化"),
    ("vectorstore/embeddings.py", "536", "三种嵌入后端：BGE（默认）/ hash（离线兜底）/ openai"),
    ("vectorstore/__init__.py", "61", "导出向量库与嵌入接口"),
    ("llm/client.py", "768", "OpenAI 客户端、同步/流式工具循环、JSON 容错解析、状态查询"),
    ("llm/intent.py", "310", "意图识别：工具 Schema、LLM 调用、关键词权重兜底"),
    ("llm/prompts.py", "508", "系统提示词、阶段策略、证据块、欢迎语、SQL/画像提示词、离线模板"),
    ("llm/web_search.py", "800", "联网搜索：Bing / DuckDuckGo / SerpAPI，失败降级本地库，120s 熔断"),
    ("llm/__init__.py", "76", "导出客户端、意图、提示词与搜索接口"),
    ("services/chat.py", "432", "主流程生成器 stream_chat 与非流式聚合 process_chat"),
    ("services/rag.py", "177", "按意图调度检索工具、记录 [tool_call] 日志、拼装证据块"),
    ("services/user_info.py", "345", "手机号校验、画像规则抽取、LLM 抽取、字段白名单更新"),
    ("services/__init__.py", "14", "包导出"),
    ("tools/functions.py", "477", "6 个工具 Schema、参数归一化、执行分发、永不抛异常"),
    ("tools/__init__.py", "35", "导出 get_openai_tools / execute_tool"),
    ("templates/index.html", "393", "单文件聊天前端：SSE 解析、Markdown 渲染、意图/画像/工具标签"),
    ("tests/conftest.py", "67", "测试隔离：临时数据目录、无 Key、hash 嵌入、关闭联网"),
    ("tests/test_api.py", "191", "11 项：路由、SSE、错误码、模板转义回归"),
    ("tests/test_chat.py", "185", "15 项：主流程、欢迎语、降级、画像保存"),
    ("tests/test_database.py", "434", "49 项：建库、CRUD、预算解析、只读 SQL 拦截"),
    ("tests/test_vectorstore.py", "458", "24 项：增删改查、非法集合、元数据清洗、幂等初始化"),
    ("tests/test_llm_tools.py", "373", "20 项：6 个工具 Schema 与执行、异常契约"),
    ("tests/test_integration_llm.py", "117", "4 项：mock LLM 全链路"),
    ("tests/mock_llm_server.py", "255", "本地 mock 的 OpenAI 兼容服务，用于离线集成测试"),
    ("tests/ui_smoke.js", "156", "jsdom 前端冒烟：30 项断言（含 Markdown 渲染回归）"),
    ("报告/bench.py", "64", "HTTP 并发压测（/chat、/health）"),
    ("报告/bench_embed.py", "33", "BGE 嵌入与向量检索耗时测量"),
    ("报告/shot.js", "166", "CDP 驱动 Chrome 截取真实运行截图"),
]
table(["文件", "行数", "职责"], files, [5.4, 1.5, 9.7], caption="表 2-2　源码文件职责清单", size=7.8)

h2("2.4　关键设计决策")
bullets([
    "**选本地 Chroma 而非云向量库**：作业要求本地部署，知识库只有 15 条文档；Chroma 用单文件 SQLite 持久化，零额外服务、可离线、拷贝 data/chroma 即可迁移。",
    "**禁用它自带的嵌入函数**：`embedding_function=None` 避免首次使用时联网下载 ONNX 模型，保证断网也能起服务，向量质量交给 BGE 控制。",
    "**BGE + hash 双后端**：BGE 语义可用（相关句 0.877、无关句 0.258）但首载慢；hash 纯标准库、跨进程稳定，保证 123 项单测在任何机器可复现。",
    "**全链路可降级**：意图、检索、回答、SQL 四处都有规则兜底；没有 Key 也能完成“识别→检索→模板回答→落库”，并避免上游 400/超时打断对话。",
    "**SQLite 只读校验前置**：把“只能 SELECT”做成数据库层硬约束（前缀、字符串外分号、关键词黑名单、LIMIT 100），不依赖提示词约束模型。",
])

# ---------------------------------------------------------------------------
# 第 3 章
# ---------------------------------------------------------------------------
h1("3　Function Calling 工具定义")

h2("3.1　基本流程")
p("Function Calling 让大模型在“说话”之外还能“做事”：开发者用 JSON Schema 描述一批函数，模型在需要时返回"
  "结构化的调用请求，由本地代码执行，再把结果回灌给模型继续推理。一次完整的闭环包含五步：")
numbered([
    "**定义**：为每个工具声明 `name`、`description`、`parameters`（JSON Schema），随请求的 `tools` 字段一起发送；"
    "`description` 决定模型“什么时候想起来用它”。",
    "**决策**：模型返回 `tool_calls`（每项含 `id`、`name`、`arguments` 字符串），或直接返回 `content` 结束。",
    "**执行**：本地按 `name` 分发到 Python 函数，解析参数、校验取值、调用依赖（数据库/向量库/HTTP）。",
    "**回填**：把工具结果作为 `role=\"tool\"`、带 `tool_call_id` 与 `name` 的消息追加进对话。",
    "**再推理**：模型基于工具结果生成最终回答；若还需要工具，则重复 2~4 步，直到没有 `tool_calls` 或达到轮数上限。",
])
p("本项目把上限设为 `LLM_MAX_TOOL_ROUNDS=6`，并在超限时强制发起一次“不带 tools”的请求收尾，"
  "既避免模型无限调用，也保证用户一定能拿到回答。")

h2("3.2　工具 Schema 的组织方式")
p("每个工具在所属模块内定义为一个模块级常量字典，再由 `tools/functions.py` 汇总成 `TOOL_SCHEMAS` 列表，"
  "并用 `_SCHEMA_BY_NAME` 建索引。这样做的好处是：Schema 与实现放在同一文件，改参数时不会漏改描述；"
  "同时可以按链路裁剪工具集——意图识别只下发 `detect_intent`，SQL Agent 只下发 `query_database`，"
  "主对话才下发全部 6 个工具。下面以 `web_search` 为例（tools/functions.py:43-68）：")
code("""WEB_SEARCH_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "联网搜索互联网上的实时信息，适合回答行业最新趋势、岗位薪资行情、招聘动态等"
            "本地知识库可能过时的问题。query 传简洁的搜索关键词，不要传整段口语化长句。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，例如「AI 应用开发工程师 薪资 前景」"},
                "top_k": {"type": "integer", "description": "返回结果条数，默认 5，取值范围 1-10",
                          "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
}""", caption="代码 3-1　web_search 的工具 Schema（节选）")

h2("3.3　六个工具的功能定义")
tool_rows = [
    ("detect_intent", "intent_code（1 职业 / 2 课程 / 0 其他）、reason",
     "判断用户问题属于哪类咨询，返回编号与理由", "llm/intent.py", "关键词权重规则兜底"),
    ("web_search", "query、top_k（1~10，默认 5）",
     "联网搜索岗位薪资、行业趋势等实时信息", "llm/web_search.py",
     "Bing→DuckDuckGo→SerpAPI 依次尝试，全失败读本地 career_info，并熔断 120 秒"),
    ("search_market_docs", "query、top_k（默认 RAG_TOP_K=3）",
     "在本地 market 集合做语义检索（8 条职业文档）", "vectorstore/store.py", "返回空列表，不阻断回答"),
    ("search_course_docs", "query、top_k（默认 3）",
     "在本地 course 集合做语义检索（7 条课程文档）", "vectorstore/store.py", "返回空列表，不阻断回答"),
    ("query_database", "sql（单条只读 SELECT/WITH）",
     "查询 SQLite 的 users/chat_history/courses/career_info", "database/sql_agent.py",
     "非法 SQL 返回错误文本，模型可改写后重试"),
    ("save_chat_record", "role（user/assistant）、content、intent_code",
     "把指定消息写入 chat_history，供模型主动留痕", "database/db.py", "返回错误，不影响主流程落库"),
]
table(["工具", "关键参数", "功能", "实现位置", "失败时行为"], tool_rows, [2.9, 3.4, 4.2, 3.7, 2.2],
      caption="表 3-1　六个 Function Calling 工具", size=7.8)

h2("3.4　工具执行契约")
p("所有工具走同一个入口 `execute_tool(name, arguments, context)`，其契约是“永不抛异常”：无论未知工具、参数非法还是"
  "依赖异常，都转成 `{\"ok\": false, \"error\": ...}` 的结果返回。这样模型永远能读到一条可理解的结果消息，"
  "而不是让异常穿透到 Flask 层把整轮对话打断（tools/functions.py:430-477 节选）：")
code("""def execute_tool(name, arguments=None, context=None) -> dict:
    tool_name = str(name or "").strip()
    if not tool_name:
        return {"ok": False, "tool": "", "result": None, "error": "工具名不能为空"}

    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return {"ok": False, "tool": tool_name, "result": None,
                "error": f"未知工具：{tool_name}（可用：{', '.join(get_tool_names())}）"}

    try:
        result = handler(_normalize_arguments(arguments), dict(context or {}))
        return {"ok": True, "tool": tool_name, "result": result, "error": None}
    except ValueError as exc:          # 参数类错误：记录 warning
        return {"ok": False, "tool": tool_name, "result": None, "error": str(exc)}
    except Exception as exc:            # 依赖异常：转成结果，不能打断对话
        return {"ok": False, "tool": tool_name, "result": None,
                "error": f"{type(exc).__name__}: {exc}"}""",
     caption="代码 3-2　工具执行分发（节选，注释有精简）")

h2("3.5　循环控制与流式增量合并")
p("同步链路用 `run_tool_calling_loop`：携带 tools 请求模型 → 有 tool_calls 就逐条执行并以 `role=\"tool\"` 回填 → "
  "再次请求；没有 tool_calls 即返回最终文本，同时把执行过的调用记录一并返回，供前端展示（llm/client.py:446-478 节选）：")
code("""for round_index in range(1, rounds_limit + 1):
    response = chat_completion(working, tools=tools, temperature=temperature)
    tool_calls, content = response["tool_calls"], response["content"]

    if not tool_calls:                       # 模型给出最终回答，循环结束
        return {"content": content, "tool_calls": executed,
                "rounds": round_index, "mode": "llm"}

    working.append(_assistant_message(content, tool_calls))
    for call in tool_calls:
        tool_result = _execute_tool_call(tool_executor, call)   # 内部再兜一层异常
        executed.append({"id": call["id"], "name": call["name"],
                         "arguments": call["arguments"], "result": tool_result})
        working.append(_tool_result_message(call["id"], call["name"], tool_result))""",
     caption="代码 3-3　工具调用循环主体（节选）")
p("流式链路多一层麻烦：`tool_calls` 会按 `index` 拆成多个 delta 分片陆续到达，`arguments` 甚至会被切成几段"
  "不完整的 JSON。`run_streaming_tool_loop` 用 `_merge_tool_delta` 按 index 累积 `id`、`name`、`arguments` 字符串，"
  "待 `finish_reason=tool_calls` 后统一 `json.loads` 再交给执行器；执行完成后继续下一轮流式生成，"
  "并把 `delta` 文本与 `tool_call` 事件实时推给前端。")

h2("3.6　实测：一条问题触发 9 次工具调用")
p("图 3 是真实运行截图。用户问“AI 应用开发工程师现在薪资多少”，模型先调 `web_search` 查行情、"
  "调 `search_market_docs` 查本地职业库，中间再调 `detect_intent` 复核意图，随后又补了一轮搜索，"
  "并用 4 次 `query_database` 分别核对薪资区间、课程与岗位信息，前后共 9 次工具调用，全部成功，最终给出带表格的回答。")
figure("04-职业咨询意图1.png", "图 3　职业咨询（意图 1）：一轮回答内 9 次工具调用全部成功", width=14.6)
p("日志同样记录了这一过程（附录 C 摘录）：`本轮结束：phone=13700002222，intent=1，mode=llm，工具调用=9 次，耗时=15560ms`。"
  "课程咨询（图 4）则是更轻的链路：`search_course_docs` → `detect_intent` → `search_course_docs` → `query_database`，"
  "4 次调用、约 7 秒完成。")

h2("3.7　小结与已知限制")
p("Function Calling 在本项目中不是演示，而是回答质量的主要来源：模型可以自主决定“先搜网络还是先查库”"
  "“要不要用 SQL 核对数字”，并在工具失败时改写参数重试。已知限制有两点：其一，意图识别链路为了稳定拿到结构化结果"
  "使用了强制 `tool_choice`，与 deepseek-flash 的 thinking 模式冲突（HTTP 400），导致意图识别实际一直走关键词规则"
  "（详见 8.1）；其二，模型写 SQL 时偶发臆造列名，首轮会失败、靠下一轮自我修正（详见 8.4）。两处都不影响功能可用性，"
  "但都值得修复。")
