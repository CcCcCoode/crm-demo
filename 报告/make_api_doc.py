# -*- coding: utf-8 -*-
"""生成《SeaChat 接口文档（含并发能力预估）》DOCX。

与 make_report.py 相互独立：本脚本只负责接口文档，可单独运行。
运行：python make_api_doc.py
"""

from __future__ import annotations

import datetime
import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "SeaChat接口文档.docx")

DARK = RGBColor(0x1F, 0x38, 0x64)
MID = RGBColor(0x2E, 0x5C, 0x8A)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0x7F, 0x7F, 0x7F)

BODY_FONT = "宋体"
HEAD_FONT = "微软雅黑"
MONO_FONT = "Consolas"
TEXT_WIDTH_CM = 16.4

doc = Document()

# ---------------------------------------------------------------- 页面与样式
section = doc.sections[0]
section.page_width = Cm(21)
section.page_height = Cm(29.7)
section.top_margin = Cm(2.2)
section.bottom_margin = Cm(2.2)
section.left_margin = Cm(2.3)
section.right_margin = Cm(2.3)

normal = doc.styles["Normal"]
normal.font.name = BODY_FONT
normal.font.size = Pt(10.5)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
normal.paragraph_format.line_spacing = 1.35
normal.paragraph_format.space_after = Pt(3)


def _style_heading(name: str, size: float, color: RGBColor, before: float, after: float) -> None:
    style = doc.styles[name]
    style.font.name = HEAD_FONT
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = color
    style._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)
    style.paragraph_format.keep_with_next = True


_style_heading("Heading 1", 18, DARK, 18, 10)
_style_heading("Heading 2", 13.5, MID, 14, 6)
_style_heading("Heading 3", 11.5, MID, 10, 4)


def add_field(paragraph, instr: str) -> None:
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    text_el = OxmlElement("w:t")
    text_el.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr_el)
    run._r.append(fld_sep)
    run._r.append(text_el)
    run._r.append(fld_end)


# 页脚：SeaChat 接口文档 · 第 X 页 / 共 Y 页
footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
for text, instr in (("SeaChat 接口文档　·　第 ", " PAGE "), (" 页 / 共 ", " NUMPAGES "), (" 页", None)):
    run = footer.add_run(text)
    run.font.size = Pt(8.5)
    run.font.name = BODY_FONT
    run.font.color.rgb = LIGHT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    if instr:
        add_field(footer, instr)
        for r in footer.runs[-1:]:
            r.font.size = Pt(8.5)
            r.font.color.rgb = LIGHT


# ---------------------------------------------------------------- 基础组件
def h1(text: str, page_break: bool = False):
    par = doc.add_paragraph(style="Heading 1")
    if page_break:
        par.paragraph_format.page_break_before = True
    par.add_run(text)
    return par


def h2(text: str):
    par = doc.add_paragraph(style="Heading 2")
    par.add_run(text)
    return par


def h3(text: str):
    par = doc.add_paragraph(style="Heading 3")
    par.add_run(text)
    return par


def rich(par, text: str, size: float | None = None, color: RGBColor | None = None) -> None:
    """解析 **加粗** 与 `等宽` 两种行内标记。"""
    import re

    tokens = re.split(r"(\*\*.+?\*\*|`.+?`)", text)
    for token in tokens:
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            run = par.add_run(token[2:-2])
            run.bold = True
            run.font.name = BODY_FONT
            run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            run = par.add_run(token[1:-1])
            run.font.name = MONO_FONT
            run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
            run.font.size = Pt((size or 10.5) - 1.0)
            run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        else:
            run = par.add_run(token)
            run.font.name = BODY_FONT
            run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        if size and not (token.startswith("`") and token.endswith("`")):
            run.font.size = Pt(size)
        if color is not None and not (token.startswith("`") and token.endswith("`")):
            run.font.color.rgb = color


def p(text: str, size: float = 10.5, first_indent: bool = True, color: RGBColor | None = None,
      space_after: float = 4, align=None):
    par = doc.add_paragraph()
    fmt = par.paragraph_format
    fmt.space_after = Pt(space_after)
    fmt.first_line_indent = Cm(0.74) if first_indent else Cm(0)
    if align is not None:
        par.alignment = align
    rich(par, text, size=size, color=color)
    return par


def bullet(text: str, size: float = 10.5, indent: float = 0.55):
    par = doc.add_paragraph()
    fmt = par.paragraph_format
    fmt.left_indent = Cm(indent)
    fmt.first_line_indent = Cm(-0.35)
    fmt.space_after = Pt(2)
    rich(par, "· " + text, size=size)
    return par


def note(text: str, size: float = 9.2):
    par = doc.add_paragraph()
    fmt = par.paragraph_format
    fmt.left_indent = Cm(0.4)
    fmt.space_before = Pt(2)
    fmt.space_after = Pt(6)
    rich(par, text, size=size, color=GRAY)
    return par


def _shade(paragraph, hex_fill: str) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    pPr.append(shd)


def code(text: str, caption: str | None = None, size: float = 8.3) -> None:
    lines = text.rstrip("\n").split("\n")
    par = doc.add_paragraph()
    fmt = par.paragraph_format
    fmt.space_before = Pt(6)
    fmt.space_after = Pt(4)
    fmt.left_indent = Cm(0.2)
    fmt.line_spacing = 1.05
    for index, line in enumerate(lines):
        run = par.add_run(line)
        run.font.name = MONO_FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0x1B, 0x1B, 0x1B)
        if index < len(lines) - 1:
            run.add_break()
    _shade(par, "F3F4F6")
    if len(lines) <= 14:
        fmt.keep_together = True
    if caption:
        fmt.keep_with_next = True
        add_caption(caption)


def add_caption(text: str) -> None:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(1)
    par.paragraph_format.space_after = Pt(8)
    run = par.add_run(text)
    run.font.size = Pt(8.8)
    run.font.name = BODY_FONT
    run.font.color.rgb = GRAY
    run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)


def _cell_text(cell, text: str, size: float, bold: bool = False, mono: bool = False):
    cell.text = ""
    par = cell.paragraphs[0]
    par.paragraph_format.space_before = Pt(1)
    par.paragraph_format.space_after = Pt(1)
    par.paragraph_format.line_spacing = 1.15
    import re
    for line_index, line in enumerate(str(text).split("\n")):
        for token in re.split(r"(\*\*.+?\*\*|`[^`]+`)", line):
            if not token:
                continue
            is_code = token.startswith("`") and token.endswith("`") and len(token) > 2
            is_bold = token.startswith("**") and token.endswith("**") and len(token) > 4
            run = par.add_run(token[1:-1] if is_code else (token[2:-2] if is_bold else token))
            run.font.size = Pt(size - 0.6 if is_code else size)
            run.bold = bold or is_bold
            run.font.name = MONO_FONT if (mono or is_code) else BODY_FONT
            run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
            if is_code:
                run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        if line_index < len(str(text).split("\n")) - 1:
            par.add_run().add_break()


def _fix_table_layout(tbl, widths) -> None:
    """固定列宽：清掉重复的 tblW/tblLayout 后各写一份，并按 schema 顺序排列。"""
    tblPr = tbl._tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout"):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    total = int(sum(widths) * 567)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(total))
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    order = ["w:tblStyle", "w:tblpPr", "w:tblOverlap", "w:bidiVisual", "w:tblStyleRowBandSize",
             "w:tblStyleColBandSize", "w:tblW", "w:jc", "w:tblCellSpacing", "w:tblInd",
             "w:tblBorders", "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook"]
    tblPr.append(tblW)
    tblPr.append(layout)
    children = list(tblPr)
    children.sort(key=lambda el: order.index(el.tag.replace("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", "w:"))
                  if el.tag.replace("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", "w:") in order else 99)
    for el in children:
        tblPr.remove(el)
    for el in children:
        tblPr.append(el)


def _borders(tbl) -> None:
    tblPr = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "BFBFBF")
        borders.append(el)
    tblPr.append(borders)


def _cell_margins(tbl) -> None:
    tblPr = tbl._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, width in (("top", 18), ("left", 72), ("bottom", 18), ("right", 72)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(width))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def table(headers, rows, widths, caption: str | None = None, size: float = 8.2,
          mono_cols=()) -> None:
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    _borders(tbl)
    _cell_margins(tbl)
    hdr = tbl.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))
    for index, text in enumerate(headers):
        cell = hdr.cells[index]
        _cell_text(cell, text, size, bold=True)
        pPr = cell.paragraphs[0]._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "E8EEF7")
        pPr.append(shd)
        cell.paragraphs[0].paragraph_format.keep_with_next = True
    for row in rows:
        cells = tbl.add_row().cells
        for index, value in enumerate(row):
            _cell_text(cells[index], value, size, mono=index in mono_cols)
    for row in tbl.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        for index, width in enumerate(widths):
            row.cells[index].width = Cm(width)
    for index, width in enumerate(widths):
        tbl.columns[index].width = Cm(width)
    if len(rows) <= 6:
        for row in tbl.rows[:-1]:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True
    _fix_table_layout(tbl, widths)
    if caption:
        for cell in tbl.rows[-1].cells:
            for par in cell.paragraphs:
                par.paragraph_format.keep_with_next = True
        add_caption(caption)


# ---------------------------------------------------------------- 封面
for _ in range(5):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("SeaChat AI 客服咨询系统")
run.font.size = Pt(28)
run.font.bold = True
run.font.name = HEAD_FONT
run.font.color.rgb = DARK
run._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_before = Pt(10)
run = sub.add_run("接口文档与并发能力预估")
run.font.size = Pt(19)
run.font.bold = True
run.font.name = HEAD_FONT
run.font.color.rgb = MID
run._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)

tag = doc.add_paragraph()
tag.alignment = WD_ALIGN_PARAGRAPH.CENTER
tag.paragraph_format.space_before = Pt(26)
run = tag.add_run("HTTP / JSON　·　SSE 流式　·　Flask 3.1")
run.font.size = Pt(10.5)
run.font.name = BODY_FONT
run.font.color.rgb = GRAY
run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

for _ in range(6):
    doc.add_paragraph()

info = doc.add_table(rows=0, cols=2)
info.alignment = WD_TABLE_ALIGNMENT.CENTER
meta = [
    ("文档版本", "v1.0（与代码提交同步）"),
    ("接口数量", "11 个（页面 1、健康检查 1、对话 1、向量库 6、数据库 2）"),
    ("基础地址", "http://127.0.0.1:5000"),
    ("协议", "HTTP/1.1，UTF-8，JSON；/chat 支持 SSE"),
    ("实测环境", "Windows 11 · AMD Ryzen 7 4800H (8C/16T) · Python 3.14.3"),
    ("编写日期", datetime.date.today().strftime("%Y 年 %m 月 %d 日")),
]
for key, value in meta:
    cells = info.add_row().cells
    _cell_text(cells[0], key, 9.5, bold=True)
    _cell_text(cells[1], value, 9.5)
    cells[0].width = Cm(3.4)
    cells[1].width = Cm(11.6)
for row in info.rows:
    row.cells[0].width = Cm(3.4)
    row.cells[1].width = Cm(11.6)
_fix_table_layout(info, [3.4, 11.6])

# ---------------------------------------------------------------- 1 概述
h1("1　概述", page_break=True)
p("SeaChat 是面向 AI 职业教育机构的售前咨询系统：用户输入手机号与问题，服务端完成意图识别、"
  "素材检索（联网搜索 / 本地向量库 / 本地数据库）、调用 DeepSeek-Flash 生成回答，并把本轮对话与"
  "用户画像增量写回 SQLite。所有能力以 HTTP 接口对外开放，前端页面只是其中一个调用方。")
p("文档覆盖 11 个接口的请求参数、响应字段、错误语义与真实返回示例；第 9 章给出基于实测数据的"
  "并发能力预估与部署建议。文中所有示例均为本机真实调用结果（非构造示例），涉及密钥与手机号"
  "的部分已脱敏。")

h2("1.1　技术栈与运行方式")
table(
    ["项目", "说明"],
    [
        ("Web 框架", "Flask 3.1.3 + Werkzeug 3.1.3（开发服务器 threaded=True）"),
        ("大模型", "deepseek-flash，OpenAI 兼容接口 https://api.deepseek.com"),
        ("向量库", "ChromaDB 1.5.9 本地持久化（data/chroma），集合 market / course"),
        ("嵌入模型", "BAAI/bge-small-zh-v1.5（512 维，CPU），缺失时降级 hash"),
        ("关系库", "SQLite 3（data/crm.db，4 张业务表）"),
        ("启动命令", "python app.py　　默认 0.0.0.0:5000，可用 FLASK_PORT 覆盖"),
    ],
    [3.0, 13.4],
    caption="表 1-1　运行环境",
)

# ---------------------------------------------------------------- 2 通用约定
h1("2　通用约定")
h2("2.1　请求与响应格式")
bullet("除 `GET /` 返回 HTML 外，所有接口的请求体与响应体均为 `application/json; charset=utf-8`。")
bullet("请求头建议显式声明 `Content-Type: application/json`；未带请求体时按空对象处理。")
bullet("响应 JSON 的顶层统一含布尔字段 `ok`：成功为 `true`，失败为 `false` 且带 `error` 文本。")
bullet("时间字段为本地时间字符串，格式 `YYYY-MM-DDTHH:MM:SS`（秒级，无时区后缀）。")
bullet("中文不做 \\uXXXX 转义，便于直接阅读与验收比对。")

h2("2.2　统一错误结构")
p("所有错误（含 404/405/413/500）都返回同一结构，不会出现 Flask 默认的 HTML 错误页：")
code('{\n  "ok": false,\n  "error": "缺少 collection 参数：market、course"\n}', "代码 2-1　统一错误响应")
table(
    ["HTTP 状态", "含义", "常见触发原因"],
    [
        ("200", "成功", "正常返回；`/database/query` 的“查不出结果”也可能是 200 + ok=false"),
        ("400", "参数错误", "缺字段、类型不符、集合名不在白名单、手机号格式错误"),
        ("404", "接口不存在", "路径拼写错误"),
        ("405", "方法不允许", "对只支持 POST 的接口发了 GET"),
        ("413", "请求体过大", "超过 MAX_CONTENT_LENGTH = 4 MB"),
        ("500", "服务内部错误", "向量库/数据库/SQL Agent 异常，细节见服务端日志"),
    ],
    [2.2, 3.2, 11.0],
    caption="表 2-1　状态码约定",
)

h2("2.3　跨域、体积与超时")
bullet("CORS：响应统一带 `Access-Control-Allow-Origin: *`、`Allow-Methods: GET, POST, OPTIONS`、"
       "`Allow-Headers: Content-Type, Authorization`，预检缓存 86400 秒（演示配置，生产需按域名收紧）。")
bullet("体积：整个请求体上限 4 MB，超出返回 413。单条文档建议不超过 2 KB。")
bullet("超时：服务端调用大模型的超时为 60 秒；`/chat` 流式接口本身不设总时长限制，"
       "但单轮工具调用最多 6 轮。")

# ---------------------------------------------------------------- 3 接口总览
h1("3　接口总览")
table(
    ["#", "方法", "路径", "用途", "返回形式"],
    [
        ("1", "GET", "/", "内置聊天页面", "HTML"),
        ("2", "GET", "/health", "各组件健康状态", "JSON"),
        ("3", "POST", "/chat", "发起咨询（核心接口）", "SSE / JSON"),
        ("4", "POST", "/vectorstore/add", "新增向量文档", "JSON"),
        ("5", "POST", "/vectorstore/update", "按 id 覆盖更新", "JSON"),
        ("6", "POST", "/vectorstore/delete", "按 id 删除", "JSON"),
        ("7", "GET", "/vectorstore/list", "列出集合与文档数", "JSON"),
        ("8", "POST", "/vectorstore/search", "向量检索（验收更新是否生效）", "JSON"),
        ("9", "POST", "/vectorstore/init", "幂等补齐示例数据", "JSON"),
        ("10", "POST", "/database/query", "SQL Agent：自然语言查库", "JSON"),
        ("11", "GET", "/database/schema", "表结构说明", "JSON"),
    ],
    [1.0, 1.6, 4.3, 6.7, 2.8],
    caption="表 3-1　接口总览",
)

# ---------------------------------------------------------------- 4 对话接口
h1("4　对话接口 POST /chat")
p("核心接口。按 `stream` 参数返回 SSE 流（默认）或一次性 JSON；两种模式的业务结果字段一致，"
  "差别只在传输方式。")
h2("4.1　请求参数")
table(
    ["参数", "类型", "必填", "说明"],
    [
        ("phone_number", "string", "是", "11 位大陆手机号（1 开头、第二位 3~9）；兼容别名 `phone`"),
        ("query", "string", "是", "用户问题，兼容别名 `message`；非空且 ≤ 2000 字"),
        ("stream", "bool", "否", "默认 true（SSE）；false / \"0\" / \"no\" / \"off\" 时返回普通 JSON"),
    ],
    [3.2, 1.8, 1.4, 10.0],
    caption="表 4-1　/chat 请求参数",
)
code('curl -X POST http://127.0.0.1:5000/chat \\\n'
     '  -H "Content-Type: application/json" \\\n'
     '  -d \'{"phone_number":"13700002222","query":"你们 AI 应用开发课程多少钱"}\'',
     "代码 4-1　调用示例（SSE）")

h2("4.2　SSE 事件流")
p("响应头为 `Content-Type: text/event-stream; charset=utf-8`，并带 `Cache-Control: no-cache`、"
  "`X-Accel-Buffering: no`（禁用 Nginx 缓冲）。每帧格式固定为 `data: {JSON}\\n\\n`，"
  "以 `data: [DONE]` 结束；即使中途异常也会补发结束帧。")
table(
    ["事件 type", "关键字段", "说明"],
    [
        ("start", "phone_number、is_new、intent_code、intent_name、mode", "会话开始；新用户直接以 welcome 模式返回"),
        ("intent", "intent_code、intent_name、reason、mode", "意图识别结果；mode 为识别方式（llm/rule）"),
        ("tool_call", "name、arguments、ok、error", "一次工具调用，前端可折叠展示"),
        ("delta", "content", "增量文本，按到达顺序拼接即为完整回答"),
        ("done", "reply、intent_code、intent_name、mode、tool_calls、user_info、rounds、elapsed_ms", "本轮结束；mode 为回答生成方式"),
        ("error", "message", "异常提示，随后仍会收到 [DONE]"),
    ],
    [2.2, 8.4, 5.8],
    caption="表 4-2　SSE 事件类型",
)
p("需要特别注意：`start` / `intent` 帧里的 `mode` 表示**意图识别**方式（llm 或 rule），"
  "`done` 帧里的 `mode` 表示**回答生成**方式。两者可能不同，例如“规则识别意图 + 大模型生成回答”。")
code('data: {"type": "start", "phone_number": "13700002222", "is_new": false, '
     '"intent_code": 2, "intent_name": "课程咨询", "mode": "rule"}\n\n'
     'data: {"type": "intent", "intent_code": 2, "intent_name": "课程咨询", '
     '"reason": "命中课程类关键词「课程、多少钱」，判断为课程咨询", "mode": "rule"}\n\n'
     'data: {"type": "tool_call", "name": "search_course_docs", '
     '"arguments": {"query": "AI 应用开发课程多少钱？", "top_k": 3}, "ok": true, "error": null}\n\n'
     'data: {"type": "delta", "content": "## Python 数据分析与可视化（课程编号 course-002）"}\n\n'
     'data: {"type": "done", "reply": "……", "intent_code": 2, "mode": "llm", '
     '"rounds": 2, "elapsed_ms": 6880, "tool_calls": [...]}\n\n'
     'data: [DONE]', "代码 4-2　真实 SSE 帧（截断）")

h2("4.3　非流式响应（stream=false）")
p("一次性返回聚合结果，字段与 `done` 帧一致，便于脚本或压测使用：")
table(
    ["字段", "类型", "说明"],
    [
        ("ok", "bool", "是否成功"),
        ("reply", "string", "完整回答（Markdown 文本）"),
        ("phone_number", "string", "规范化后的手机号"),
        ("is_new", "bool", "是否本次新建用户"),
        ("intent_code / intent_name", "int / string", "1 职业咨询、2 课程咨询、0 其他"),
        ("mode", "string", "llm（大模型生成）/ rule / fallback / welcome"),
        ("rounds", "int", "工具调用循环轮次（仅 llm 模式）"),
        ("elapsed_ms", "int", "服务端耗时（毫秒）"),
        ("tool_calls", "array", "本轮调用的工具列表：name、arguments、ok、error"),
        ("user_info", "object", "最新画像：name、intention_direction、budget、stage、notes"),
    ],
    [4.4, 2.2, 9.8],
    caption="表 4-3　/chat 非流式响应字段",
)
code('{\n  "ok": true,\n  "reply": "## Python 数据分析与可视化（课程编号 course-002）……",\n'
     '  "phone_number": "13700002222",\n  "is_new": false,\n  "intent_code": 2,\n'
     '  "intent_name": "课程咨询",\n  "mode": "llm",\n  "rounds": 2,\n  "elapsed_ms": 6880,\n'
     '  "tool_calls": [{"name": "search_course_docs", "arguments": {"query": "……", "top_k": 3},'
     ' "ok": true, "error": null}],\n'
     '  "user_info": {"phone_number": "13700002222", "name": null, '
     '"intention_direction": "Python 数据分析", "budget": null, "stage": "已了解", "notes": "……"}\n}',
     "代码 4-3　非流式真实响应（脱敏截断）")
note("接口幂等性：同一手机号重复调用会不断累加对话历史，但画像字段只做增量补全，不会重复写入。")

# ---------------------------------------------------------------- 5 向量库接口
h1("5　向量库接口")
p("6 个接口构成向量库的完整维护面：新增、更新、删除、列表、检索、初始化。集合白名单只有 "
  "`market`（8 条）与 `course`（7 条）两个；写入的文档由服务端调用 BGE 计算向量，调用方无需关心向量本身。")
h2("5.1　新增文档 POST /vectorstore/add")
table(
    ["参数", "类型", "必填", "说明"],
    [
        ("collection", "string", "是", "market 或 course"),
        ("documents", "string[]", "是", "非空文档数组，单条建议 ≤ 2 KB"),
        ("ids", "string[]", "否", "不传则自动生成 `{collection}-{uuid4}`；长度须与 documents 一致"),
        ("metadatas", "object[]", "否", "与 documents 等长的元数据数组，值须为标量"),
    ],
    [2.8, 2.0, 1.4, 10.2],
    caption="表 5-1　/vectorstore/add 请求参数",
)
code('curl -X POST http://127.0.0.1:5000/vectorstore/add \\\n'
     '  -H "Content-Type: application/json" \\\n'
     '  -d \'{"collection":"course","documents":["办公自动化训练营：3980 元，8 周"],'
     '"ids":["report-demo-001"]}\'',
     "代码 5-1　新增文档")
p("响应返回写入的 id 列表、本批条数与集合当前总数，便于调用方对账：")
code('{\n  "ok": true,\n  "collection": "course",\n  "ids": ["report-demo-001"],\n'
     '  "count": 1,\n  "stats": 8\n}', "代码 5-2　新增文档响应（真实返回）")

h2("5.2　更新文档 POST /vectorstore/update")
p("按 id 覆盖：服务端先 `get(ids)` 检查存在性，不存在的 id 会被忽略（不报错），"
  "存在的文档重新计算向量后覆盖。请求体为 `collection` + `ids` + `documents`（+ 可选 `metadatas`），"
  "`ids` 与 `documents` 必须等长。")
code('{\n  "ok": true,\n  "collection": "course",\n  "ids": ["report-demo-001"],\n  "updated": 1\n}',
     "代码 5-3　更新响应（真实返回）")

h2("5.3　删除文档 POST /vectorstore/delete")
p("按 id 删除，删除不存在的 id 不报错（幂等）。响应带剩余文档数：")
code('{\n  "ok": true,\n  "collection": "course",\n  "deleted": ["report-demo-001"],\n  "stats": 7\n}',
     "代码 5-4　删除响应（真实返回）")

h2("5.4　集合列表 GET /vectorstore/list")
code('{\n  "ok": true,\n  "collections": [{"name": "course", "count": 7}, {"name": "market", "count": 8}],\n'
     '  "count": 2\n}', "代码 5-5　列表响应（真实返回）")

h2("5.5　向量检索 POST /vectorstore/search")
table(
    ["参数", "类型", "必填", "说明"],
    [
        ("collection", "string", "是", "market 或 course"),
        ("query", "string", "是", "检索语句，内部会做一次嵌入"),
        ("top_k", "int", "否", "返回条数，默认 3（RAG_TOP_K）"),
    ],
    [2.8, 2.0, 1.4, 10.2],
    caption="表 5-2　/vectorstore/search 请求参数",
)
p("每条结果含 `id`、`document`、`metadata` 与 `distance`（余弦距离，等于 1 − 余弦相似度，越小越相关）：")
code('{\n  "ok": true,\n  "collection": "course",\n  "query": "办公自动化 多少钱",\n'
     '  "results": [\n    {"id": "report-demo-001", "document": "办公自动化训练营：3980 元，8 周", '
     '"metadata": {}, "distance": 0.3167},\n'
     '    {"id": "course-007", "document": "……", "metadata": {"price": 3980}, "distance": 0.3640}\n  ],\n'
     '  "count": 2\n}', "代码 5-6　检索响应（真实返回，新写入文档排第一）")

h2("5.6　初始化 POST /vectorstore/init")
p("仅在集合为空时灌入示例数据，重复调用不会产生重复文档，可用于首次部署或数据被清空后的恢复。"
  "响应返回两个集合的即时统计。")

# ---------------------------------------------------------------- 6 数据库接口
h1("6　数据库接口")
h2("6.1　SQL Agent POST /database/query")
table(
    ["参数", "类型", "必填", "说明"],
    [
        ("question", "string", "是", "自然语言问题，兼容别名 `query`"),
    ],
    [2.8, 2.0, 1.4, 10.2],
    caption="表 6-1　/database/query 请求参数",
)
p("服务端先尝试大模型生成 SQL（严格只读），失败或未配置 Key 时退化为规则匹配。"
  "响应同时返回 `sql`、`rows` 与中文 `answer`，便于核对“它到底查了什么”：")
table(
    ["字段", "类型", "说明"],
    [
        ("ok", "bool", "是否得到结果；无匹配时 false，HTTP 状态为 400"),
        ("mode", "string", "llm（模型生成 SQL）或 rule（规则兜底）"),
        ("sql", "string", "实际执行的 SQL（只读）"),
        ("rows", "array", "查询结果，最多 SQL_MAX_ROWS = 100 行"),
        ("answer", "string", "中文总结，可能含 Markdown 表格"),
        ("error", "string", "失败原因（仅 ok=false 时出现）"),
    ],
    [2.6, 2.0, 11.8],
    caption="表 6-2　/database/query 响应字段",
)
code('{\n  "ok": true,\n  "mode": "llm",\n'
     '  "sql": "SELECT title, industry, salary_min, salary_max, skills FROM career_info '
     'ORDER BY salary_max DESC LIMIT 3",\n'
     '  "rows": [["大模型算法工程师", "人工智能", 25000, 50000, "……"], "……"],\n'
     '  "answer": "薪资最高的三个职业是……"\n}', "代码 6-1　真实返回（rows 截断）")

h2("6.2　表结构 GET /database/schema")
p("返回 4 张表的建表说明与示例数据，供调试与提示词核对：")
code('{\n  "ok": true,\n  "tables": ["career_info", "chat_history", "courses", "users"],\n'
     '  "schema": "## 表：users\\n……（约 3.4 KB 文本）"\n}', "代码 6-2　表结构响应")

# ---------------------------------------------------------------- 7 健康检查
h1("7　健康检查与页面")
h2("7.1　GET /health")
p("汇总大模型、嵌入、向量库、数据库、联网搜索五组件的状态，是部署自检与前端状态灯的数据源。"
  "任一组件降级都不会让该接口失败，而是在对应字段里体现。")
code('{\n  "ok": true, "status": "ok", "service": "SeaChatInterview AI CRM",\n'
     '  "timestamp": "2026-11-18T13:00:39", "data_dir": "C:\\\\Users\\\\IKUN\\\\Desktop\\\\crm项目\\\\data",\n'
     '  "llm": {"available": true, "base_url": "https://api.deepseek.com", "model": "deepseek-flash"},\n'
     '  "embedding": {"configured_provider": "bge", "provider": "bge", '
     '"model": "BAAI/bge-small-zh-v1.5", "dim": 512, "backend": "sentence-transformers", "ready": true},\n'
     '  "vectorstore": [{"name": "course", "count": 7}, {"name": "market", "count": 8}],\n'
     '  "database": {"users": 21, "chat_history": 139, "courses": 7, "career_info": 8},\n'
     '  "search": {"enabled": true, "provider": "bing", "calls": 4, "fallback_used": false, "last_error": null}\n}',
     "代码 7-1　真实健康检查响应")

h2("7.2　GET /")
p("返回内置聊天页（`templates/index.html`），页面加载时请求一次 `/health` 点亮状态灯，"
  "之后通过 `fetch` + `ReadableStream` 消费 `/chat` 的 SSE 帧。")

# ---------------------------------------------------------------- 8 错误与排错
h1("8　错误码与排错")
table(
    ["现象", "可能原因", "排查方式"],
    [
        ("400 手机号格式不正确", "未按 11 位大陆手机号传入", "检查 `phone_number`，注意前后空格"),
        ("400 缺少 collection 参数", "集合名为空或拼写错误", "只允许 market / course"),
        ("400 不支持的集合名", "传了白名单外的名字", "见响应内的可选值提示"),
        ("400 ids 与 documents 长度必须一致", "更新时数组不等长", "逐条对齐后再提交"),
        ("400 无法将问题映射为数据库查询", "问题与库内字段无关", "换用课程/薪资/人数类问题"),
        ("413 Payload Too Large", "请求体超过 4 MB", "拆分批量写入"),
        ("500 向量检索失败", "嵌入模型或 Chroma 异常", "看服务端日志；必要时 `POST /vectorstore/init`"),
        ("/chat 一直返回 rule 模式", "未配置 Key 或上游 4xx", "`GET /health` 看 `llm.available` 与日志"),
    ],
    [5.6, 5.0, 5.8],
    caption="表 8-1　常见错误与排查",
)

# ---------------------------------------------------------------- 9 并发预估
h1("9　并发能力预估")
h2("9.1　测试方法")
p("压测脚本 `报告/bench.py`（标准库 urllib + threading，无第三方依赖）对 `/chat`（stream=false）"
  "与 `/health` 分别施压，并发度 1/2/4/8/16，每档固定请求数，统计 QPS、p50、p95 与最大时延。"
  "为隔离变量，压测分两组：A 组为规则模式（无 Key、hash 嵌入、关闭联网），B 组为大模型模式"
  "（真实 Key、关闭联网、嵌入仍用 hash）。")
note("注意：`$env:DEEPSEEK_API_KEY=\"\"` 在 PowerShell 中等于删除变量，python-dotenv 会重新读回 .env 里的真实 Key；"
     "因此规则模式必须用非空占位（空格）来强制降级，否则测到的仍是大模型模式。")

h2("9.2　实测数据")
table(
    ["并发", "A 组 QPS", "A 组 p50", "A 组 p95", "B 组 QPS", "B 组 p50", "B 组 p95"],
    [
        ("1", "31.7", "32 ms", "—", "0.13", "7.3 s", "—"),
        ("2", "51.5", "20 ms", "—", "0.24", "7.9 s", "—"),
        ("4", "52.4", "48 ms", "—", "0.51", "7.4 s", "—"),
        ("8", "41.6", "126 ms", "275 ms", "1.02", "7.4 s", "—"),
        ("16", "40.3", "131 ms", "582 ms", "2.03", "7.2 s", "8.3 s"),
    ],
    [1.6, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4],
    caption="表 9-1　/chat 两组压测结果（本机单进程）",
)
table(
    ["并发", "A 组 /health QPS", "A 组 p50", "B 组 /health QPS", "B 组 p50"],
    [
        ("4", "206.8", "11 ms", "178", "16 ms"),
        ("16", "301.0", "51 ms", "295", "52 ms"),
    ],
    [2.0, 4.0, 2.6, 4.0, 2.6],
    caption="表 9-2　/health 压测结果",
)

h2("9.3　数据解读")
bullet("**规则模式**：QPS 在并发 4 时达到峰值 52，继续加并发反而下降到 40 左右——瓶颈是 GIL 与"
       "SQLite/Chroma 的串行段，此时 p95 从 275 ms 涨到 582 ms，属于典型的“过载拐点”。")
bullet("**大模型模式**：每请求耗时稳定在 7.2~7.9 秒，几乎与并发无关；QPS 随并发线性增长（0.13→2.03），"
       "说明时间几乎全部花在等待上游 API，线程可并行，瓶颈不在本机。")
bullet("**健康检查**：纯内存计算，300 QPS 量级，远高于业务接口，可放心用于探活。")

h2("9.4　容量模型与结论")
p("按 Little 定律 L = λ × W 估算：规则模式取 40 QPS 保守值、大模型模式取 2 QPS 实测值，"
  "以“每用户每日 10 次对话、峰值集中在 4 小时、峰值系数 3”折算：")
table(
    ["模式", "可持续 QPS", "峰值可承接请求", "折算日活用户"],
    [
        ("规则 / 离线降级", "40", "4 h × 3600 × 40 / 3 ≈ 19.2 万", "约 19,000"),
        ("大模型（单进程）", "2", "4 h × 3600 × 2 / 3 ≈ 9,600", "约 960"),
        ("大模型（4 worker × 8 线程）", "约 8（受上游配额约束）", "约 3.8 万", "约 3,800"),
    ],
    [4.6, 3.4, 5.6, 2.8],
    caption="表 9-3　容量估算（每用户每日 10 次）",
)
p("结论：单机单进程在本机环境下，规则模式可支撑约 40 QPS（日活万级），大模型模式约 2 QPS（日活千级）。"
  "真实业务的瓶颈几乎总是上游大模型的 TPM/RPM 配额，而非本服务；因此扩容优先级为"
  "“上游配额 > 进程/线程数 > CPU”，降本优先级为“缓存 > 少调工具 > 换更小的模型”。")
p("部署建议：生产环境用 `gunicorn -w 4 -k gthread --threads 8`（32 并发）+ Nginx 反向代理；"
  "为 `/chat` 设置 60 秒读超时；对同一手机号做简单限流（如每分钟 6 次）；"
  "用 Redis 缓存常见问题答案与 query 向量；监控至少覆盖上游 429/5xx 比例、p95 时延、工具失败率。")

# ---------------------------------------------------------------- 附录
h1("附录　curl 速查", page_break=True)
p("以下命令可直接复制到终端执行（Windows PowerShell 请将行末的 \\ 换成反引号，或写成单行）。")

code('# 健康检查\ncurl http://127.0.0.1:5000/health\n\n'
     '# 非流式对话\ncurl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" \\\n'
     '  -d \'{"phone_number":"13700002222","query":"AI 应用开发工程师薪资多少","stream":false}\'',
     "代码 A-1　健康检查与非流式对话")

code('# 列出集合\ncurl http://127.0.0.1:5000/vectorstore/list\n\n'
     '# 新增\ncurl -X POST http://127.0.0.1:5000/vectorstore/add -H "Content-Type: application/json" \\\n'
     '  -d \'{"collection":"course","documents":["测试文档：3980 元"]}\'\n\n'
     '# 检索\ncurl -X POST http://127.0.0.1:5000/vectorstore/search -H "Content-Type: application/json" \\\n'
     '  -d \'{"collection":"course","query":"多少钱","top_k":2}\'\n\n'
     '# 删除\ncurl -X POST http://127.0.0.1:5000/vectorstore/delete -H "Content-Type: application/json" \\\n'
     '  -d \'{"collection":"course","ids":["course-xxxx"]}\'',
     "代码 A-2　向量库运维")

code('# SQL Agent：自然语言查库\n'
     'curl -X POST http://127.0.0.1:5000/database/query -H "Content-Type: application/json" \\\n'
     '  -d \'{"question":"薪资最高的三个职业是什么"}\'\n\n'
     '# 表结构\ncurl http://127.0.0.1:5000/database/schema',
     "代码 A-3　SQL Agent 与表结构")

h2("变更记录")
table(
    ["版本", "日期", "说明"],
    [
        ("v1.0", "2026-11-18", "首版：11 个接口的请求/响应说明、SSE 事件表、错误码与并发能力预估"),
    ],
    [2.4, 3.2, 10.8],
)

doc.save(OUT)
print("saved:", OUT)
