"""联网搜索模块：SerpAPI / DuckDuckGo HTML 抓取 + 本地职业知识库兜底。

设计约束（见 docs/CONTRACT.md §5）：
1. 只用 ``requests`` + 标准库 ``html.parser`` 解析 DuckDuckGo，不引入 bs4；
2. SerpAPI 有 Key 时优先，无 Key / 失败则尝试 DuckDuckGo 的 html、lite 两个页面；
3. 全部失败或 ``WEB_SEARCH_ENABLED=false`` 时降级读 SQLite ``career_info`` 表，
   返回 ``source="本地知识库兜底"``、``provider="local"`` 的本地结果；
4. 本模块独立，绝不 import ``tools/``，任何情况下都不向调用方抛异常。

搜索状态（calls / last_error / fallback_used）通过模块级字典维护，
方便 /api/status 之类的接口展示「搜索当前是否健康」。
"""

from __future__ import annotations

import html as _html
import re
import sqlite3
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from config import config, get_logger

logger = get_logger("crm.llm.web_search")

# requests 属于运行时依赖，这里做一次容错导入：
# 即使环境缺失 requests，web_search 也应降级到本地知识库而不是抛 ImportError。
try:  # pragma: no cover - 依赖已安装，容错分支仅用于极端环境
    import requests
except Exception as _requests_exc:  # noqa: BLE001
    requests = None  # type: ignore[assignment]
    _REQUESTS_IMPORT_ERROR: str | None = str(_requests_exc)
else:
    _REQUESTS_IMPORT_ERROR = None

# 远程结果统一标记来源
REMOTE_SOURCE = "网络搜索"
LOCAL_SOURCE = "本地知识库兜底"

# DuckDuckGo 页面地址（按顺序尝试）
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"
# SerpAPI 搜索地址
SERPAPI_URL = "https://serpapi.com/search.json"
# Bing 国内站搜索地址（cn.bing.com 在大部分网络环境可直连，作为 DuckDuckGo 的备用源）
BING_SEARCH_URL = "https://cn.bing.com/search"

# 浏览器 UA：DuckDuckGo 对默认 python-requests UA 较敏感，伪装成普通浏览器
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://duckduckgo.com/",
}

# 搜索状态：calls=web_search 调用次数；fallback_used=是否用过本地兜底；
# last_error=最近一次远程搜索失败原因；cooldown_until=远程搜索失败后的熔断截止时间
_status: dict[str, Any] = {
    "calls": 0,
    "last_error": None,
    "fallback_used": False,
    "cooldown_until": 0.0,
}

# 远程搜索失败后的熔断冷却时间（秒）。
# 本机/服务器网络不可达 DuckDuckGo 时，如果不熔断，每轮对话都会卡满
# 「页面数 × WEB_SEARCH_TIMEOUT」的超时（实测一次职业咨询要等 50+ 秒）。
# 冷却期内 web_search 直接走 SQLite 本地知识库兜底，保证对话响应速度。
REMOTE_COOLDOWN_SECONDS = 120.0

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _clean_text(text: str | None) -> str:
    """压缩空白字符并去掉首尾空格（HTML 抓取结果里常有多余换行）。"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _decode_ddg_url(href: str) -> str:
    """还原 DuckDuckGo 的跳转链接。

    DuckDuckGo 的搜索结果链接常写成 ``//duckduckgo.com/l/?uddg=<urlencoded>``，
    需要取出 ``uddg`` 参数并 url-decode；普通 http(s) 链接原样返回。
    """
    raw = (href or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if "duckduckgo.com/l/" in lowered or "/l/?uddg=" in lowered or "uddg=" in lowered:
        try:
            query = parse_qs(urlparse(raw if raw.startswith("http") else "https:" + raw).query)
            target = query.get("uddg")
            if target:
                decoded = unquote(target[0])
                if decoded.startswith(("http://", "https://")):
                    return decoded
        except Exception:  # noqa: BLE001 - 单个链接解析失败不影响整体
            return ""
    if raw.startswith("//"):
        return "https:" + raw
    return raw


def _truncate(text: str, limit: int = 300) -> str:
    """截断过长摘要，避免结果列表臃肿。"""
    text = _clean_text(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# DuckDuckGo HTML 解析（标准库 html.parser，禁止 bs4）
# ---------------------------------------------------------------------------
_TITLE_CLASS_TOKENS = ("result__a", "result-link")
_SNIPPET_CLASS_TOKENS = ("result__snippet", "result-snippet")


def _has_class(attrs: list[tuple[str, str | None]], tokens: tuple[str, ...]) -> bool:
    """判断标签的 class 是否包含任一目标 token（大小写不敏感）。"""
    class_text = ""
    for name, value in attrs:
        if name == "class" and value:
            class_text += " " + str(value)
    lowered = class_text.lower()
    return any(token in lowered for token in tokens)


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str:
    """从标签属性列表中取某个属性的值，不存在返回空串。"""
    for key, value in attrs:
        if key == name:
            return str(value or "")
    return ""


class _DuckDuckGoParser(HTMLParser):
    """解析 DuckDuckGo html / lite 页面，提取标题、摘要与链接。

    html 版结构：``<a class="result__a" href="...">标题</a>`` +
    ``<a class="result__snippet">摘要</a>``；
    lite 版结构：``<a class="result-link" ...>标题</a>`` +
    ``<td class="result-snippet">摘要</td>``。
    两种结构统一处理，按出现顺序收集结果。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        # 正在收集文本的锚点/摘要位置
        self._in_title = False
        self._in_snippet = False
        self._title_buffer: list[str] = []
        self._snippet_buffer: list[str] = []
        self._current_href = ""
        # lite 版把 class="result-link" 放在 <td>/<div> 容器上，内部 <a> 不带 class，
        # 因此需要记录「当前是否处于结果标题单元格中」
        self._in_link_cell = False
        self._link_cell_tag = ""

    # ---- HTMLParser 回调 ----
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # lite 版：<td class="result-link"><a href="...">标题</a></td>
        if tag in ("td", "div", "span") and _has_class(attrs, ("result-link",)):
            self._in_link_cell = True
            self._link_cell_tag = tag
            return
        # 摘要可能是 <a class="result__snippet"> 或 <td class="result-snippet">
        if _has_class(attrs, _SNIPPET_CLASS_TOKENS) and not self._in_snippet:
            self._in_snippet = True
            self._snippet_buffer = []
            return
        # html 版锚点自带 result__a/result-link；lite 版锚点在标题单元格内但没有 class
        if tag == "a" and not self._in_title and (
            _has_class(attrs, _TITLE_CLASS_TOKENS) or self._in_link_cell
        ):
            self._flush_snippet()
            self._in_title = True
            self._title_buffer = []
            self._current_href = _decode_ddg_url(_attr(attrs, "href"))

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            title = _clean_text("".join(self._title_buffer))
            href = self._current_href
            self._current_href = ""
            if title and href and href.startswith(("http://", "https://")):
                self.results.append({"title": title, "snippet": "", "url": href})
            return
        if self._in_link_cell and tag == self._link_cell_tag:
            self._in_link_cell = False
            self._link_cell_tag = ""
            self._flush_snippet()
            return
        if tag in ("a", "td", "div") and self._in_snippet:
            self._flush_snippet()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buffer.append(data)
        elif self._in_snippet:
            self._snippet_buffer.append(data)

    # ---- 内部方法 ----
    def _flush_snippet(self) -> None:
        """把已收集的摘要文本挂到最近一条还没摘要的结果上。"""
        if not self._in_snippet:
            return
        self._in_snippet = False
        snippet = _clean_text("".join(self._snippet_buffer))
        self._snippet_buffer = []
        if not snippet:
            return
        for item in reversed(self.results):
            if not item["snippet"]:
                item["snippet"] = snippet
                break


def _parse_duckduckgo(html_text: str) -> list[dict[str, str]]:
    """把 DuckDuckGo 页面 HTML 解析成若干 ``{"title","snippet","url"}``。

    Args:
        html_text: 页面 HTML 文本。

    Returns:
        解析出的原始结果列表（可能为空，不抛异常）。
    """
    if not html_text:
        return []
    parser = _DuckDuckGoParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - 脏 HTML 不应影响搜索
        logger.warning("解析 DuckDuckGo 页面失败：%s", exc)
        return []
    # 去掉标题/链接重复项，保持页面顺序
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in parser.results:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# 远程搜索：SerpAPI / DuckDuckGo
# ---------------------------------------------------------------------------


def _http_get(url: str, params: dict[str, Any]) -> str:
    """发起 GET 请求并返回响应文本；失败抛异常由调用方捕获。"""
    if requests is None:
        raise RuntimeError(f"requests 不可用：{_REQUESTS_IMPORT_ERROR or '未安装'}")
    response = requests.get(
        url,
        params=params,
        headers=_BROWSER_HEADERS,
        timeout=float(config.WEB_SEARCH_TIMEOUT),
    )
    # 非 2xx 视为远程失败，交给上层继续尝试下一个 provider/页面
    response.raise_for_status()
    # DuckDuckGo 页面是 UTF-8，但响应头不一定带 charset，requests 可能误判为 latin-1，
    # 这里显式指定，避免中文标题/摘要乱码。
    response.encoding = "utf-8"
    return response.text


def _search_with_serpapi(query: str, top_k: int) -> list[dict[str, Any]]:
    """调用 SerpAPI（Google 引擎）并转换为标准结果结构。

    Args:
        query: 搜索词。
        top_k: 最多返回条数。

    Returns:
        标准结果列表；失败时抛异常。
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": config.SERPAPI_API_KEY,
        "hl": "zh-cn",
        "num": max(1, int(top_k)),
    }
    if requests is None:
        raise RuntimeError(f"requests 不可用：{_REQUESTS_IMPORT_ERROR or '未安装'}")
    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=float(config.WEB_SEARCH_TIMEOUT),
    )
    response.raise_for_status()
    payload = response.json()
    results: list[dict[str, Any]] = []
    for item in payload.get("organic_results") or []:
        title = _clean_text(item.get("title"))
        url = _clean_text(item.get("link"))
        if not title or not url:
            continue
        results.append(
            {
                "title": title,
                "snippet": _truncate(item.get("snippet") or ""),
                "url": url,
                "source": REMOTE_SOURCE,
                "provider": "serpapi",
            }
        )
        if len(results) >= top_k:
            break
    return results


def _search_with_duckduckgo(query: str, top_k: int) -> list[dict[str, Any]]:
    """抓取 DuckDuckGo html/lite 页面并解析结果。

    依次尝试 html、lite 两个地址，任一页面解析出结果即返回；
    两个都失败时抛出最后一个异常，供上层记录 last_error。

    Args:
        query: 搜索词。
        top_k: 最多返回条数。

    Returns:
        标准结果列表；失败时抛异常。
    """
    errors: list[str] = []
    for page_url in (DUCKDUCKGO_HTML_URL, DUCKDUCKGO_LITE_URL):
        try:
            html_text = _http_get(page_url, {"q": query})
            parsed = _parse_duckduckgo(html_text)
            if parsed:
                results: list[dict[str, Any]] = []
                for item in parsed:
                    if len(results) >= top_k:
                        break
                    results.append(
                        {
                            "title": _truncate(item["title"], 200),
                            "snippet": _truncate(item["snippet"]),
                            "url": item["url"],
                            "source": REMOTE_SOURCE,
                            "provider": "duckduckgo",
                        }
                    )
                # 防御：DuckDuckGo 偶尔返回「没有结果」的页面但解析出导航链接
                return results
            errors.append(f"{page_url} 未解析出结果")
        except Exception as exc:  # noqa: BLE001 - 换下一个页面继续尝试
            errors.append(f"{page_url}: {exc}")
            logger.warning("DuckDuckGo 页面抓取失败（%s）：%s", page_url, exc)
            # 连接类错误（网络不可达 / DNS 失败 / 超时）换页面也没用，直接结束，
            # 避免再等一次超时（本机网络受限时能让首次失败从 2×timeout 降到 1×timeout）。
            if requests is not None and isinstance(
                exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
            ):
                break
    raise RuntimeError("; ".join(errors) or "DuckDuckGo 搜索失败")


# ---------------------------------------------------------------------------
# Bing HTML 解析（cn.bing.com；DuckDuckGo 不可达时的备用真实搜索源）
# ---------------------------------------------------------------------------
_BING_RESULT_BLOCK_RE = re.compile(r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>', re.S)
_BING_TITLE_RE = re.compile(r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_BING_SNIPPET_RE = re.compile(r'<p[^>]*>(.*?)</p>', re.S)


def _strip_tags(text: str | None) -> str:
    """去掉 HTML 标签并反转义实体，得到纯文本。"""
    if not text:
        return ""
    plain = re.sub(r"<[^>]+>", "", str(text))
    return _clean_text(_html.unescape(plain))


def _parse_bing(html_text: str) -> list[dict[str, str]]:
    """从 Bing 搜索结果页 HTML 中提取标题/摘要/链接。

    结果结构：每个结果是一个 ``<li class="b_algo">``，标题在
    ``<h2><a href="...">``，摘要在 ``<p>``。解析失败返回空列表，不抛异常。
    """
    results: list[dict[str, str]] = []
    if not html_text:
        return results
    for block in _BING_RESULT_BLOCK_RE.findall(html_text):
        title_match = _BING_TITLE_RE.search(block)
        if not title_match:
            continue
        url = _html.unescape(title_match.group(1) or "").strip()
        title = _strip_tags(title_match.group(2))
        if not title or not url.startswith(("http://", "https://")):
            continue
        snippet_match = _BING_SNIPPET_RE.search(block)
        snippet = _strip_tags(snippet_match.group(1)) if snippet_match else ""
        results.append({"title": title, "snippet": snippet, "url": url})
    return results


def _search_with_bing(query: str, top_k: int) -> list[dict[str, Any]]:
    """抓取 cn.bing.com 搜索结果并转换为标准结果结构。

    DuckDuckGo 在部分网络环境不可达（连接超时），Bing 国内站通常可直连，
    因此作为备用真实联网搜索源；解析不到结果时抛异常由上层继续兜底。
    """
    html_text = _http_get(
        BING_SEARCH_URL,
        {"q": query, "setlang": "zh-CN", "count": str(max(1, int(top_k)))},
    )
    parsed = _parse_bing(html_text)
    if not parsed:
        raise RuntimeError("Bing 未解析出搜索结果（页面结构可能变化或被风控）")
    results: list[dict[str, Any]] = []
    for item in parsed:
        if len(results) >= top_k:
            break
        results.append(
            {
                "title": _truncate(item["title"], 200),
                "snippet": _truncate(item["snippet"]),
                "url": item["url"],
                "source": REMOTE_SOURCE,
                "provider": "bing",
            }
        )
    return results


def _normalize_provider() -> str:
    """把配置里的 provider 名称归一化为 ``serpapi`` / ``duckduckgo`` / ``bing`` / ``none``。"""
    raw = str(getattr(config, "WEB_SEARCH_PROVIDER", "") or "").strip().lower()
    if raw in ("", "auto"):
        return "duckduckgo"
    if raw in ("serpapi", "google", "google_serpapi", "google-search"):
        return "serpapi"
    if raw in ("duckduckgo", "ddg", "duck", "html"):
        return "duckduckgo"
    if raw in ("bing", "cn.bing", "cnbing", "bing-cn", "bingcn"):
        return "bing"
    if raw in ("none", "off", "disabled", "local"):
        return "none"
    # 未知 provider：默认按 duckduckgo 处理，保证仍可能拿到远程结果
    logger.warning("未知的 WEB_SEARCH_PROVIDER=%r，按 duckduckgo 处理", raw)
    return "duckduckgo"


def _active_provider() -> str:
    """返回当前实际生效的远程搜索提供方（关闭时为 ``none``）。"""
    if not bool(getattr(config, "WEB_SEARCH_ENABLED", False)):
        return "none"
    provider = _normalize_provider()
    if provider == "none":
        return "none"
    # 配置了 serpapi 但没填 Key：自动回落到 DuckDuckGo，而不是直接离线
    if provider == "serpapi" and not str(getattr(config, "SERPAPI_API_KEY", "") or "").strip():
        return "duckduckgo"
    return provider


def _remote_in_cooldown() -> bool:
    """远程搜索是否处于失败熔断冷却期。"""
    return time.monotonic() < float(_status.get("cooldown_until") or 0.0)


def _search_remote(query: str, top_k: int) -> list[dict[str, Any]]:
    """按当前配置执行远程搜索。

    候选源顺序：显式配置的 provider 优先；单一来源失败时自动尝试另一个
    真实来源，避免某个搜索站被墙/风控时整个联网搜索直接失效。

    Args:
        query: 搜索词。
        top_k: 最多返回条数。

    Returns:
        远程结果列表；不可用或全部失败时抛异常。
    """
    provider = _active_provider()
    if provider == "none":
        raise RuntimeError("联网搜索未启用")

    if provider == "serpapi":
        candidates = [("serpapi", _search_with_serpapi), ("bing", _search_with_bing)]
    elif provider == "bing":
        candidates = [("bing", _search_with_bing), ("duckduckgo", _search_with_duckduckgo)]
    else:
        candidates = [("duckduckgo", _search_with_duckduckgo), ("bing", _search_with_bing)]

    errors: list[str] = []
    for name, func in candidates:
        try:
            results = func(query, top_k)
            if results:
                logger.info("远程搜索成功：source=%s provider=%s 条数=%s", name, name, len(results))
                return results
            errors.append(f"{name}: 无结果")
        except Exception as exc:  # noqa: BLE001 - 继续尝试下一个搜索源
            errors.append(f"{name}: {exc}")
            logger.warning("远程搜索源不可用（%s）：%s", name, exc)
    raise RuntimeError("; ".join(errors) or "所有远程搜索源均不可用")


# ---------------------------------------------------------------------------
# 本地知识库兜底（SQLite career_info）
# ---------------------------------------------------------------------------
# career_info 表由 database/schema.sql 定义，字段可能调整，因此这里用
# PRAGMA table_info 动态读取列名，并按常见命名做优先级选择，避免硬编码列名。
_TITLE_COLUMNS = ("title", "job_title", "position", "name", "career", "job", "role", "name_cn")
_SNIPPET_COLUMNS = (
    "description",
    "detail",
    "content",
    "summary",
    "requirements",
    "skills",
    "responsibility",
    "intro",
)
_SALARY_COLUMNS = ("salary", "salary_range", "salary_min", "salary_max", "pay")
_URL_COLUMNS = ("url", "link", "source_url")

_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def _query_tokens(query: str) -> list[str]:
    """把搜索词切成用于本地匹配的 token（英文按词、中文按整段 + 二元组）。"""
    text = _clean_text(query).lower()
    tokens: list[str] = []
    if not text:
        return tokens
    for part in re.split(r"[\s,，。、;；:：!！?？/\\|（）()\[\]【】\"'“”‘’]+", text):
        if not part:
            continue
        if _CJK_RUN_RE.search(part):
            tokens.append(part)
            if len(part) > 2:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        elif len(part) >= 2:
            tokens.append(part)
    # 去重并保序
    return list(dict.fromkeys(token for token in tokens if token))


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """按优先级从实际列名中挑选一个（大小写不敏感），找不到返回 None。"""
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _row_to_search_item(
    row: sqlite3.Row, columns: list[str], provider: str = "local"
) -> dict[str, Any]:
    """把 career_info 的一行转换成标准搜索结果结构。

    Args:
        row: sqlite3.Row 数据行。
        columns: 表的列名列表。
        provider: 结果提供方标记。

    Returns:
        ``{"title","snippet","url","source","provider"}``
    """
    title_col = _pick_column(columns, _TITLE_COLUMNS)
    title = _clean_text(row[title_col]) if title_col else ""
    if not title:
        # 没有标题列时退化为「职业信息」+ 第一个文本列
        title = "职业信息"

    snippet_parts: list[str] = []
    snippet_col = _pick_column(columns, _SNIPPET_COLUMNS)
    if snippet_col and row[snippet_col]:
        snippet_parts.append(_clean_text(row[snippet_col]))

    # 薪资：优先 *_min / *_max 组合，其次 salary/salary_range 单列
    salary_min_col = _pick_column(columns, ("salary_min", "min_salary"))
    salary_max_col = _pick_column(columns, ("salary_max", "max_salary"))
    salary_value = ""
    if salary_min_col and salary_max_col and row[salary_min_col] and row[salary_max_col]:
        salary_value = f"{row[salary_min_col]}-{row[salary_max_col]}"
    else:
        salary_col = _pick_column(columns, ("salary", "salary_range", "pay"))
        if salary_col and row[salary_col]:
            salary_value = _clean_text(row[salary_col])
    if salary_value:
        snippet_parts.append(f"薪资：{salary_value}")

    # 行业 / 经验 / 学历 等补充字段，存在就拼进摘要
    for extra_name in ("industry", "experience", "education", "demand", "city"):
        extra_col = _pick_column(columns, (extra_name,))
        if extra_col and row[extra_col]:
            snippet_parts.append(f"{extra_name}: {_clean_text(row[extra_col])}")

    snippet = "；".join(part for part in snippet_parts if part) or _clean_text(
        " ".join(str(row[column]) for column in columns if row[column] not in (None, ""))
    )

    url_col = _pick_column(columns, _URL_COLUMNS)
    url = _clean_text(row[url_col]) if url_col else ""
    return {
        "title": title,
        "snippet": _truncate(snippet),
        "url": url,
        "source": LOCAL_SOURCE,
        "provider": provider,
    }


def _local_fallback(query: str, top_k: int) -> list[dict[str, Any]]:
    """读取 SQLite ``career_info`` 表作为本地知识库兜底。

    评分方式：把搜索词切成 token，命中行内任意文本的 token 越多分越高；
    全部 0 分时按库内顺序返回前 top_k 条（离线也给出可用职业信息）。

    Args:
        query: 搜索词。
        top_k: 最多返回条数。

    Returns:
        标准结果列表；数据库不存在/表不存在/任何异常时返回 ``[]``。
    """
    db_path = Path(getattr(config, "SQLITE_PATH", ""))
    if not db_path or not db_path.exists():
        logger.warning("本地知识库兜底不可用：数据库文件不存在（%s）", db_path)
        return []

    connection: sqlite3.Connection | None = None
    try:
        uri = db_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=3.0)
        connection.row_factory = sqlite3.Row
        table_check = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='career_info'"
        ).fetchone()
        if table_check is None:
            logger.warning("本地知识库兜底不可用：career_info 表不存在")
            return []
        columns = [str(row["name"]) for row in connection.execute("PRAGMA table_info(career_info)")]
        if not columns:
            return []
        rows = connection.execute("SELECT * FROM career_info").fetchall()
        if not rows:
            logger.warning("本地知识库兜底：career_info 表为空")
            return []

        tokens = _query_tokens(query)
        scored: list[tuple[int, int, sqlite3.Row]] = []
        for index, row in enumerate(rows):
            haystack = " ".join(
                str(row[column]) for column in columns if row[column] not in (None, "")
            ).lower()
            score = sum(1 for token in tokens if token and token in haystack)
            scored.append((score, index, row))
        # 先按命中数降序，再按原顺序，保证离线结果稳定可测
        scored.sort(key=lambda item: (-item[0], item[1]))

        results: list[dict[str, Any]] = [
            _row_to_search_item(row, columns)
            for _score, _index, row in scored[: max(1, int(top_k))]
        ]
        logger.warning(
            "联网搜索不可用，已降级本地职业知识库：query=%r 命中 %s 条（最高分 %s）",
            query[:50],
            len(results),
            scored[0][0] if scored else 0,
        )
        return results
    except Exception as exc:  # noqa: BLE001 - 兜底路径绝不抛异常
        logger.warning("读取本地职业知识库失败：%s", exc)
        return []
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - 关闭失败无需处理
                pass


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------


def web_search(query: str, top_k: int = 5) -> list[dict]:
    """联网搜索，失败或关闭时自动降级到本地职业知识库。

    Args:
        query: 搜索关键词（用户问题或检索词）。
        top_k: 最多返回条数，默认 5。

    Returns:
        ``[{"title": str, "snippet": str, "url": str, "source": str, "provider": str}]``；
        远程失败时 ``source="本地知识库兜底"``、``provider="local"``；
        仍然没有结果返回 ``[]``。任何情况都不抛异常。
    """
    _status["calls"] += 1
    text = _clean_text(query)
    if not text:
        _status["last_error"] = "搜索词为空"
        return []

    try:
        limit = int(top_k) if top_k is not None else int(config.WEB_SEARCH_TOP_K)
    except (TypeError, ValueError):
        limit = int(config.WEB_SEARCH_TOP_K)
    limit = max(1, min(limit, 20))

    # 1) 远程搜索（只有启用且 provider 有效时才尝试；失败后会进入熔断冷却）
    if bool(getattr(config, "WEB_SEARCH_ENABLED", False)) and _active_provider() != "none":
        if _remote_in_cooldown():
            remaining = float(_status.get("cooldown_until") or 0.0) - time.monotonic()
            logger.info("联网搜索处于熔断冷却期（剩余 %.0f 秒），直接使用本地知识库兜底", max(0.0, remaining))
        else:
            try:
                results = _search_remote(text, limit)
                if results:
                    _status["last_error"] = None
                    _status["cooldown_until"] = 0.0
                    logger.info("联网搜索成功：query=%r provider=%s 条数=%s", text[:50], _active_provider(), len(results))
                    return results[:limit]
                _status["last_error"] = "远程搜索未返回结果"
                _status["cooldown_until"] = time.monotonic() + REMOTE_COOLDOWN_SECONDS
                logger.warning("联网搜索未返回结果，进入 %d 秒熔断冷却，降级本地知识库：query=%r", int(REMOTE_COOLDOWN_SECONDS), text[:50])
            except Exception as exc:  # noqa: BLE001 - 搜索失败不抛给上层
                _status["last_error"] = str(exc)
                _status["cooldown_until"] = time.monotonic() + REMOTE_COOLDOWN_SECONDS
                logger.warning(
                    "联网搜索失败，进入 %d 秒熔断冷却，降级本地知识库：%s",
                    int(REMOTE_COOLDOWN_SECONDS),
                    exc,
                )
    else:
        # 关闭搜索属于正常配置，不记录成 error
        _status["last_error"] = None
        logger.warning("联网搜索已关闭（WEB_SEARCH_ENABLED=false），使用本地知识库兜底：query=%r", text[:50])

    # 2) 本地知识库兜底
    _status["fallback_used"] = True
    try:
        return _local_fallback(text, limit)[:limit]
    except Exception as exc:  # noqa: BLE001 - 双保险：兜底再失败也返回 []
        logger.warning("本地知识库兜底异常：%s", exc)
        return []


def is_search_available() -> bool:
    """联网搜索此刻是否可用（已启用、provider 有效且 requests 可导入）。

    Returns:
        True 表示会尝试远程搜索；False 表示 web_search 直接走本地兜底。
    """
    if not bool(getattr(config, "WEB_SEARCH_ENABLED", False)):
        return False
    if _active_provider() == "none":
        return False
    if _remote_in_cooldown():
        return False
    return requests is not None


def get_search_status() -> dict:
    """返回搜索模块运行状态，供健康检查/前端展示。

    Returns:
        ``{"enabled": bool, "provider": str, "last_error": str | None,
        "fallback_used": bool, "calls": int, "cooldown_remaining": float}``
    """
    cooldown_until = float(_status.get("cooldown_until") or 0.0)
    return {
        "enabled": bool(getattr(config, "WEB_SEARCH_ENABLED", False)),
        "provider": _active_provider(),
        "last_error": _status.get("last_error"),
        "fallback_used": bool(_status.get("fallback_used")),
        "calls": int(_status.get("calls") or 0),
        "cooldown_remaining": round(max(0.0, cooldown_until - time.monotonic()), 1),
    }


def reset_search_status() -> None:
    """重置搜索状态（测试用，避免用例之间互相影响）。"""
    _status["calls"] = 0
    _status["last_error"] = None
    _status["fallback_used"] = False
    _status["cooldown_until"] = 0.0
