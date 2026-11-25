"""采集真实接口返回，用于报告中的接口示例。"""
import json
import urllib.request

BASE = "http://127.0.0.1:5099"


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def show(title, status, body, limit=700):
    text = json.dumps(body, ensure_ascii=False)
    print(f"### {title} -> HTTP {status}")
    print(text[:limit])
    print()


status, health = call("GET", "/health")
show("GET /health", status, health, 900)

status, body = call("GET", "/vectorstore/list")
show("GET /vectorstore/list", status, body)

status, body = call("POST", "/vectorstore/add", {
    "collection": "course",
    "documents": ["Python 自动化办公训练营：3980 元，8 周，适合职场白领用脚本替代重复劳动。"],
    "metadatas": [{"name": "Python 自动化办公训练营", "price": 3980, "source": "report-demo"}],
    "ids": ["report-demo-001"],
})
show("POST /vectorstore/add", status, body)
new_id = (body.get("ids") or ["report-demo-001"])[0]

status, body = call("POST", "/vectorstore/update", {
    "collection": "course",
    "ids": [new_id],
    "documents": ["Python 自动化办公训练营：3980 元，8 周，Excel 批处理/邮件自动化/报表生成，适合行政与财务。"],
})
show("POST /vectorstore/update", status, body)

status, body = call("POST", "/vectorstore/search", {
    "collection": "course", "query": "办公自动化 多少钱", "top_k": 2,
})
b = dict(body)
if b.get("results"):
    b["results"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in b["results"]]
show("POST /vectorstore/search", status, b, 900)

status, body = call("POST", "/vectorstore/delete", {"collection": "course", "ids": [new_id]})
show("POST /vectorstore/delete", status, body)

status, body = call("GET", "/database/schema")
show("GET /database/schema", status, {"ok": body.get("ok"), "tables": body.get("tables"),
                                     "schema_head": (body.get("schema") or "")[:120]})

status, body = call("POST", "/database/query", {"question": "薪资最高的三个职业是什么"})
rows = body.get("rows") or []
body = dict(body)
body["rows"] = rows[:3]
show("POST /database/query", status, body, 1000)
