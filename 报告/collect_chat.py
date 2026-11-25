"""采集 /chat 的真实响应（非流式摘要 + 流式原始帧）。"""
import json
import urllib.request

BASE = "http://127.0.0.1:5099"
PHONE = "13700002222"


def chat(payload):
    req = urllib.request.Request(
        BASE + "/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=180)


resp = chat({"phone_number": PHONE, "query": "Python 数据分析课程学多久，多少钱？", "stream": False})
body = json.loads(resp.read())
print("status:", resp.status)
print("keys:", sorted(body))
print("intent:", body.get("intent_code"), body.get("intent_name"), "mode:", body.get("mode"),
      "rounds:", body.get("rounds"), "elapsed_ms:", body.get("elapsed_ms"))
print("tools:", [t.get("name") for t in body.get("tool_calls", [])])
print("user_info:", json.dumps(body.get("user_info"), ensure_ascii=False))
print("reply_head:", (body.get("reply") or "")[:300].replace("\n", "\\n"))
print()

resp = chat({"phone_number": PHONE, "query": "AI 应用开发课程多少钱？", "stream": True})
print("stream status:", resp.status, "| content-type:", resp.headers.get("Content-Type"))
lines = []
for raw in resp:
    line = raw.decode("utf-8").rstrip("\n")
    lines.append(line)
    if len(lines) >= 9:
        break
resp.close()
for line in lines:
    print(repr(line[:260]))
