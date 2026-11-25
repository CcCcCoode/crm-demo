"""采集 tool_calls 与 done 帧的真实结构。"""
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


body = json.loads(chat({"phone_number": PHONE, "query": "数据分析课程适合零基础吗", "stream": False}).read())
print("tool_call[0]:", json.dumps(body["tool_calls"][0], ensure_ascii=False)[:400])
print("is_new:", body["is_new"], "rounds:", body["rounds"], "mode:", body["mode"])

resp = chat({"phone_number": PHONE, "query": "AI 应用开发课程多少钱", "stream": True})
done = None
for raw in resp:
    line = raw.decode("utf-8").strip()
    if line.startswith("data: ") and '"type": "done"' in line:
        done = json.loads(line[6:])
        break
resp.close()
if done:
    done["reply"] = done["reply"][:60] + "…（截断）"
    print("done frame:", json.dumps(done, ensure_ascii=False)[:900])
