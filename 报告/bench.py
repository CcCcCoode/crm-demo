"""并发压测：/chat（规则模式）、/health，统计吞吐与分位延迟。"""
import json
import statistics
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:5100"
PHONE = "13900009999"


def post_chat(query: str) -> float:
    body = json.dumps({"phone_number": PHONE, "query": query, "stream": False}).encode()
    req = urllib.request.Request(
        BASE + "/chat", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()
    return time.perf_counter() - t0


def get_health() -> float:
    t0 = time.perf_counter()
    with urllib.request.urlopen(BASE + "/health", timeout=60) as resp:
        resp.read()
    return time.perf_counter() - t0


def run(concurrency: int, total: int, fn) -> dict:
    lat = []
    lock = threading.Lock()
    per = total // concurrency

    def worker():
        local = []
        for _ in range(per):
            local.append(fn())
        with lock:
            lat.extend(local)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    lat.sort()
    return {
        "concurrency": concurrency,
        "requests": len(lat),
        "wall": round(wall, 2),
        "qps": round(len(lat) / wall, 2),
        "p50_ms": round(statistics.median(lat) * 1000, 1),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1] * 1000, 1),
        "max_ms": round(lat[-1] * 1000, 1),
    }


if __name__ == "__main__":
    for i in range(30):  # 等服务就绪
        try:
            get_health()
            break
        except Exception:
            time.sleep(1)

    post_chat("你们有哪些 Python 课程？多少钱？")  # 预热（含建用户 + 一次向量检索）
    time.sleep(0.5)

    print("== /chat（无 Key 规则模式，hash 嵌入）==")
    for c in (1, 2, 4, 8, 16):
        print(run(c, max(c * 2, 4), lambda: post_chat("你们有哪些 Python 课程？多少钱？")))

    print("== /health ==")
    for c in (4, 16):
        print(run(c, c * 4, get_health))
