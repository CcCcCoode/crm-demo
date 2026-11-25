"""测量真实 BGE 嵌入与向量检索的延迟，用于并发能力估算。"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config as cfg
from vectorstore import embeddings, store

t0 = time.perf_counter()
embeddings.embed_texts(["预热"])
print("BGE 加载+首条编码: %.2fs" % (time.perf_counter() - t0))
print(embeddings.get_embedding_info())

QUERIES = [
    "AI 应用开发工程师现在薪资多少",
    "你们有哪些 Python 课程？多少钱",
    "数据分析师需要会什么",
    "想转行学 AI，预算两万，怎么规划",
    "大模型算法工程师的要求",
] * 4

lat = []
for q in QUERIES:
    t = time.perf_counter()
    embeddings.embed_query(q)
    lat.append(time.perf_counter() - t)
print("embed_query: n=%d avg=%.1fms p50=%.1fms max=%.1fms"
      % (len(lat), statistics.mean(lat) * 1000, statistics.median(lat) * 1000, max(lat) * 1000))

slow = []
for q in QUERIES[:10]:
    t = time.perf_counter()
    store.search(cfg.COLLECTION_COURSE, q, top_k=3)
    slow.append(time.perf_counter() - t)
print("store.search(含嵌入): n=%d avg=%.1fms max=%.1fms"
      % (len(slow), statistics.mean(slow) * 1000, max(slow) * 1000))
