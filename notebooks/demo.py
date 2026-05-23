"""
End-to-end RAG pipeline demo.
Run: make ingest then make serve then make demo
"""
from __future__ import annotations
import requests, json

BASE = "http://localhost:8082"

def post(path, payload): return requests.post(f"{BASE}{path}", json=payload).json()
def get(path):           return requests.get(f"{BASE}{path}").json()

print("\n=== RAG Pipeline Demo ===\n")

# 1. Health
health = get("/health")
print(f"1. Health check:")
print(f"   chunks indexed: {health['chunks_indexed']}")
print(f"   model loaded:   {health['model_loaded']}")

# 2. Index stats
stats = get("/index/stats")
print(f"\n2. Index stats:")
print(f"   {json.dumps(stats, indent=4)}")

# 3. Run queries
queries = [
    "What is supervised learning?",
    "How does gradient descent work?",
    "What is overfitting and how do you prevent it?",
    "What is transfer learning?",
]

print("\n3. Running queries...\n")
chunk_ids_for_eval = []

for q in queries:
    r = post("/query", {"query": q, "top_k": 3})
    print(f"Q: {q}")
    print(f"A: {r['answer'][:300]}...")
    print(f"   faithfulness={r['eval']['faithfulness']}  relevance={r['eval']['answer_relevance']}")
    if r.get("chunks"):
        chunk_ids_for_eval.append(r["chunks"][0]["chunk_id"])
    print()

# 4. Eval with ground truth
print("4. Eval query with ground truth retrieval metrics...")
r = post("/query/eval", {
    "query": "What is overfitting?",
    "relevant_doc_ids": chunk_ids_for_eval[:2],
    "top_k": 3,
})
print(f"   hit_rate={r['eval']['hit_rate']}")
print(f"   mrr={r['eval']['mrr']}")
print(f"   faithfulness={r['eval']['faithfulness']}")
print(f"   answer_relevance={r['eval']['answer_relevance']}")

# 5. Summary
print("\n5. Eval summary across all queries:")
summary = get("/eval/summary")
print(f"   {json.dumps(summary, indent=4)}")

print(f"\nAPI docs → http://localhost:8082/docs")
print("\nDone.")
