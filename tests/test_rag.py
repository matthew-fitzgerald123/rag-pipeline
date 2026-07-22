from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, ".")

from scripts.ingest import *  # noqa

from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["chunks_indexed"] > 0

def test_index_stats():
    r = client.get("/index/stats")
    assert r.status_code == 200
    assert r.json()["total_chunks"] > 0

def test_query_returns_answer():
    r = client.post("/query", json={
        "query": "What is supervised learning?",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "chunks" in data
    assert len(data["chunks"]) <= 3
    assert "eval" in data
    assert "faithfulness" in data["eval"]

def test_query_eval_with_ground_truth():
    r = client.post("/query", json={"query": "overfitting", "top_k": 1})
    chunk_id = r.json()["chunks"][0]["chunk_id"]

    r = client.post("/query/eval", json={
        "query": "What is overfitting?",
        "relevant_doc_ids": [chunk_id],
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert "hit_rate" in data["eval"]
    assert "mrr" in data["eval"]
    assert "ndcg" in data["eval"]
    assert 0.0 <= data["eval"]["hit_rate"] <= 1.0
    assert 0.0 <= data["eval"]["mrr"] <= 1.0
    assert 0.0 <= data["eval"]["ndcg"] <= 1.0


def test_ndcg_perfect_ranking():
    from app.evaluator import ndcg_at_k
    # All relevant docs at the very top gives the ideal ranking.
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0


def test_ndcg_rewards_higher_ranks():
    from app.evaluator import ndcg_at_k
    top = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
    bottom = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
    assert top == 1.0
    assert bottom < top
    assert bottom > 0.0


def test_ndcg_no_relevant_retrieved_is_zero():
    from app.evaluator import ndcg_at_k
    assert ndcg_at_k(["x", "y", "z"], ["a"], k=3) == 0.0


def test_ndcg_empty_relevant_is_zero():
    from app.evaluator import ndcg_at_k
    assert ndcg_at_k(["a", "b"], [], k=3) == 0.0


def test_ndcg_respects_k_cutoff():
    from app.evaluator import ndcg_at_k
    # The single relevant doc sits at rank 3, outside k=2, so it cannot contribute.
    assert ndcg_at_k(["x", "y", "a"], ["a"], k=2) == 0.0

def test_eval_summary():
    r = client.get("/eval/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_queries" in data
    assert data["total_queries"] > 0

def test_eval_summary_includes_avg_ndcg():
    # Drive a /query/eval request so ndcg is logged, then confirm summary surfaces it.
    r_q = client.post("/query", json={"query": "overfitting", "top_k": 1})
    chunk_id = r_q.json()["chunks"][0]["chunk_id"]
    client.post("/query/eval", json={
        "query": "What is overfitting?",
        "relevant_doc_ids": [chunk_id],
        "top_k": 3,
    })
    r = client.get("/eval/summary")
    assert r.status_code == 200
    data = r.json()
    assert "avg_ndcg" in data
    assert data["avg_ndcg"] is not None

def test_eval_history():
    r = client.get("/eval/history?limit=5")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_eval_history_includes_ndcg_field():
    r = client.get("/eval/history?limit=5")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    for row in rows:
        assert "ndcg" in row

# ── Evaluator unit tests ──────────────────────────────────

def test_hit_rate_all_relevant_retrieved():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b", "c"], ["a", "b"]) == 1.0

def test_hit_rate_partial():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "x", "y"], ["a", "b"]) == 0.5

def test_hit_rate_none_retrieved():
    from app.evaluator import hit_rate
    assert hit_rate(["x", "y", "z"], ["a", "b"]) == 0.0

def test_hit_rate_empty_relevant_is_zero():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b"], []) == 0.0


def test_mrr_found_at_rank_one():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

def test_mrr_found_at_rank_two():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["x", "a", "c"], ["a"]) == 0.5

def test_mrr_not_found():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

def test_mrr_empty_relevant_is_zero():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["a", "b"], []) == 0.0


def test_faithfulness_fully_supported():
    from app.evaluator import faithfulness
    chunks = [{"text": "Supervised learning trains models on labeled data examples."}]
    answer = "Supervised learning trains models on labeled data."
    assert faithfulness(answer, chunks) == 1.0

def test_faithfulness_not_supported():
    from app.evaluator import faithfulness
    chunks = [{"text": "Deep learning uses neural networks."}]
    answer = "Quantum mechanics leverages superposition states."
    assert faithfulness(answer, chunks) == 0.0

def test_faithfulness_empty_answer_is_zero():
    from app.evaluator import faithfulness
    assert faithfulness("", [{"text": "some context text here"}]) == 0.0


def test_answer_relevance_full_overlap():
    from app.evaluator import answer_relevance
    assert answer_relevance("overfitting regularization", "overfitting regularization prevents model overfitting") == 1.0

def test_answer_relevance_no_overlap():
    from app.evaluator import answer_relevance
    assert answer_relevance("overfitting regularization", "cats dogs unrelated content") == 0.0

def test_answer_relevance_stopwords_only_query_is_zero():
    from app.evaluator import answer_relevance
    assert answer_relevance("the a an", "some answer text here") == 0.0


# ── /query/stream SSE integration test ───────────────────

def test_query_stream_returns_sse():
    r = client.post("/query/stream", json={
        "query": "What is supervised learning?",
        "top_k": 3,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.content.decode()
    assert "data:" in body
    assert "[DONE]" in body

def test_query_stream_events_are_valid_json_or_done():
    r = client.post("/query/stream", json={
        "query": "What is overfitting?",
        "top_k": 3,
    })
    assert r.status_code == 200
    import json
    for line in r.content.decode().splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        parsed = json.loads(payload)
        assert "token" in parsed


def test_empty_query_still_returns():
    r = client.post("/query", json={"query": "xyzzy nonsense query 12345", "top_k": 3})
    assert r.status_code in [200, 404]


def test_query_returns_citations():
    r = client.post("/query", json={"query": "What is supervised learning?", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "citations" in data
    assert isinstance(data["citations"], list)
    for entry in data["citations"]:
        assert "sentence" in entry
        assert "citations" in entry


def test_citation_extraction_unit():
    from app.citations import extract_citations
    chunks = [
        {"chunk_id": "c1", "text": "Supervised learning trains on labeled data.", "metadata": {"title": "ML Basics"}},
        {"chunk_id": "c2", "text": "Neural networks are universal function approximators.", "metadata": {}},
    ]
    answer = "Supervised learning uses labeled training data."
    result = extract_citations(answer, chunks, threshold=0.2)
    assert len(result) >= 1
    assert result[0]["citations"][0]["chunk_id"] == "c1"
