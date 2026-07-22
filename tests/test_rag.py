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


# ── Chunker unit tests ────────────────────────────────────

def test_chunker_short_text_produces_single_chunk():
    from app.chunker import chunk_document
    chunks = chunk_document("doc1", "Hello world", {"title": "test"})
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc1_chunk_0"
    assert chunks[0].text == "Hello world"
    assert chunks[0].doc_id == "doc1"


def test_chunker_empty_text_produces_no_chunks():
    from app.chunker import chunk_document
    assert chunk_document("doc1", "   ", {}) == []


def test_chunker_preserves_metadata_fields():
    from app.chunker import chunk_document
    chunks = chunk_document("doc1", "Some text", {"title": "foo"})
    assert chunks[0].metadata["title"] == "foo"
    assert chunks[0].metadata["doc_id"] == "doc1"
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunker_long_text_produces_multiple_chunks():
    from app.chunker import chunk_document
    text = "A" * 600
    chunks = chunk_document("doc1", text, {}, chunk_size=512, overlap=64)
    assert len(chunks) == 2


def test_chunker_chunk_ids_are_sequential():
    from app.chunker import chunk_document
    chunks = chunk_document("doc1", "x" * 2000, {}, chunk_size=512, overlap=64)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"doc1_chunk_{i}"


def test_chunker_adjacent_chunks_share_overlap_content():
    from app.chunker import chunk_document
    # chunk[0] covers [0:512], chunk[1] covers [448:600]
    # so text[448:512] must be a suffix of chunk[0] and prefix of chunk[1]
    text = "A" * 600
    chunks = chunk_document("doc1", text, {}, chunk_size=512, overlap=64)
    overlap_region = text[448:512]
    assert chunks[0].text.endswith(overlap_region)
    assert chunks[1].text.startswith(overlap_region)


def test_chunker_custom_chunk_size_and_overlap():
    from app.chunker import chunk_document
    text = "word " * 40  # ~200 chars
    chunks = chunk_document("doc1", text, {}, chunk_size=100, overlap=20)
    # step = 100-20 = 80; text len ~200, so we expect 3 chunks
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= 100


# ── Reranker unit tests ───────────────────────────────────

def test_rerank_empty_candidates_returns_empty():
    from app.reranker import rerank
    assert rerank("any query", [], top_k=5) == []


def test_rerank_sorts_candidates_by_score_descending():
    from unittest.mock import patch
    import numpy as np
    from app.reranker import rerank

    candidates = [
        {"chunk_id": "a", "text": "low relevance"},
        {"chunk_id": "b", "text": "high relevance"},
        {"chunk_id": "c", "text": "medium relevance"},
    ]
    with patch("app.reranker._get_model") as mock_get_model:
        mock_get_model.return_value.predict.return_value = np.array([0.1, 0.9, 0.5])
        result = rerank("query", candidates, top_k=3)

    assert result[0]["chunk_id"] == "b"
    assert result[1]["chunk_id"] == "c"
    assert result[2]["chunk_id"] == "a"


def test_rerank_respects_top_k():
    from unittest.mock import patch
    import numpy as np
    from app.reranker import rerank

    candidates = [{"chunk_id": str(i), "text": f"text {i}"} for i in range(5)]
    with patch("app.reranker._get_model") as mock_get_model:
        mock_get_model.return_value.predict.return_value = np.array([0.5, 0.1, 0.9, 0.3, 0.7])
        result = rerank("query", candidates, top_k=2)

    assert len(result) == 2
    assert result[0]["chunk_id"] == "2"  # score 0.9
    assert result[1]["chunk_id"] == "4"  # score 0.7


def test_rerank_adds_rerank_score_field():
    from unittest.mock import patch
    import numpy as np
    from app.reranker import rerank

    candidates = [{"chunk_id": "x", "text": "some text"}]
    with patch("app.reranker._get_model") as mock_get_model:
        mock_get_model.return_value.predict.return_value = np.array([0.75])
        result = rerank("query", candidates, top_k=1)

    assert "rerank_score" in result[0]
    assert result[0]["rerank_score"] == 0.75


def test_rerank_top_k_larger_than_candidates_returns_all():
    from unittest.mock import patch
    import numpy as np
    from app.reranker import rerank

    candidates = [{"chunk_id": "a", "text": "only one"}]
    with patch("app.reranker._get_model") as mock_get_model:
        mock_get_model.return_value.predict.return_value = np.array([0.6])
        result = rerank("query", candidates, top_k=10)

    assert len(result) == 1
