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


# ── Reranking integration tests ───────────────────────────

def test_query_with_rerank_enabled_returns_reranked_true():
    r = client.post("/query", json={
        "query": "What is supervised learning?",
        "top_k": 3,
        "rerank": True,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["reranked"] is True


def test_query_with_rerank_enabled_chunks_have_rerank_score():
    r = client.post("/query", json={
        "query": "What is overfitting?",
        "top_k": 3,
        "rerank": True,
    })
    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert len(chunks) > 0
    for chunk in chunks:
        assert "rerank_score" in chunk
        assert isinstance(chunk["rerank_score"], float)


def test_query_without_rerank_returns_reranked_false():
    r = client.post("/query", json={
        "query": "What is supervised learning?",
        "top_k": 3,
        "rerank": False,
    })
    assert r.status_code == 200
    assert r.json()["reranked"] is False


def test_query_eval_with_rerank_enabled():
    r = client.post("/query", json={"query": "overfitting", "top_k": 1})
    chunk_id = r.json()["chunks"][0]["chunk_id"]

    r = client.post("/query/eval", json={
        "query": "What is overfitting?",
        "relevant_doc_ids": [chunk_id],
        "top_k": 3,
        "rerank": True,
    })
    assert r.status_code == 200
    data = r.json()
    assert "hit_rate" in data["eval"]
    assert "mrr" in data["eval"]
    assert "ndcg" in data["eval"]


# ── Hybrid score fields ───────────────────────────────────

# ── answer_relevance persistence ─────────────────────────

def test_query_response_eval_includes_answer_relevance():
    r = client.post("/query", json={"query": "What is supervised learning?", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "answer_relevance" in data["eval"]
    assert isinstance(data["eval"]["answer_relevance"], float)
    assert 0.0 <= data["eval"]["answer_relevance"] <= 1.0


def test_eval_history_includes_answer_relevance_field():
    client.post("/query", json={"query": "What is gradient descent?", "top_k": 3})
    r = client.get("/eval/history?limit=5")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    for row in rows:
        assert "answer_relevance" in row


def test_eval_summary_avg_answer_relevance_uses_stored_value():
    client.post("/query", json={"query": "What is a neural network?", "top_k": 3})
    r = client.get("/eval/summary")
    assert r.status_code == 200
    data = r.json()
    assert "avg_answer_relevance" in data
    assert data["avg_answer_relevance"] is not None
    assert 0.0 <= data["avg_answer_relevance"] <= 1.0


def test_eval_query_persists_answer_relevance_in_history():
    r_q = client.post("/query", json={"query": "overfitting", "top_k": 1})
    chunk_id = r_q.json()["chunks"][0]["chunk_id"]
    client.post("/query/eval", json={
        "query": "What is overfitting?",
        "relevant_doc_ids": [chunk_id],
        "top_k": 3,
    })
    r = client.get("/eval/history?limit=3")
    assert r.status_code == 200
    rows = r.json()
    most_recent = rows[0]
    assert "answer_relevance" in most_recent
    assert most_recent["answer_relevance"] is not None


def test_query_chunks_expose_dense_and_bm25_scores():
    r = client.post("/query", json={
        "query": "What is gradient descent?",
        "top_k": 3,
    })
    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert len(chunks) > 0
    for chunk in chunks:
        assert "dense_score" in chunk, "chunk is missing dense_score"
        assert "bm25_score" in chunk, "chunk is missing bm25_score"
        assert "score" in chunk, "chunk is missing fused score"
        assert 0.0 <= chunk["dense_score"] <= 1.0
        assert 0.0 <= chunk["bm25_score"] <= 1.0


# ── hit_rate unit tests ───────────────────────────────────

def test_hit_rate_all_found():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b", "c"], ["a", "b"]) == 1.0


def test_hit_rate_none_found():
    from app.evaluator import hit_rate
    assert hit_rate(["x", "y", "z"], ["a", "b"]) == 0.0


def test_hit_rate_partial():
    from app.evaluator import hit_rate
    result = hit_rate(["a", "x", "y"], ["a", "b"])
    assert result == 0.5


def test_hit_rate_empty_relevant():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b"], []) == 0.0


# ── mean_reciprocal_rank unit tests ──────────────────────

def test_mrr_first_rank_is_one():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_second_rank_is_half():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["x", "a", "c"], ["a"]) == 0.5


def test_mrr_no_relevant_found():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


def test_mrr_uses_first_hit():
    from app.evaluator import mean_reciprocal_rank
    # Both "a" and "b" are relevant; the first hit ("b" at rank 2) determines MRR.
    assert mean_reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5


# ── faithfulness unit tests ───────────────────────────────

def test_faithfulness_grounded_sentence():
    from app.evaluator import faithfulness
    chunks = [{"text": "Supervised learning trains models on labeled data to make predictions."}]
    answer = "Supervised learning trains on labeled data."
    score = faithfulness(answer, chunks)
    assert score == 1.0


def test_faithfulness_ungrounded_sentence():
    from app.evaluator import faithfulness
    chunks = [{"text": "Quantum mechanics describes subatomic particles."}]
    answer = "Neural networks learn hierarchical representations."
    score = faithfulness(answer, chunks)
    assert score == 0.0


def test_faithfulness_empty_answer():
    from app.evaluator import faithfulness
    assert faithfulness("", [{"text": "some context text"}]) == 0.0


def test_faithfulness_empty_context():
    from app.evaluator import faithfulness
    assert faithfulness("The model trains on data.", []) == 0.0


def test_faithfulness_partial_grounding():
    from app.evaluator import faithfulness
    chunks = [{"text": "Overfitting occurs when a model memorizes training data."}]
    # First sentence overlaps context; second has no connection.
    answer = "Overfitting occurs when a model memorizes training data. Quantum physics is unrelated."
    score = faithfulness(answer, chunks)
    assert 0.0 < score < 1.0


# ── answer_relevance unit tests ───────────────────────────

def test_answer_relevance_full_overlap():
    from app.evaluator import answer_relevance
    # Non-stopword query terms all appear in answer.
    score = answer_relevance("overfitting neural network", "overfitting neural network training")
    assert score == 1.0


def test_answer_relevance_no_overlap():
    from app.evaluator import answer_relevance
    score = answer_relevance("gradient descent learning rate", "quantum mechanics wave function")
    assert score == 0.0


def test_answer_relevance_partial():
    from app.evaluator import answer_relevance
    score = answer_relevance("overfitting regularization dropout", "overfitting reduces variance")
    assert 0.0 < score < 1.0


def test_answer_relevance_stopwords_only_query():
    from app.evaluator import answer_relevance
    # Query collapses to empty set after stopword removal.
    assert answer_relevance("the is a", "some answer text") == 0.0


# ── chunk_document unit tests ─────────────────────────────

def test_chunk_document_short_text_produces_one_chunk():
    from app.chunker import chunk_document
    chunks = chunk_document("doc1", "Short text here.", {}, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text here."


def test_chunk_document_long_text_produces_multiple_chunks():
    from app.chunker import chunk_document
    text = "x" * 1000
    chunks = chunk_document("doc1", text, {}, chunk_size=512, overlap=64)
    assert len(chunks) >= 2


def test_chunk_document_chunk_id_starts_at_zero():
    from app.chunker import chunk_document
    chunks = chunk_document("my_doc", "Some text.", {}, chunk_size=512, overlap=64)
    assert chunks[0].chunk_id == "my_doc_chunk_0"


def test_chunk_document_chunk_ids_are_sequential():
    from app.chunker import chunk_document
    text = "a" * 1000
    chunks = chunk_document("my_doc", text, {}, chunk_size=512, overlap=64)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"my_doc_chunk_{i}"


def test_chunk_document_doc_id_on_every_chunk():
    from app.chunker import chunk_document
    text = "a" * 1000
    chunks = chunk_document("test_doc", text, {}, chunk_size=512, overlap=64)
    for chunk in chunks:
        assert chunk.doc_id == "test_doc"


def test_chunk_document_metadata_includes_chunk_index():
    from app.chunker import chunk_document
    text = "a" * 1000
    chunks = chunk_document("doc1", text, {}, chunk_size=512, overlap=64)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_chunk_document_metadata_includes_doc_id_field():
    from app.chunker import chunk_document
    chunks = chunk_document("doc1", "Some text.", {}, chunk_size=512, overlap=64)
    assert chunks[0].metadata["doc_id"] == "doc1"


def test_chunk_document_original_metadata_preserved():
    from app.chunker import chunk_document
    meta = {"source": "wiki", "author": "alice"}
    chunks = chunk_document("doc1", "Some text.", meta, chunk_size=512, overlap=64)
    assert chunks[0].metadata["source"] == "wiki"
    assert chunks[0].metadata["author"] == "alice"


def test_chunk_document_overlap_creates_shared_content():
    from app.chunker import chunk_document
    # chunk_size=10, overlap=4 → stride=6
    # chunk 0: text[0:10], chunk 1: text[6:16]
    # chars 6-9 ("ghij") appear at the tail of chunk 0 and head of chunk 1.
    text = "abcdefghijklmno"  # 15 chars
    chunks = chunk_document("doc", text, {}, chunk_size=10, overlap=4)
    assert len(chunks) >= 2
    assert "ghij" in chunks[0].text
    assert chunks[1].text.startswith("ghij")


def test_chunk_document_empty_text_returns_no_chunks():
    from app.chunker import chunk_document
    assert chunk_document("doc", "", {}) == []


def test_chunk_document_whitespace_only_returns_no_chunks():
    from app.chunker import chunk_document
    assert chunk_document("doc", "   \n\t  ", {}) == []


def test_chunk_document_last_chunk_covers_end_of_text():
    from app.chunker import chunk_document
    text = "abcdefghijklmno"  # 15 chars
    chunks = chunk_document("doc", text, {}, chunk_size=10, overlap=4)
    assert chunks[-1].text.endswith(text[-1])


# ── extract_citations unit tests ──────────────────────────

def test_extract_citations_empty_answer_returns_empty():
    from app.citations import extract_citations
    result = extract_citations("", [{"chunk_id": "c1", "text": "some text", "metadata": {}}])
    assert result == []


def test_extract_citations_no_chunks_returns_empty_citations_per_sentence():
    from app.citations import extract_citations
    result = extract_citations("Supervised learning uses labeled data.", [])
    assert len(result) == 1
    assert result[0]["sentence"] == "Supervised learning uses labeled data."
    assert result[0]["citations"] == []


def test_extract_citations_above_threshold_is_cited():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "Supervised learning trains on labeled data.", "metadata": {"title": "ML"}}]
    result = extract_citations("Supervised learning trains on labeled data.", chunks, threshold=0.3)
    assert len(result) == 1
    assert len(result[0]["citations"]) == 1
    assert result[0]["citations"][0]["chunk_id"] == "c1"


def test_extract_citations_below_threshold_not_cited():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "quantum mechanics wave function", "metadata": {}}]
    result = extract_citations("Supervised learning trains on labeled data.", chunks, threshold=0.3)
    assert result[0]["citations"] == []


def test_extract_citations_multiple_sentences_each_mapped():
    from app.citations import extract_citations
    chunks = [
        {"chunk_id": "c1", "text": "Supervised learning trains on labeled data.", "metadata": {}},
        {"chunk_id": "c2", "text": "Overfitting occurs when model memorizes training data.", "metadata": {}},
    ]
    answer = "Supervised learning trains on labeled data. Overfitting occurs when model memorizes training data."
    result = extract_citations(answer, chunks, threshold=0.3)
    assert len(result) == 2
    assert result[0]["citations"][0]["chunk_id"] == "c1"
    assert result[1]["citations"][0]["chunk_id"] == "c2"


def test_extract_citations_sorted_by_overlap_descending():
    from app.citations import extract_citations
    # c2 has more token overlap with the query sentence than c1
    chunks = [
        {"chunk_id": "c1", "text": "supervised learning labeled", "metadata": {}},
        {"chunk_id": "c2", "text": "supervised learning trains labeled data model", "metadata": {}},
    ]
    result = extract_citations("Supervised learning trains on labeled data.", chunks, threshold=0.2)
    assert len(result[0]["citations"]) == 2
    first_overlap = result[0]["citations"][0]["overlap"]
    second_overlap = result[0]["citations"][1]["overlap"]
    assert first_overlap >= second_overlap


def test_extract_citations_title_falls_back_to_empty_when_missing():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "Supervised learning trains on labeled data.", "metadata": {}}]
    result = extract_citations("Supervised learning trains on labeled data.", chunks, threshold=0.3)
    assert result[0]["citations"][0]["title"] == ""


def test_extract_citations_title_uses_metadata_field():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "Supervised learning trains on labeled data.", "metadata": {"title": "ML Basics"}}]
    result = extract_citations("Supervised learning trains on labeled data.", chunks, threshold=0.3)
    assert result[0]["citations"][0]["title"] == "ML Basics"


def test_extract_citations_overlap_is_rounded_to_three_decimal_places():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "Supervised learning trains on labeled data.", "metadata": {}}]
    result = extract_citations("Supervised learning trains on labeled data.", chunks, threshold=0.3)
    overlap = result[0]["citations"][0]["overlap"]
    assert overlap == round(overlap, 3)


def test_extract_citations_stopword_only_sentence_is_skipped():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "some real content here", "metadata": {}}]
    # After stopword removal "The is a." collapses to empty — sentence is skipped
    result = extract_citations("The is a.", chunks, threshold=0.1)
    assert result == []


def test_extract_citations_sentence_structure_has_required_keys():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "Neural networks learn representations.", "metadata": {}}]
    result = extract_citations("Neural networks learn representations.", chunks, threshold=0.1)
    assert len(result) >= 1
    entry = result[0]
    assert "sentence" in entry
    assert "citations" in entry
    for citation in entry["citations"]:
        assert "chunk_id" in citation
        assert "title" in citation
        assert "overlap" in citation
