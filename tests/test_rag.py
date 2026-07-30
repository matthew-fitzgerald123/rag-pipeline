from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, ".")

from scripts.ingest import *  # noqa

from app.main import app

client = TestClient(app)

@pytest.mark.integration
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["chunks_indexed"] > 0

@pytest.mark.integration
def test_index_stats():
    r = client.get("/index/stats")
    assert r.status_code == 200
    assert r.json()["total_chunks"] > 0

@pytest.mark.integration
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

@pytest.mark.integration
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
    assert ndcg_at_k(["x", "y", "a"], ["a"], k=2) == 0.0

@pytest.mark.integration
def test_eval_summary():
    r = client.get("/eval/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_queries" in data
    assert data["total_queries"] > 0

@pytest.mark.integration
def test_eval_summary_includes_avg_ndcg():
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

@pytest.mark.integration
def test_eval_history():
    r = client.get("/eval/history?limit=5")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

@pytest.mark.integration
def test_eval_history_includes_ndcg_field():
    r = client.get("/eval/history?limit=5")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    for row in rows:
        assert "ndcg" in row

@pytest.mark.integration
def test_empty_query_still_returns():
    r = client.post("/query", json={"query": "xyzzy nonsense query 12345", "top_k": 3})
    assert r.status_code in [200, 404]

@pytest.mark.integration
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


# ── ingest_file() unit tests ──────────────────────────────────


def _make_path(content="Some content here.", stem="test_doc"):
    from unittest.mock import MagicMock
    path = MagicMock()
    path.read_text.return_value = content
    path.stem = stem
    path.__str__ = lambda self: f"data/{stem}.txt"
    return path


def test_ingest_file_empty_content_does_not_add_to_db():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock
    path = _make_path(content="")
    db = MagicMock()
    ingest_file(path, db)
    db.add.assert_not_called()


def test_ingest_file_whitespace_content_does_not_add_to_db():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock
    path = _make_path(content="   \n\t  ")
    db = MagicMock()
    ingest_file(path, db)
    db.add.assert_not_called()


def test_ingest_file_empty_content_does_not_commit():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock
    path = _make_path(content="")
    db = MagicMock()
    ingest_file(path, db)
    db.commit.assert_not_called()


def test_ingest_file_duplicate_title_skips_add_chunks():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock, patch
    path = _make_path(content="Some content.")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = MagicMock()
    with patch("scripts.ingest.vector_store") as mock_vs:
        ingest_file(path, db)
    mock_vs.add_chunks.assert_not_called()


def test_ingest_file_duplicate_title_does_not_add_to_db():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock, patch
    path = _make_path(content="Some content.")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = MagicMock()
    with patch("scripts.ingest.vector_store"):
        ingest_file(path, db)
    db.add.assert_not_called()


def test_ingest_file_new_doc_sets_title_from_stem():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock, patch
    path = _make_path(content="ML content.", stem="ml_basics")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()
    with patch("scripts.ingest.Document", side_effect=capture), \
         patch("scripts.ingest.chunk_document", return_value=[]), \
         patch("scripts.ingest.vector_store"):
        ingest_file(path, db)
    assert captured["title"] == "ml_basics"


def test_ingest_file_new_doc_doc_id_is_8_chars():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock, patch
    path = _make_path(content="Some content.")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()
    with patch("scripts.ingest.Document", side_effect=capture), \
         patch("scripts.ingest.chunk_document", return_value=[]), \
         patch("scripts.ingest.vector_store"):
        ingest_file(path, db)
    assert len(captured["doc_id"]) == 8


def test_ingest_file_new_doc_adds_to_db():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock, patch
    path = _make_path(content="Some content.")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None
    with patch("scripts.ingest.Document"), \
         patch("scripts.ingest.chunk_document", return_value=[]), \
         patch("scripts.ingest.vector_store"):
        ingest_file(path, db)
    db.add.assert_called_once()


def test_ingest_file_new_doc_commits_to_db():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock, patch
    path = _make_path(content="Some content.")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None
    with patch("scripts.ingest.Document"), \
         patch("scripts.ingest.chunk_document", return_value=[]), \
         patch("scripts.ingest.vector_store"):
        ingest_file(path, db)
    db.commit.assert_called_once()


def test_ingest_file_passes_stripped_content_to_chunk_document():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock, patch
    path = _make_path(content="  ML content here.  ")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None
    with patch("scripts.ingest.Document"), \
         patch("scripts.ingest.chunk_document", return_value=[]) as mock_chunk, \
         patch("scripts.ingest.vector_store"):
        ingest_file(path, db)
    call_kwargs = mock_chunk.call_args[1]
    assert call_kwargs["text"] == "ML content here."


def test_ingest_file_metadata_includes_title():
    from scripts.ingest import ingest_file
    from unittest.mock import MagicMock, patch
    path = _make_path(content="Content.", stem="my_doc")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None
    with patch("scripts.ingest.Document"), \
         patch("scripts.ingest.chunk_document", return_value=[]) as mock_chunk, \
         patch("scripts.ingest.vector_store"):
        ingest_file(path, db)
    call_kwargs = mock_chunk.call_args[1]
    assert call_kwargs["metadata"]["title"] == "my_doc"


def test_ingest_file_calls_add_chunks_with_chunk_document_result():
    from scripts.ingest import ingest_file
    from app.chunker import Chunk
    from unittest.mock import MagicMock, patch
    path = _make_path(content="Some content.")
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None
    fake_chunks = [Chunk(chunk_id="c1", doc_id="abc12345", text="Some content.", metadata={})]
    with patch("scripts.ingest.Document"), \
         patch("scripts.ingest.chunk_document", return_value=fake_chunks), \
         patch("scripts.ingest.vector_store") as mock_vs:
        ingest_file(path, db)
    mock_vs.add_chunks.assert_called_once_with(fake_chunks)


# ── hit_rate() unit tests ─────────────────────────────────────


def test_hit_rate_all_relevant_retrieved():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b", "c"], ["a", "b"]) == 1.0


def test_hit_rate_partial_overlap():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "x", "y"], ["a", "b"]) == 0.5


def test_hit_rate_no_overlap():
    from app.evaluator import hit_rate
    assert hit_rate(["x", "y", "z"], ["a", "b"]) == 0.0


def test_hit_rate_empty_relevant_is_zero():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b"], []) == 0.0


def test_hit_rate_empty_retrieved_is_zero():
    from app.evaluator import hit_rate
    assert hit_rate([], ["a"]) == 0.0


# ── mean_reciprocal_rank() unit tests ────────────────────────


def test_mrr_first_result_relevant():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_second_result_relevant():
    from app.evaluator import mean_reciprocal_rank
    result = mean_reciprocal_rank(["x", "a", "c"], ["a"])
    assert abs(result - 0.5) < 0.001


def test_mrr_no_relevant_retrieved():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


def test_mrr_empty_relevant_is_zero():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["a", "b"], []) == 0.0


def test_mrr_rewards_earlier_rank():
    from app.evaluator import mean_reciprocal_rank
    early = mean_reciprocal_rank(["a", "x", "y"], ["a"])
    late = mean_reciprocal_rank(["x", "y", "a"], ["a"])
    assert early > late


# ── faithfulness() unit tests ────────────────────────────────


def test_faithfulness_empty_answer_is_zero():
    from app.evaluator import faithfulness
    assert faithfulness("", [{"text": "Some context."}]) == 0.0


def test_faithfulness_grounded_answer():
    from app.evaluator import faithfulness
    chunks = [{"text": "Supervised learning trains models on labeled data."}]
    answer = "Supervised learning uses labeled training data."
    score = faithfulness(answer, chunks)
    assert score > 0.0


def test_faithfulness_ungrounded_answer():
    from app.evaluator import faithfulness
    chunks = [{"text": "Cats are mammals that purr."}]
    answer = "Quantum mechanics describes subatomic particles."
    score = faithfulness(answer, chunks)
    assert score == 0.0


def test_faithfulness_multiple_sentences_partial():
    from app.evaluator import faithfulness
    chunks = [{"text": "Neural networks learn from data."}]
    answer = "Neural networks learn from data. Cats are mammals."
    score = faithfulness(answer, chunks)
    assert 0.0 < score < 1.0


def test_faithfulness_returns_float_in_range():
    from app.evaluator import faithfulness
    chunks = [{"text": "Some context text here."}]
    score = faithfulness("Some answer text.", chunks)
    assert 0.0 <= score <= 1.0


# ── answer_relevance() unit tests ────────────────────────────


def test_answer_relevance_full_overlap():
    from app.evaluator import answer_relevance
    assert answer_relevance("supervised learning", "supervised learning trains models") == 1.0


def test_answer_relevance_no_overlap():
    from app.evaluator import answer_relevance
    assert answer_relevance("supervised learning", "cats purr loudly") == 0.0


def test_answer_relevance_partial_overlap():
    from app.evaluator import answer_relevance
    score = answer_relevance("supervised learning models", "supervised training data")
    assert 0.0 < score < 1.0


def test_answer_relevance_stopwords_only_query_is_zero():
    from app.evaluator import answer_relevance
    assert answer_relevance("the a an", "anything goes here") == 0.0


def test_answer_relevance_returns_float_in_range():
    from app.evaluator import answer_relevance
    score = answer_relevance("what is overfitting", "overfitting occurs when a model memorizes training data")
    assert 0.0 <= score <= 1.0


# ── chunk_document() unit tests ──────────────────────────────


def test_chunk_document_empty_text_returns_no_chunks():
    from app.chunker import chunk_document
    assert chunk_document(doc_id="d1", text="", metadata={}) == []


def test_chunk_document_whitespace_text_returns_no_chunks():
    from app.chunker import chunk_document
    assert chunk_document(doc_id="d1", text="   \n  ", metadata={}) == []


def test_chunk_document_short_text_is_single_chunk():
    from app.chunker import chunk_document
    chunks = chunk_document(doc_id="d1", text="Hello world.", metadata={})
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_chunk_document_chunk_ids_use_doc_id():
    from app.chunker import chunk_document
    chunks = chunk_document(doc_id="doc99", text="Hello world.", metadata={})
    assert chunks[0].chunk_id.startswith("doc99")


def test_chunk_document_chunk_ids_are_sequential():
    from app.chunker import chunk_document
    text = "a" * 600
    chunks = chunk_document(doc_id="d1", text=text, metadata={}, chunk_size=256, overlap=0)
    ids = [c.chunk_id for c in chunks]
    assert ids[0].endswith("_0")
    assert ids[1].endswith("_1")


def test_chunk_document_respects_chunk_size():
    from app.chunker import chunk_document
    text = "x" * 1000
    chunks = chunk_document(doc_id="d1", text=text, metadata={}, chunk_size=200, overlap=0)
    for c in chunks[:-1]:
        assert len(c.text) <= 200


def test_chunk_document_overlap_produces_repeated_content():
    from app.chunker import chunk_document
    text = "a" * 100
    chunks = chunk_document(doc_id="d1", text=text, metadata={}, chunk_size=60, overlap=20)
    assert len(chunks) >= 2
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_chunk_document_metadata_propagated():
    from app.chunker import chunk_document
    meta = {"title": "ml_basics"}
    chunks = chunk_document(doc_id="d1", text="Hello world.", metadata=meta)
    assert chunks[0].metadata["title"] == "ml_basics"


def test_chunk_document_doc_id_in_chunk():
    from app.chunker import chunk_document
    chunks = chunk_document(doc_id="abc123", text="Hello world.", metadata={})
    assert chunks[0].doc_id == "abc123"


def test_chunk_document_long_text_produces_multiple_chunks():
    from app.chunker import chunk_document
    text = "word " * 300
    chunks = chunk_document(doc_id="d1", text=text, metadata={}, chunk_size=100, overlap=10)
    assert len(chunks) > 1


# ── rerank() unit tests ───────────────────────────────────────


def _make_candidates(*texts):
    return [{"chunk_id": f"c{i}", "text": t, "score": 0.5} for i, t in enumerate(texts)]


def test_rerank_empty_candidates_returns_empty():
    from app.reranker import rerank
    from unittest.mock import patch
    with patch("app.reranker._get_model"):
        assert rerank("query", [], top_k=3) == []


def test_rerank_annotates_candidates_with_rerank_score():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    model = MagicMock()
    model.predict.return_value = [0.9, 0.4]
    with patch("app.reranker._get_model", return_value=model):
        results = rerank("q", _make_candidates("text A", "text B"), top_k=2)
    assert all("rerank_score" in r for r in results)
    assert all(isinstance(r["rerank_score"], float) for r in results)


def test_rerank_returns_sorted_by_score_descending():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    model = MagicMock()
    model.predict.return_value = [0.2, 0.9, 0.5]
    with patch("app.reranker._get_model", return_value=model):
        results = rerank("q", _make_candidates("low", "high", "mid"), top_k=3)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_respects_top_k_limit():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    model = MagicMock()
    model.predict.return_value = [0.9, 0.8, 0.7, 0.6]
    with patch("app.reranker._get_model", return_value=model):
        results = rerank("q", _make_candidates("a", "b", "c", "d"), top_k=2)
    assert len(results) == 2


def test_rerank_top_k_larger_than_candidates_returns_all():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    model = MagicMock()
    model.predict.return_value = [0.8, 0.3]
    with patch("app.reranker._get_model", return_value=model):
        results = rerank("q", _make_candidates("a", "b"), top_k=10)
    assert len(results) == 2


def test_rerank_calls_model_with_correct_pairs():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    model = MagicMock()
    model.predict.return_value = [0.5, 0.6]
    candidates = _make_candidates("chunk text one", "chunk text two")
    with patch("app.reranker._get_model", return_value=model):
        rerank("my query", candidates, top_k=2)
    model.predict.assert_called_once_with([("my query", "chunk text one"), ("my query", "chunk text two")])


def test_rerank_score_is_rounded_to_4_decimal_places():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    model = MagicMock()
    model.predict.return_value = [0.123456789]
    with patch("app.reranker._get_model", return_value=model):
        results = rerank("q", _make_candidates("text"), top_k=1)
    score = results[0]["rerank_score"]
    assert score == round(score, 4)


def test_rerank_preserves_other_candidate_fields():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    model = MagicMock()
    model.predict.return_value = [0.7]
    candidate = {"chunk_id": "abc", "text": "hello", "score": 0.99, "metadata": {"title": "doc"}}
    with patch("app.reranker._get_model", return_value=model):
        results = rerank("q", [candidate], top_k=1)
    assert results[0]["chunk_id"] == "abc"
    assert results[0]["text"] == "hello"
    assert results[0]["score"] == 0.99
    assert results[0]["metadata"] == {"title": "doc"}


def test_rerank_highest_scoring_is_first():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    model = MagicMock()
    model.predict.return_value = [0.1, 0.95, 0.3]
    candidates = _make_candidates("low", "best", "mid")
    with patch("app.reranker._get_model", return_value=model):
        results = rerank("q", candidates, top_k=3)
    assert results[0]["chunk_id"] == "c1"


# ── extract_citations() additional unit tests ─────────────────


def test_extract_citations_empty_answer_returns_empty():
    from app.citations import extract_citations
    assert extract_citations("", [{"chunk_id": "c1", "text": "some text", "metadata": {}}]) == []


def test_extract_citations_no_chunks_returns_sentences_with_no_citations():
    from app.citations import extract_citations
    result = extract_citations("Supervised learning uses labeled data.", [], threshold=0.0)
    assert len(result) == 1
    assert result[0]["citations"] == []


def test_extract_citations_below_threshold_not_cited():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "cats purr loudly at night", "metadata": {}}]
    result = extract_citations("Supervised learning uses labeled data.", chunks, threshold=0.5)
    assert result[0]["citations"] == []


def test_extract_citations_multiple_chunks_all_above_threshold_cited():
    from app.citations import extract_citations
    chunks = [
        {"chunk_id": "c1", "text": "Supervised learning trains on labeled data.", "metadata": {}},
        {"chunk_id": "c2", "text": "Supervised models learn from labeled examples.", "metadata": {}},
    ]
    result = extract_citations("Supervised learning uses labeled data.", chunks, threshold=0.2)
    assert len(result[0]["citations"]) == 2


def test_extract_citations_citations_sorted_by_overlap_descending():
    from app.citations import extract_citations
    chunks = [
        {"chunk_id": "weak", "text": "learning supervised", "metadata": {}},
        {"chunk_id": "strong", "text": "supervised learning trains labeled data models", "metadata": {}},
    ]
    result = extract_citations("Supervised learning trains on labeled data.", chunks, threshold=0.1)
    overlaps = [c["overlap"] for c in result[0]["citations"]]
    assert overlaps == sorted(overlaps, reverse=True)


def test_extract_citations_multiple_sentences_each_attributed_independently():
    from app.citations import extract_citations
    chunks = [
        {"chunk_id": "c1", "text": "supervised learning labeled data training", "metadata": {}},
        {"chunk_id": "c2", "text": "neural networks universal function approximators", "metadata": {}},
    ]
    answer = "Supervised learning uses labeled training data. Neural networks are universal function approximators."
    result = extract_citations(answer, chunks, threshold=0.2)
    assert len(result) == 2
    ids_first = {c["chunk_id"] for c in result[0]["citations"]}
    ids_second = {c["chunk_id"] for c in result[1]["citations"]}
    assert "c1" in ids_first
    assert "c2" in ids_second


def test_extract_citations_title_from_metadata():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "supervised learning labeled data.", "metadata": {"title": "ML Guide"}}]
    result = extract_citations("Supervised learning uses labeled data.", chunks, threshold=0.2)
    assert result[0]["citations"][0]["title"] == "ML Guide"


def test_extract_citations_missing_title_defaults_to_empty_string():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "supervised learning labeled data.", "metadata": {}}]
    result = extract_citations("Supervised learning uses labeled data.", chunks, threshold=0.2)
    assert result[0]["citations"][0]["title"] == ""
