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
