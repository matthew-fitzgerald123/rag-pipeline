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


# ── Generator.answer_stream() unit tests ──────────────────

import asyncio
import json as _json


async def _collect_stream(async_gen):
    items = []
    async for item in async_gen:
        items.append(item)
    return items


def test_answer_stream_raises_when_model_not_loaded():
    from app.generator import Generator
    gen = Generator()
    gen.model = None

    async def _run():
        async for _ in gen.answer_stream("query", [{"text": "ctx"}]):
            pass

    with pytest.raises(RuntimeError, match="Generator not loaded"):
        asyncio.run(_run())


def test_answer_stream_ends_with_done_sentinel():
    from app.generator import Generator
    from unittest.mock import MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    def fake_stream(prompt, max_tokens, q, done):
        q.put("hello")
        done.set()

    gen._stream_into_queue = fake_stream
    items = asyncio.run(_collect_stream(gen.answer_stream("q", [{"text": "ctx"}])))
    assert items[-1] == "[DONE]"


def test_answer_stream_tokens_are_json_encoded():
    from app.generator import Generator
    from unittest.mock import MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    def fake_stream(prompt, max_tokens, q, done):
        q.put("tok1")
        q.put("tok2")
        done.set()

    gen._stream_into_queue = fake_stream
    items = asyncio.run(_collect_stream(gen.answer_stream("q", [{"text": "ctx"}])))
    token_items = [i for i in items if i != "[DONE]"]
    assert len(token_items) == 2
    for item in token_items:
        parsed = _json.loads(item)
        assert "token" in parsed


def test_answer_stream_yields_all_tokens_in_order():
    from app.generator import Generator
    from unittest.mock import MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    def fake_stream(prompt, max_tokens, q, done):
        for tok in ["alpha", " ", "beta"]:
            q.put(tok)
        done.set()

    gen._stream_into_queue = fake_stream
    items = asyncio.run(_collect_stream(gen.answer_stream("q", [{"text": "ctx"}])))
    tokens = [_json.loads(i)["token"] for i in items if i != "[DONE]"]
    assert tokens == ["alpha", " ", "beta"]


def test_answer_stream_empty_queue_yields_only_done():
    from app.generator import Generator
    from unittest.mock import MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    def fake_stream(prompt, max_tokens, q, done):
        done.set()

    gen._stream_into_queue = fake_stream
    items = asyncio.run(_collect_stream(gen.answer_stream("q", [])))
    assert items == ["[DONE]"]


def test_answer_stream_respects_max_tokens():
    from app.generator import Generator
    from unittest.mock import MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    captured = {}

    def fake_stream(prompt, max_tokens, q, done):
        captured["max_tokens"] = max_tokens
        done.set()

    gen._stream_into_queue = fake_stream
    asyncio.run(_collect_stream(gen.answer_stream("q", [{"text": "ctx"}], max_tokens=256)))
    assert captured["max_tokens"] == 256


def test_answer_stream_default_max_tokens_is_512():
    from app.generator import Generator
    from unittest.mock import MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    captured = {}

    def fake_stream(prompt, max_tokens, q, done):
        captured["max_tokens"] = max_tokens
        done.set()

    gen._stream_into_queue = fake_stream
    asyncio.run(_collect_stream(gen.answer_stream("q", [{"text": "ctx"}])))
    assert captured["max_tokens"] == 512


def test_answer_stream_passes_built_prompt():
    from app.generator import Generator, _build_prompt
    from unittest.mock import MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    captured = {}
    chunks = [{"text": "context content"}]

    def fake_stream(prompt, max_tokens, q, done):
        captured["prompt"] = prompt
        done.set()

    gen._stream_into_queue = fake_stream
    asyncio.run(_collect_stream(gen.answer_stream("What is ML?", chunks)))
    assert captured["prompt"] == _build_prompt("What is ML?", chunks)


# ── hit_rate() unit tests ─────────────────────────────────

def test_hit_rate_all_relevant_retrieved():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b", "c"], ["a", "b"]) == 1.0


def test_hit_rate_none_relevant_retrieved():
    from app.evaluator import hit_rate
    assert hit_rate(["x", "y", "z"], ["a", "b"]) == 0.0


def test_hit_rate_partial():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "x", "y"], ["a", "b"]) == 0.5


def test_hit_rate_empty_relevant_ids_is_zero():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b"], []) == 0.0


def test_hit_rate_empty_retrieved_is_zero():
    from app.evaluator import hit_rate
    assert hit_rate([], ["a"]) == 0.0


def test_hit_rate_single_relevant_found():
    from app.evaluator import hit_rate
    assert hit_rate(["a", "b", "c"], ["a"]) == 1.0


def test_hit_rate_result_bounded():
    from app.evaluator import hit_rate
    score = hit_rate(["a", "b", "c", "d"], ["a", "b", "c"])
    assert 0.0 <= score <= 1.0


# ── mean_reciprocal_rank() unit tests ────────────────────

def test_mrr_first_result_relevant():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_second_result_relevant():
    from app.evaluator import mean_reciprocal_rank
    assert round(mean_reciprocal_rank(["x", "a", "c"], ["a"]), 4) == 0.5


def test_mrr_third_result_relevant():
    from app.evaluator import mean_reciprocal_rank
    assert round(mean_reciprocal_rank(["x", "y", "a"], ["a"]), 4) == round(1 / 3, 4)


def test_mrr_no_relevant_retrieved():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


def test_mrr_empty_relevant_ids():
    from app.evaluator import mean_reciprocal_rank
    assert mean_reciprocal_rank(["a", "b"], []) == 0.0


def test_mrr_rewards_earlier_rank():
    from app.evaluator import mean_reciprocal_rank
    high = mean_reciprocal_rank(["a", "x", "y"], ["a"])
    low  = mean_reciprocal_rank(["x", "y", "a"], ["a"])
    assert high > low


def test_mrr_multiple_relevant_uses_first_hit():
    from app.evaluator import mean_reciprocal_rank
    # "b" appears at rank 2; "a" at rank 1 — first hit drives the score.
    assert mean_reciprocal_rank(["a", "b", "c"], ["a", "b"]) == 1.0


# ── faithfulness() unit tests ─────────────────────────────

def test_faithfulness_fully_grounded():
    from app.evaluator import faithfulness
    chunks = [{"text": "Supervised learning uses labeled training data to learn a mapping."}]
    answer = "Supervised learning uses labeled training data."
    score = faithfulness(answer, chunks)
    assert score > 0.5


def test_faithfulness_ungrounded_answer():
    from app.evaluator import faithfulness
    chunks = [{"text": "Photosynthesis converts sunlight into glucose."}]
    answer = "Quantum mechanics describes particle wave duality."
    score = faithfulness(answer, chunks)
    assert score == 0.0


def test_faithfulness_empty_answer():
    from app.evaluator import faithfulness
    chunks = [{"text": "Some context text here."}]
    assert faithfulness("", chunks) == 0.0


def test_faithfulness_empty_context():
    from app.evaluator import faithfulness
    assert faithfulness("The answer is here.", []) == 0.0


def test_faithfulness_result_bounded():
    from app.evaluator import faithfulness
    chunks = [{"text": "Neural networks learn representations from data."}]
    score = faithfulness("Neural networks learn from data.", chunks)
    assert 0.0 <= score <= 1.0


def test_faithfulness_multiple_chunks_increase_score():
    from app.evaluator import faithfulness
    chunks_one  = [{"text": "Overfitting occurs when a model memorizes training data."}]
    chunks_many = [
        {"text": "Overfitting occurs when a model memorizes training data."},
        {"text": "Regularization helps reduce overfitting in machine learning models."},
    ]
    answer = "Overfitting happens when models memorize training data and regularization helps."
    score_one  = faithfulness(answer, chunks_one)
    score_many = faithfulness(answer, chunks_many)
    assert score_many >= score_one


# ── answer_relevance() unit tests ─────────────────────────

def test_answer_relevance_full_overlap():
    from app.evaluator import answer_relevance
    query  = "overfitting regularization"
    answer = "Overfitting can be reduced by regularization techniques."
    score = answer_relevance(query, answer)
    assert score == 1.0


def test_answer_relevance_no_overlap():
    from app.evaluator import answer_relevance
    query  = "gradient descent optimization"
    answer = "Photosynthesis converts sunlight into glucose."
    score = answer_relevance(query, answer)
    assert score == 0.0


def test_answer_relevance_partial_overlap():
    from app.evaluator import answer_relevance
    query  = "supervised learning classification regression"
    answer = "Supervised learning can solve classification problems."
    score = answer_relevance(query, answer)
    assert 0.0 < score < 1.0


def test_answer_relevance_empty_query():
    from app.evaluator import answer_relevance
    assert answer_relevance("", "Some answer here.") == 0.0


def test_answer_relevance_stopwords_only_query():
    from app.evaluator import answer_relevance
    assert answer_relevance("the a an is", "Something unrelated.") == 0.0


def test_answer_relevance_result_bounded():
    from app.evaluator import answer_relevance
    score = answer_relevance("machine learning model", "Machine learning models generalize.")
    assert 0.0 <= score <= 1.0


def test_answer_relevance_longer_answer_does_not_inflate():
    from app.evaluator import answer_relevance
    query       = "supervised learning"
    short_ans   = "Supervised learning uses labeled examples."
    verbose_ans = "Supervised learning uses labeled examples. " * 20
    assert answer_relevance(query, short_ans) == answer_relevance(query, verbose_ans)


# ── chunk_document() unit tests ───────────────────────────

def test_chunker_empty_text_returns_empty():
    from app.chunker import chunk_document
    assert chunk_document("doc1", "", {}) == []


def test_chunker_whitespace_only_returns_empty():
    from app.chunker import chunk_document
    assert chunk_document("doc1", "   \n\t  ", {}) == []


def test_chunker_short_text_produces_one_chunk():
    from app.chunker import chunk_document
    chunks = chunk_document("doc1", "Hello world.", {}, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_chunker_long_text_produces_multiple_chunks():
    from app.chunker import chunk_document
    text = "x" * 1000
    chunks = chunk_document("doc1", text, {}, chunk_size=512, overlap=64)
    assert len(chunks) > 1


def test_chunker_chunk_ids_follow_naming_convention():
    from app.chunker import chunk_document
    text = "w" * 1000
    chunks = chunk_document("mydoc", text, {}, chunk_size=512, overlap=64)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"mydoc_chunk_{i}"


def test_chunker_doc_id_attached_to_each_chunk():
    from app.chunker import chunk_document
    chunks = chunk_document("docA", "Some text here.", {})
    assert all(c.doc_id == "docA" for c in chunks)


def test_chunker_metadata_preserved_in_each_chunk():
    from app.chunker import chunk_document
    meta = {"source": "test.txt", "author": "Alice"}
    chunks = chunk_document("doc1", "Sample text.", meta)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.metadata["author"] == "Alice"


def test_chunker_metadata_includes_chunk_index():
    from app.chunker import chunk_document
    text = "w" * 1000
    chunks = chunk_document("doc1", text, {}, chunk_size=512, overlap=64)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_chunker_metadata_includes_doc_id():
    from app.chunker import chunk_document
    chunks = chunk_document("docB", "Content here.", {"title": "T"})
    assert all(c.metadata["doc_id"] == "docB" for c in chunks)


def test_chunker_chunk_size_respected():
    from app.chunker import chunk_document
    text = "a" * 2000
    chunks = chunk_document("doc1", text, {}, chunk_size=200, overlap=20)
    for chunk in chunks:
        assert len(chunk.text) <= 200


def test_chunker_overlap_shared_content():
    from app.chunker import chunk_document
    # Build a text where overlap is detectable character-by-character.
    text = "abcdefghij" * 10
    chunks = chunk_document("doc1", text, {}, chunk_size=20, overlap=5)
    assert len(chunks) >= 2
    # The tail of chunk[0] and the head of chunk[1] must share characters.
    tail = chunks[0].text[-5:]
    head = chunks[1].text[:5]
    assert tail == head


def test_chunker_text_exactly_at_chunk_size():
    from app.chunker import chunk_document
    text = "b" * 512
    chunks = chunk_document("doc1", text, {}, chunk_size=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunker_returns_chunk_dataclass():
    from app.chunker import chunk_document, Chunk
    chunks = chunk_document("doc1", "Hello.", {})
    assert all(isinstance(c, Chunk) for c in chunks)


# ── citations._tokens() and _overlap() unit tests ─────────

def test_citations_tokens_lowercases_and_splits():
    from app.citations import _tokens
    result = _tokens("Supervised Learning")
    assert "supervised" in result
    assert "learning" in result


def test_citations_tokens_removes_stopwords():
    from app.citations import _tokens
    result = _tokens("the cat is on the mat")
    assert "the" not in result
    assert "is" not in result
    assert "on" not in result
    assert "cat" in result
    assert "mat" in result


def test_citations_tokens_empty_string():
    from app.citations import _tokens
    assert _tokens("") == set()


def test_citations_overlap_identical_sets():
    from app.citations import _overlap
    s = {"neural", "network", "learns"}
    assert _overlap(s, s) == 1.0


def test_citations_overlap_disjoint_sets():
    from app.citations import _overlap
    assert _overlap({"a", "b"}, {"c", "d"}) == 0.0


def test_citations_overlap_partial():
    from app.citations import _overlap
    a = {"a", "b", "c"}
    b = {"a", "b", "x"}
    assert round(_overlap(a, b), 4) == round(2 / 3, 4)


def test_citations_overlap_empty_first():
    from app.citations import _overlap
    assert _overlap(set(), {"a", "b"}) == 0.0


def test_citations_overlap_empty_second():
    from app.citations import _overlap
    assert _overlap({"a", "b"}, set()) == 0.0


def test_extract_citations_empty_answer():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "Some text.", "metadata": {}}]
    result = extract_citations("", chunks)
    assert result == []


def test_extract_citations_no_chunks_above_threshold():
    from app.citations import extract_citations
    chunks = [{"chunk_id": "c1", "text": "Photosynthesis converts sunlight.", "metadata": {}}]
    result = extract_citations("Quantum mechanics describes particles.", chunks, threshold=0.8)
    assert all(len(entry["citations"]) == 0 for entry in result)


def test_extract_citations_multiple_sentences():
    from app.citations import extract_citations
    chunks = [
        {"chunk_id": "c1", "text": "Neural networks learn representations.", "metadata": {"title": "ML"}},
        {"chunk_id": "c2", "text": "Overfitting occurs when models memorize data.", "metadata": {"title": "ML"}},
    ]
    answer = "Neural networks learn features. Overfitting is a common problem."
    result = extract_citations(answer, chunks, threshold=0.2)
    assert len(result) == 2
    sentences = [r["sentence"] for r in result]
    assert any("Neural" in s for s in sentences)
    assert any("Overfitting" in s for s in sentences)


def test_extract_citations_sorted_by_overlap_descending():
    from app.citations import extract_citations
    chunks = [
        {"chunk_id": "c1", "text": "supervised learning uses labeled training data classification regression", "metadata": {}},
        {"chunk_id": "c2", "text": "supervised training", "metadata": {}},
    ]
    answer = "Supervised learning uses labeled training data for classification."
    result = extract_citations(answer, chunks, threshold=0.1)
    assert len(result) >= 1
    cites = result[0]["citations"]
    if len(cites) >= 2:
        assert cites[0]["overlap"] >= cites[1]["overlap"]


# ── rerank() unit tests ───────────────────────────────────

def _make_candidates(*texts):
    return [
        {"chunk_id": f"c{i}", "text": t, "metadata": {}, "score": 0.5}
        for i, t in enumerate(texts)
    ]


def test_rerank_empty_candidates_returns_empty():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    with patch("app.reranker._get_model", return_value=MagicMock(predict=lambda p: [])):
        assert rerank("q", [], top_k=3) == []


def test_rerank_annotates_rerank_score():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    candidates = _make_candidates("text A", "text B")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.3]
    with patch("app.reranker._get_model", return_value=mock_model):
        result = rerank("query", candidates, top_k=2)
    assert all("rerank_score" in c for c in result)


def test_rerank_sorts_by_score_descending():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    candidates = _make_candidates("low", "mid", "high")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.5, 0.9]
    with patch("app.reranker._get_model", return_value=mock_model):
        result = rerank("query", candidates, top_k=3)
    scores = [c["rerank_score"] for c in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_returns_top_k():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    candidates = _make_candidates("a", "b", "c", "d", "e")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5, 0.4, 0.3, 0.2, 0.1]
    with patch("app.reranker._get_model", return_value=mock_model):
        result = rerank("query", candidates, top_k=2)
    assert len(result) == 2


def test_rerank_top_k_greater_than_candidates_returns_all():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    candidates = _make_candidates("only one")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.7]
    with patch("app.reranker._get_model", return_value=mock_model):
        result = rerank("query", candidates, top_k=10)
    assert len(result) == 1


def test_rerank_preserves_original_fields():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    candidates = [{"chunk_id": "x1", "text": "some text", "metadata": {"title": "Doc"}, "score": 0.8}]
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.6]
    with patch("app.reranker._get_model", return_value=mock_model):
        result = rerank("query", candidates, top_k=1)
    assert result[0]["chunk_id"] == "x1"
    assert result[0]["text"] == "some text"
    assert result[0]["metadata"]["title"] == "Doc"
    assert result[0]["score"] == 0.8


def test_rerank_highest_score_is_first():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    candidates = _make_candidates("worst", "best", "middle")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.99, 0.5]
    with patch("app.reranker._get_model", return_value=mock_model):
        result = rerank("query", candidates, top_k=3)
    assert result[0]["chunk_id"] == "c1"


def test_rerank_score_is_rounded():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    candidates = _make_candidates("text")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.123456789]
    with patch("app.reranker._get_model", return_value=mock_model):
        result = rerank("query", candidates, top_k=1)
    assert result[0]["rerank_score"] == round(0.123456789, 4)


def test_rerank_passes_query_chunk_pairs_to_model():
    from app.reranker import rerank
    from unittest.mock import patch, MagicMock
    candidates = _make_candidates("chunk one", "chunk two")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5, 0.3]
    with patch("app.reranker._get_model", return_value=mock_model):
        rerank("my query", candidates, top_k=2)
    called_pairs = mock_model.predict.call_args[0][0]
    assert called_pairs == [("my query", "chunk one"), ("my query", "chunk two")]


# ── _build_prompt() unit tests ────────────────────────────

def test_build_prompt_single_chunk_numbered():
    from app.generator import _build_prompt
    chunks = [{"text": "The sky is blue."}]
    prompt = _build_prompt("What color is the sky?", chunks)
    assert "[1] The sky is blue." in prompt


def test_build_prompt_multiple_chunks_numbered_sequentially():
    from app.generator import _build_prompt
    chunks = [{"text": "First."}, {"text": "Second."}, {"text": "Third."}]
    prompt = _build_prompt("question", chunks)
    assert "[1] First." in prompt
    assert "[2] Second." in prompt
    assert "[3] Third." in prompt


def test_build_prompt_query_appears_in_prompt():
    from app.generator import _build_prompt
    prompt = _build_prompt("What is machine learning?", [{"text": "ctx"}])
    assert "What is machine learning?" in prompt


def test_build_prompt_includes_instruction_text():
    from app.generator import _build_prompt
    prompt = _build_prompt("q", [{"text": "ctx"}])
    assert "Answer the question using only the context below" in prompt


def test_build_prompt_includes_fallback_instruction():
    from app.generator import _build_prompt
    prompt = _build_prompt("q", [{"text": "ctx"}])
    assert "I don't have enough information" in prompt


def test_build_prompt_empty_chunks_produces_no_numbered_context():
    from app.generator import _build_prompt
    prompt = _build_prompt("q", [])
    assert "[1]" not in prompt


def test_build_prompt_chunks_separated_by_blank_line():
    from app.generator import _build_prompt
    chunks = [{"text": "A"}, {"text": "B"}]
    prompt = _build_prompt("q", chunks)
    assert "[1] A\n\n[2] B" in prompt


def test_build_prompt_inst_tags_wrap_content():
    from app.generator import _build_prompt
    prompt = _build_prompt("q", [{"text": "ctx"}])
    assert prompt.startswith("[INST]")
    assert prompt.endswith("[/INST]")
