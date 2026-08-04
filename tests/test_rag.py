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

@pytest.mark.integration
def test_eval_summary():
    r = client.get("/eval/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_queries" in data
    assert data["total_queries"] > 0

@pytest.mark.integration
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


# ── Generator.answer() unit tests ────────────────────────

def test_answer_raises_when_model_not_loaded():
    from app.generator import Generator
    gen = Generator()
    gen.model = None
    with pytest.raises(RuntimeError, match="Generator not loaded"):
        gen.answer("query", [{"text": "ctx"}])


def test_answer_calls_generate_with_correct_prompt():
    from app.generator import Generator, _build_prompt
    from unittest.mock import patch, MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()
    chunks = [{"text": "context here"}]
    captured = {}
    with patch("app.generator.generate", side_effect=lambda m, t, prompt, max_tokens, verbose: captured.update({"prompt": prompt}) or "answer") as _:
        gen.answer("What is ML?", chunks)
    assert captured["prompt"] == _build_prompt("What is ML?", chunks)


def test_answer_passes_max_tokens():
    from app.generator import Generator
    from unittest.mock import patch, MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()
    captured = {}
    with patch("app.generator.generate", side_effect=lambda m, t, prompt, max_tokens, verbose: captured.update({"max_tokens": max_tokens}) or "response"):
        gen.answer("q", [{"text": "ctx"}], max_tokens=256)
    assert captured["max_tokens"] == 256


def test_answer_default_max_tokens_is_512():
    from app.generator import Generator
    from unittest.mock import patch, MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()
    captured = {}
    with patch("app.generator.generate", side_effect=lambda m, t, prompt, max_tokens, verbose: captured.update({"max_tokens": max_tokens}) or "response"):
        gen.answer("q", [{"text": "ctx"}])
    assert captured["max_tokens"] == 512


def test_answer_strips_whitespace_from_response():
    from app.generator import Generator
    from unittest.mock import patch, MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()
    with patch("app.generator.generate", return_value="  padded answer  \n"):
        result = gen.answer("q", [{"text": "ctx"}])
    assert result == "padded answer"


def test_answer_returns_generate_output():
    from app.generator import Generator
    from unittest.mock import patch, MagicMock
    gen = Generator()
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()
    with patch("app.generator.generate", return_value="The answer."):
        result = gen.answer("q", [{"text": "ctx"}])
    assert result == "The answer."


def test_answer_passes_model_and_tokenizer_to_generate():
    from app.generator import Generator
    from unittest.mock import patch, MagicMock
    gen = Generator()
    gen.model = MagicMock(name="the_model")
    gen.tokenizer = MagicMock(name="the_tokenizer")
    captured = {}
    with patch("app.generator.generate", side_effect=lambda m, t, prompt, max_tokens, verbose: captured.update({"model": m, "tokenizer": t}) or "ans"):
        gen.answer("q", [{"text": "ctx"}])
    assert captured["model"] is gen.model
    assert captured["tokenizer"] is gen.tokenizer


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


def test_chunker_overlap_equal_to_chunk_size_raises():
    from app.chunker import chunk_document
    with pytest.raises(ValueError, match="overlap"):
        chunk_document("doc1", "some text", {}, chunk_size=64, overlap=64)


def test_chunker_overlap_greater_than_chunk_size_raises():
    from app.chunker import chunk_document
    with pytest.raises(ValueError, match="overlap"):
        chunk_document("doc1", "some text", {}, chunk_size=64, overlap=100)


def test_chunker_overlap_just_below_chunk_size_is_valid():
    from app.chunker import chunk_document
    chunks = chunk_document("doc1", "x" * 200, {}, chunk_size=64, overlap=63)
    assert len(chunks) >= 1


def test_chunker_zero_overlap_is_valid():
    from app.chunker import chunk_document
    chunks = chunk_document("doc1", "Hello world.", {}, chunk_size=512, overlap=0)
    assert len(chunks) == 1


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


# ── VectorStore.add_chunks() unit tests ──────────────────

def test_add_chunks_empty_list_is_noop():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock, patch
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs._bm25 = None
    vs._bm25_ids = []
    with patch.object(vs, "_rebuild_bm25") as mock_rebuild:
        vs.add_chunks([])
    vs.collection.add.assert_not_called()
    mock_rebuild.assert_not_called()


def test_add_chunks_encodes_texts_with_embedder():
    from app.vector_store import VectorStore
    from app.chunker import Chunk
    from unittest.mock import MagicMock, patch
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs._bm25 = None
    vs._bm25_ids = []
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1, 0.2]])
    vs.embedder = fake_embedder
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="some text", metadata={})
    with patch.object(vs, "_rebuild_bm25"):
        vs.add_chunks([chunk])
    fake_embedder.encode.assert_called_once()
    call_kwargs = fake_embedder.encode.call_args
    assert call_kwargs[0][0] == ["some text"]
    assert call_kwargs[1].get("normalize_embeddings") is True


def test_add_chunks_passes_batch_size_32():
    from app.vector_store import VectorStore
    from app.chunker import Chunk
    from unittest.mock import MagicMock, patch
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs._bm25 = None
    vs._bm25_ids = []
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1, 0.2]])
    vs.embedder = fake_embedder
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="text", metadata={})
    with patch.object(vs, "_rebuild_bm25"):
        vs.add_chunks([chunk])
    assert fake_embedder.encode.call_args[1].get("batch_size") == 32


def test_add_chunks_calls_collection_add_with_correct_ids():
    from app.vector_store import VectorStore
    from app.chunker import Chunk
    from unittest.mock import MagicMock, patch
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs._bm25 = None
    vs._bm25_ids = []
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    vs.embedder = fake_embedder
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="first", metadata={}),
        Chunk(chunk_id="c2", doc_id="d1", text="second", metadata={}),
    ]
    with patch.object(vs, "_rebuild_bm25"):
        vs.add_chunks(chunks)
    call_kwargs = vs.collection.add.call_args[1]
    assert call_kwargs["ids"] == ["c1", "c2"]


def test_add_chunks_calls_collection_add_with_correct_documents():
    from app.vector_store import VectorStore
    from app.chunker import Chunk
    from unittest.mock import MagicMock, patch
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs._bm25 = None
    vs._bm25_ids = []
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.5, 0.6]])
    vs.embedder = fake_embedder
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="hello world", metadata={})
    with patch.object(vs, "_rebuild_bm25"):
        vs.add_chunks([chunk])
    call_kwargs = vs.collection.add.call_args[1]
    assert call_kwargs["documents"] == ["hello world"]


def test_add_chunks_calls_collection_add_with_metadatas():
    from app.vector_store import VectorStore
    from app.chunker import Chunk
    from unittest.mock import MagicMock, patch
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs._bm25 = None
    vs._bm25_ids = []
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1, 0.2]])
    vs.embedder = fake_embedder
    meta = {"title": "Test Doc", "chunk_index": 0}
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="text", metadata=meta)
    with patch.object(vs, "_rebuild_bm25"):
        vs.add_chunks([chunk])
    call_kwargs = vs.collection.add.call_args[1]
    assert call_kwargs["metadatas"] == [meta]


def test_add_chunks_rebuilds_bm25_after_adding():
    from app.vector_store import VectorStore
    from app.chunker import Chunk
    from unittest.mock import MagicMock, patch
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs._bm25 = None
    vs._bm25_ids = []
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1, 0.2]])
    vs.embedder = fake_embedder
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="text", metadata={})
    with patch.object(vs, "_rebuild_bm25") as mock_rebuild:
        vs.add_chunks([chunk])
    mock_rebuild.assert_called_once()


# ── VectorStore._dense_query() unit tests ─────────────────

def test_dense_query_converts_distance_to_score():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1, 0.9]])
    vs.embedder = fake_embedder
    vs.collection = MagicMock()
    vs.collection.count.return_value = 3
    vs.collection.query.return_value = {
        "ids": [["c1", "c2"]],
        "distances": [[0.2, 0.5]],
    }
    result = vs._dense_query("test query", top_k=2)
    assert result["c1"] == round(1 - 0.2, 4)
    assert result["c2"] == round(1 - 0.5, 4)


def test_dense_query_encodes_query_with_normalize():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1, 0.9]])
    vs.embedder = fake_embedder
    vs.collection = MagicMock()
    vs.collection.count.return_value = 1
    vs.collection.query.return_value = {"ids": [[]], "distances": [[]]}
    vs._dense_query("my query", top_k=1)
    call_args = fake_embedder.encode.call_args
    assert call_args[0][0] == ["my query"]
    assert call_args[1].get("normalize_embeddings") is True


def test_dense_query_caps_n_results_at_collection_count():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1]])
    vs.embedder = fake_embedder
    vs.collection = MagicMock()
    vs.collection.count.return_value = 3
    vs.collection.query.return_value = {"ids": [[]], "distances": [[]]}
    vs._dense_query("query", top_k=10)
    call_kwargs = vs.collection.query.call_args[1]
    assert call_kwargs["n_results"] == 3


def test_dense_query_n_results_is_top_k_times_two_when_collection_large():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1]])
    vs.embedder = fake_embedder
    vs.collection = MagicMock()
    vs.collection.count.return_value = 100
    vs.collection.query.return_value = {"ids": [[]], "distances": [[]]}
    vs._dense_query("query", top_k=5)
    call_kwargs = vs.collection.query.call_args[1]
    assert call_kwargs["n_results"] == 10


def test_dense_query_rounds_scores_to_four_decimals():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1]])
    vs.embedder = fake_embedder
    vs.collection = MagicMock()
    vs.collection.count.return_value = 1
    vs.collection.query.return_value = {
        "ids": [["c1"]],
        "distances": [[0.123456789]],
    }
    result = vs._dense_query("query", top_k=1)
    assert result["c1"] == round(1 - 0.123456789, 4)


def test_dense_query_returns_empty_when_no_results():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    import numpy as np
    vs = VectorStore.__new__(VectorStore)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.array([[0.1]])
    vs.embedder = fake_embedder
    vs.collection = MagicMock()
    vs.collection.count.return_value = 0
    vs.collection.query.return_value = {"ids": [[]], "distances": [[]]}
    result = vs._dense_query("query", top_k=5)
    assert result == {}


# ── VectorStore._rebuild_bm25() unit tests ────────────────

def test_rebuild_bm25_sets_none_when_collection_empty():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.count.return_value = 0
    vs._bm25 = MagicMock()
    vs._bm25_ids = ["old"]
    vs._rebuild_bm25()
    assert vs._bm25 is None
    assert vs._bm25_ids == []


def test_rebuild_bm25_populates_bm25_ids_from_collection():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.count.return_value = 2
    vs.collection.get.return_value = {
        "ids": ["c1", "c2"],
        "documents": ["machine learning", "deep learning neural"],
    }
    vs._bm25 = None
    vs._bm25_ids = []
    vs._rebuild_bm25()
    assert vs._bm25_ids == ["c1", "c2"]


def test_rebuild_bm25_creates_bm25_okapi_instance():
    from app.vector_store import VectorStore
    from rank_bm25 import BM25Okapi
    from unittest.mock import MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.count.return_value = 1
    vs.collection.get.return_value = {
        "ids": ["c1"],
        "documents": ["supervised learning"],
    }
    vs._bm25 = None
    vs._bm25_ids = []
    vs._rebuild_bm25()
    assert isinstance(vs._bm25, BM25Okapi)


def test_rebuild_bm25_uses_tokenized_corpus():
    from app.vector_store import VectorStore, _tokenize
    from rank_bm25 import BM25Okapi
    from unittest.mock import MagicMock, patch
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.count.return_value = 1
    vs.collection.get.return_value = {
        "ids": ["c1"],
        "documents": ["Hello, World!"],
    }
    vs._bm25 = None
    vs._bm25_ids = []
    captured = {}
    original_bm25 = BM25Okapi
    with patch("app.vector_store.BM25Okapi", side_effect=lambda corpus: captured.update({"corpus": corpus}) or original_bm25(corpus)):
        vs._rebuild_bm25()
    assert captured["corpus"] == [_tokenize("Hello, World!")]


# ── VectorStore.reset() unit tests ───────────────────────

def test_reset_deletes_existing_collection():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.client = MagicMock()
    new_collection = MagicMock()
    vs.client.get_or_create_collection.return_value = new_collection
    vs.collection = MagicMock()
    vs._bm25 = MagicMock()
    vs._bm25_ids = ["x"]
    vs.reset()
    vs.client.delete_collection.assert_called_once_with("documents")


def test_reset_creates_new_collection_with_cosine_space():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.client = MagicMock()
    new_collection = MagicMock()
    vs.client.get_or_create_collection.return_value = new_collection
    vs.collection = MagicMock()
    vs._bm25 = MagicMock()
    vs._bm25_ids = []
    vs.reset()
    call_kwargs = vs.client.get_or_create_collection.call_args[1]
    assert call_kwargs["metadata"] == {"hnsw:space": "cosine"}


def test_reset_clears_bm25_state():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.client = MagicMock()
    vs.client.get_or_create_collection.return_value = MagicMock()
    vs.collection = MagicMock()
    vs._bm25 = MagicMock()
    vs._bm25_ids = ["was", "populated"]
    vs.reset()
    assert vs._bm25 is None
    assert vs._bm25_ids == []


def test_reset_replaces_collection_reference():
    from app.vector_store import VectorStore
    from unittest.mock import MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.client = MagicMock()
    new_collection = MagicMock(name="new_collection")
    vs.client.get_or_create_collection.return_value = new_collection
    old_collection = MagicMock(name="old_collection")
    vs.collection = old_collection
    vs._bm25 = None
    vs._bm25_ids = []
    vs.reset()
    assert vs.collection is new_collection


# ── _tokenize() unit tests ────────────────────────────────

def test_tokenize_lowercases_text():
    from app.vector_store import _tokenize
    result = _tokenize("Supervised Learning")
    assert "supervised" in result
    assert "learning" in result


def test_tokenize_splits_into_words():
    from app.vector_store import _tokenize
    assert _tokenize("hello world") == ["hello", "world"]


def test_tokenize_strips_punctuation():
    from app.vector_store import _tokenize
    result = _tokenize("hello, world!")
    assert "hello" in result
    assert "world" in result


def test_tokenize_empty_string_returns_empty():
    from app.vector_store import _tokenize
    assert _tokenize("") == []


def test_tokenize_includes_numeric_tokens():
    from app.vector_store import _tokenize
    result = _tokenize("top 5 results")
    assert "5" in result


# ── VectorStore._bm25_query() unit tests ─────────────────

def test_bm25_query_no_index_returns_empty():
    from app.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    vs._bm25 = None
    vs._bm25_ids = []
    assert vs._bm25_query("anything", top_k=5) == {}


def test_bm25_query_scores_in_unit_range():
    from app.vector_store import VectorStore
    from rank_bm25 import BM25Okapi
    vs = VectorStore.__new__(VectorStore)
    corpus = [["machine", "learning", "model"], ["learning", "algorithm"], ["deep", "network"]]
    vs._bm25 = BM25Okapi(corpus)
    vs._bm25_ids = ["c1", "c2", "c3"]
    result = vs._bm25_query("learning model", top_k=5)
    for score in result.values():
        assert 0.0 <= score <= 1.0


def test_bm25_query_best_match_normalized_to_one():
    from app.vector_store import VectorStore
    from rank_bm25 import BM25Okapi
    vs = VectorStore.__new__(VectorStore)
    # Needs >= 3 docs so BM25 IDF is positive for a term in only 1 doc.
    # With 2 docs and df=1: IDF = log((2-1+0.5)/(1+0.5)) = log(1) = 0.
    corpus = [["machine", "learning"], ["deep", "network"], ["natural", "language"]]
    vs._bm25 = BM25Okapi(corpus)
    vs._bm25_ids = ["c1", "c2", "c3"]
    result = vs._bm25_query("machine", top_k=5)
    # Only c1 matches "machine"; its raw score equals the max, so normalized to 1.0.
    assert result.get("c1") == 1.0


def test_bm25_query_non_matching_query_returns_empty():
    from app.vector_store import VectorStore
    from rank_bm25 import BM25Okapi
    vs = VectorStore.__new__(VectorStore)
    corpus = [["hello", "world"], ["foo", "bar"]]
    vs._bm25 = BM25Okapi(corpus)
    vs._bm25_ids = ["c1", "c2"]
    assert vs._bm25_query("zzz_nonexistent_term", top_k=5) == {}


# ── VectorStore hybrid fusion unit tests ─────────────────

def test_query_fuses_dense_score_with_alpha():
    from app.vector_store import VectorStore, HYBRID_ALPHA
    from unittest.mock import patch, MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.get.return_value = {
        "ids": ["c1"], "documents": ["text"], "metadatas": [{}],
    }
    with patch.object(vs, "_dense_query", return_value={"c1": 0.8}), \
         patch.object(vs, "_bm25_query", return_value={}):
        result = vs.query("test", top_k=1)
    assert len(result) == 1
    assert result[0]["score"] == round(HYBRID_ALPHA * 0.8, 4)


def test_query_fuses_bm25_score_with_one_minus_alpha():
    from app.vector_store import VectorStore, HYBRID_ALPHA
    from unittest.mock import patch, MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.get.return_value = {
        "ids": ["c1"], "documents": ["text"], "metadatas": [{}],
    }
    with patch.object(vs, "_dense_query", return_value={}), \
         patch.object(vs, "_bm25_query", return_value={"c1": 0.6}):
        result = vs.query("test", top_k=1)
    assert len(result) == 1
    assert result[0]["score"] == round((1 - HYBRID_ALPHA) * 0.6, 4)


def test_query_combines_both_scores():
    from app.vector_store import VectorStore, HYBRID_ALPHA
    from unittest.mock import patch, MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.get.return_value = {
        "ids": ["c1"], "documents": ["text"], "metadatas": [{}],
    }
    with patch.object(vs, "_dense_query", return_value={"c1": 0.8}), \
         patch.object(vs, "_bm25_query", return_value={"c1": 0.6}):
        result = vs.query("test", top_k=1)
    expected = round(HYBRID_ALPHA * 0.8 + (1 - HYBRID_ALPHA) * 0.6, 4)
    assert result[0]["score"] == expected


def test_query_exposes_dense_and_bm25_scores_separately():
    from app.vector_store import VectorStore
    from unittest.mock import patch, MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.get.return_value = {
        "ids": ["c1"], "documents": ["text"], "metadatas": [{}],
    }
    with patch.object(vs, "_dense_query", return_value={"c1": 0.8}), \
         patch.object(vs, "_bm25_query", return_value={"c1": 0.5}):
        result = vs.query("test", top_k=1)
    assert result[0]["dense_score"] == 0.8
    assert result[0]["bm25_score"] == 0.5


def test_query_sorts_results_by_fused_score_descending():
    from app.vector_store import VectorStore
    from unittest.mock import patch, MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.get.return_value = {
        "ids": ["c1", "c2"],
        "documents": ["text1", "text2"],
        "metadatas": [{}, {}],
    }
    # c2 has higher dense but c1 has higher BM25 — combined order depends on alpha
    with patch.object(vs, "_dense_query", return_value={"c1": 0.9, "c2": 0.3}), \
         patch.object(vs, "_bm25_query", return_value={"c1": 0.1, "c2": 0.2}):
        result = vs.query("test", top_k=2)
    assert len(result) == 2
    assert result[0]["score"] >= result[1]["score"]


def test_query_respects_top_k():
    from app.vector_store import VectorStore
    from unittest.mock import patch, MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.get.return_value = {
        "ids": ["c1", "c2"],
        "documents": ["t1", "t2"],
        "metadatas": [{}, {}],
    }
    with patch.object(vs, "_dense_query", return_value={"c1": 0.9, "c2": 0.8, "c3": 0.7}), \
         patch.object(vs, "_bm25_query", return_value={}):
        result = vs.query("test", top_k=2)
    assert len(result) <= 2


def test_query_returns_empty_when_no_results():
    from app.vector_store import VectorStore
    from unittest.mock import patch, MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    with patch.object(vs, "_dense_query", return_value={}), \
         patch.object(vs, "_bm25_query", return_value={}):
        result = vs.query("test", top_k=5)
    assert result == []


def test_query_result_has_required_fields():
    from app.vector_store import VectorStore
    from unittest.mock import patch, MagicMock
    vs = VectorStore.__new__(VectorStore)
    vs.collection = MagicMock()
    vs.collection.get.return_value = {
        "ids": ["c1"], "documents": ["some text"], "metadatas": [{"title": "Doc"}],
    }
    with patch.object(vs, "_dense_query", return_value={"c1": 0.7}), \
         patch.object(vs, "_bm25_query", return_value={"c1": 0.4}):
        result = vs.query("test", top_k=1)
    assert len(result) == 1
    for field in ("chunk_id", "text", "metadata", "score", "dense_score", "bm25_score"):
        assert field in result[0]


# ── /query/stream route unit tests ───────────────────────────

_STREAM_FAKE_CHUNKS = [
    {
        "chunk_id": "c1",
        "text": "Supervised learning uses labeled data.",
        "metadata": {"title": "ML Basics"},
        "score": 0.9,
        "dense_score": 0.9,
        "bm25_score": 0.5,
    }
]


async def _fake_stream_tokens(query, chunks, max_tokens=512):
    yield _json.dumps({"token": "hello"})
    yield _json.dumps({"token": " world"})
    yield "[DONE]"


def test_stream_empty_index_returns_400():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 0
        r = client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    assert r.status_code == 400


def test_stream_no_chunks_returns_404():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = []
        r = client.post("/query/stream", json={"query": "nonsense query xyzzy", "top_k": 3})
    assert r.status_code == 404


def test_stream_content_type_is_event_stream():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer_stream = _fake_stream_tokens
        r = client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    assert "text/event-stream" in r.headers["content-type"]


def test_stream_events_use_sse_format():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer_stream = _fake_stream_tokens
        r = client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    assert r.status_code == 200
    assert "data: " in r.text
    assert "\n\n" in r.text


def test_stream_yields_done_sentinel():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer_stream = _fake_stream_tokens
        r = client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    assert "data: [DONE]" in r.text


def test_stream_token_events_are_json_with_token_key():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer_stream = _fake_stream_tokens
        r = client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    token_lines = [
        line for line in r.text.split("\n")
        if line.startswith("data: ") and line.strip() != "data: [DONE]"
    ]
    assert len(token_lines) == 2
    for line in token_lines:
        parsed = _json.loads(line[len("data: "):])
        assert "token" in parsed


def test_stream_calls_reranker_when_rerank_true():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen, \
         patch("app.main.rerank") as mock_rerank:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_rerank.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer_stream = _fake_stream_tokens
        client.post("/query/stream", json={"query": "What is ML?", "top_k": 3, "rerank": True})
    mock_rerank.assert_called_once()


def test_stream_skips_reranker_when_rerank_false():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen, \
         patch("app.main.rerank") as mock_rerank:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer_stream = _fake_stream_tokens
        client.post("/query/stream", json={"query": "What is ML?", "top_k": 3, "rerank": False})
    mock_rerank.assert_not_called()


# ── /query/stream DB logging unit tests ──────────────────────

def _override_db(mock_db):
    from app.database import get_db
    from app.main import app
    app.dependency_overrides[get_db] = lambda: mock_db
    return app


def _clear_overrides():
    from app.main import app
    app.dependency_overrides.clear()


def test_stream_logs_to_db_after_completion():
    from unittest.mock import patch, MagicMock
    mock_db = MagicMock()
    _override_db(mock_db)
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer_stream = _fake_stream_tokens
            r = client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    finally:
        _clear_overrides()
    assert r.status_code == 200
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


def test_stream_log_accumulates_full_answer():
    from unittest.mock import patch, MagicMock
    mock_db = MagicMock()
    _override_db(mock_db)
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer_stream = _fake_stream_tokens
            client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    finally:
        _clear_overrides()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.answer == "hello world"


def test_stream_log_stores_query():
    from unittest.mock import patch, MagicMock
    mock_db = MagicMock()
    _override_db(mock_db)
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer_stream = _fake_stream_tokens
            client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    finally:
        _clear_overrides()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.query == "What is ML?"


def test_stream_log_stores_retrieved_ids():
    from unittest.mock import patch, MagicMock
    mock_db = MagicMock()
    _override_db(mock_db)
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer_stream = _fake_stream_tokens
            client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    finally:
        _clear_overrides()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.retrieved_ids == ["c1"]


def test_stream_log_stores_faithfulness():
    from unittest.mock import patch, MagicMock
    mock_db = MagicMock()
    _override_db(mock_db)
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer_stream = _fake_stream_tokens
            client.post("/query/stream", json={"query": "What is ML?", "top_k": 3})
    finally:
        _clear_overrides()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.faithfulness is not None
    assert 0.0 <= log_obj.faithfulness <= 1.0


def test_stream_empty_tokens_logs_fallback_answer():
    from unittest.mock import patch, MagicMock
    mock_db = MagicMock()
    _override_db(mock_db)

    async def _empty_stream(query, chunks, max_tokens=512):
        yield "[DONE]"

    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer_stream = _empty_stream
            client.post("/query/stream", json={"query": "q", "top_k": 3})
    finally:
        _clear_overrides()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.answer == "(empty stream)"
    assert log_obj.faithfulness is None


# ── /health DB probe unit tests ───────────────────────────────

def test_health_returns_200_when_db_ok():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        r = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["db"] == "ok"


def test_health_returns_503_when_db_unreachable():
    from unittest.mock import patch, MagicMock
    from sqlalchemy.exc import OperationalError
    from app.database import get_db

    mock_db = MagicMock()
    mock_db.execute.side_effect = OperationalError("could not connect", None, None)
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        r = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert r.json()["db"] == "unreachable"


def test_health_includes_chunks_indexed():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs:
            mock_vs.count.return_value = 42
            r = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert r.json()["chunks_indexed"] == 42


def test_health_includes_model_loaded():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.generator") as mock_gen:
            mock_gen.model = object()
            r = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert r.json()["model_loaded"] is True


def test_health_model_loaded_false_when_not_loaded():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.generator") as mock_gen:
            mock_gen.model = None
            r = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert r.json()["model_loaded"] is False


# ── answer_relevance DB persistence unit tests ────────────────

def test_query_stores_answer_relevance_in_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            r = client.post("/query", json={"query": "supervised learning", "top_k": 1})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.answer_relevance is not None
    assert 0.0 <= log_obj.answer_relevance <= 1.0


def test_query_answer_relevance_matches_evaluator():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    from app.evaluator import answer_relevance as _ar
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    query = "supervised learning classification"
    answer = "Supervised learning solves classification problems."
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer.return_value = answer
            client.post("/query", json={"query": query, "top_k": 1})
    finally:
        app.dependency_overrides.clear()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.answer_relevance == _ar(query, answer)


def test_query_eval_stores_answer_relevance_in_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            r = client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.answer_relevance is not None
    assert 0.0 <= log_obj.answer_relevance <= 1.0


def test_stream_stores_answer_relevance_in_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    _override_db(mock_db)
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer_stream = _fake_stream_tokens
            client.post("/query/stream", json={"query": "supervised learning", "top_k": 3})
    finally:
        _clear_overrides()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.answer_relevance is not None
    assert 0.0 <= log_obj.answer_relevance <= 1.0


def test_stream_empty_tokens_stores_null_answer_relevance():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    _override_db(mock_db)

    async def _empty_stream_2(query, chunks, max_tokens=512):
        yield "[DONE]"

    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer_stream = _empty_stream_2
            client.post("/query/stream", json={"query": "q", "top_k": 3})
    finally:
        _clear_overrides()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.answer_relevance is None


# ── /eval/history answer_relevance field tests ────────────────

def test_eval_history_includes_answer_relevance_field():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    from app.models import QueryLog as QL
    mock_log = MagicMock(spec=QL)
    mock_log.query = "What is ML?"
    mock_log.answer = "Machine learning is a field of AI."
    mock_log.faithfulness = 0.8
    mock_log.answer_relevance = 0.75
    mock_log.hit_rate = None
    mock_log.mrr = None
    mock_log.ndcg = None
    mock_log.created_at = __import__("datetime").datetime(2026, 1, 1)
    mock_db = MagicMock()
    mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_log]
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        r = client.get("/eval/history?limit=1")
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert "answer_relevance" in rows[0]
    assert rows[0]["answer_relevance"] == 0.75


def test_eval_history_answer_relevance_can_be_null():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    from app.models import QueryLog as QL
    mock_log = MagicMock(spec=QL)
    mock_log.query = "What is ML?"
    mock_log.answer = "Machine learning is a field of AI."
    mock_log.faithfulness = None
    mock_log.answer_relevance = None
    mock_log.hit_rate = None
    mock_log.mrr = None
    mock_log.ndcg = None
    mock_log.created_at = __import__("datetime").datetime(2026, 1, 1)
    mock_db = MagicMock()
    mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_log]
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        r = client.get("/eval/history?limit=1")
    finally:
        app.dependency_overrides.clear()
    rows = r.json()
    assert "answer_relevance" in rows[0]
    assert rows[0]["answer_relevance"] is None


# ── /query response eval field completeness tests ─────────────

def test_query_response_eval_includes_answer_relevance():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            r = client.post("/query", json={"query": "supervised learning", "top_k": 1})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    data = r.json()
    assert "eval" in data
    assert "answer_relevance" in data["eval"]
    assert data["eval"]["answer_relevance"] is not None


def test_query_response_eval_answer_relevance_is_bounded():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.count.return_value = 5
            mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            r = client.post("/query", json={"query": "supervised learning", "top_k": 1})
    finally:
        app.dependency_overrides.clear()
    score = r.json()["eval"]["answer_relevance"]
    assert 0.0 <= score <= 1.0


# ── /query/eval route unit tests ─────────────────────────────

_EVAL_FAKE_CHUNKS = [
    {
        "chunk_id": "c1",
        "text": "Supervised learning uses labeled data.",
        "metadata": {"title": "ML Basics"},
        "score": 0.9,
        "dense_score": 0.9,
        "bm25_score": 0.5,
    }
]


def test_query_eval_no_chunks_returns_404():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs:
            mock_vs.query.return_value = []
            r = client.post("/query/eval", json={
                "query": "What is ML?",
                "relevant_doc_ids": ["c1"],
                "top_k": 3,
            })
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 404


def test_query_eval_response_has_all_eval_fields():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            r = client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    eval_fields = r.json()["eval"]
    for field in ("hit_rate", "mrr", "ndcg", "faithfulness", "answer_relevance"):
        assert field in eval_fields, f"missing eval field: {field}"


def test_query_eval_response_metrics_are_bounded():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            r = client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    for key, val in r.json()["eval"].items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


def test_query_eval_response_contains_query_and_answer():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            r = client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    data = r.json()
    assert data["query"] == "supervised learning"
    assert data["answer"] == "Supervised learning uses labeled data."
    assert "chunks" in data


def test_query_eval_stores_hit_rate_in_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.hit_rate is not None
    assert 0.0 <= log_obj.hit_rate <= 1.0


def test_query_eval_stores_mrr_in_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.mrr is not None
    assert 0.0 <= log_obj.mrr <= 1.0


def test_query_eval_stores_ndcg_in_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.ndcg is not None
    assert 0.0 <= log_obj.ndcg <= 1.0


def test_query_eval_stores_faithfulness_in_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.faithfulness is not None
    assert 0.0 <= log_obj.faithfulness <= 1.0


def test_query_eval_ndcg_matches_evaluator():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    from app.evaluator import ndcg_at_k
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    log_obj = mock_db.add.call_args[0][0]
    expected = ndcg_at_k(["c1"], ["c1"], k=1)
    assert log_obj.ndcg == expected


def test_query_eval_hit_rate_matches_evaluator():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    from app.evaluator import hit_rate
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.hit_rate == hit_rate(["c1"], ["c1"])


def test_query_eval_calls_reranker_when_rerank_true():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen, \
             patch("app.main.rerank") as mock_rerank:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_rerank.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
                "rerank": True,
            })
    finally:
        app.dependency_overrides.clear()
    mock_rerank.assert_called_once()


def test_query_eval_skips_reranker_when_rerank_false():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen, \
             patch("app.main.rerank") as mock_rerank:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
                "rerank": False,
            })
    finally:
        app.dependency_overrides.clear()
    mock_rerank.assert_not_called()


def test_query_eval_logs_retrieved_ids_in_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    log_obj = mock_db.add.call_args[0][0]
    assert log_obj.retrieved_ids == ["c1"]


def test_query_eval_commits_to_db():
    from unittest.mock import patch, MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.main.vector_store") as mock_vs, \
             patch("app.main.generator") as mock_gen:
            mock_vs.query.return_value = _EVAL_FAKE_CHUNKS
            mock_gen.answer.return_value = "Supervised learning uses labeled data."
            client.post("/query/eval", json={
                "query": "supervised learning",
                "relevant_doc_ids": ["c1"],
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


# ── /eval/summary unit tests ──────────────────────────────────

import datetime as _dt


def _make_log(**kwargs):
    from unittest.mock import MagicMock
    from app.models import QueryLog as QL
    log = MagicMock(spec=QL)
    defaults = dict(
        query="What is ML?",
        answer="Machine learning is a field of AI.",
        faithfulness=None,
        answer_relevance=None,
        hit_rate=None,
        mrr=None,
        ndcg=None,
        created_at=_dt.datetime(2026, 1, 1),
    )
    defaults.update(kwargs)
    for k, v in defaults.items():
        setattr(log, k, v)
    return log


def _summary_with_logs(logs):
    from unittest.mock import MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = logs
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        r = client.get("/eval/summary")
    finally:
        app.dependency_overrides.clear()
    return r


def test_eval_summary_empty_db_returns_message():
    r = _summary_with_logs([])
    assert r.status_code == 200
    assert "message" in r.json()


def test_eval_summary_total_queries_count():
    logs = [_make_log(faithfulness=0.8), _make_log(faithfulness=0.6)]
    r = _summary_with_logs(logs)
    assert r.json()["total_queries"] == 2


def test_eval_summary_avg_faithfulness_computed():
    logs = [_make_log(faithfulness=0.8), _make_log(faithfulness=0.6)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_faithfulness"] == round((0.8 + 0.6) / 2, 4)


def test_eval_summary_avg_faithfulness_excludes_none():
    logs = [_make_log(faithfulness=0.8), _make_log(faithfulness=None)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_faithfulness"] == 0.8


def test_eval_summary_avg_faithfulness_all_none_returns_none():
    logs = [_make_log(faithfulness=None), _make_log(faithfulness=None)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_faithfulness"] is None


def test_eval_summary_avg_hit_rate_computed():
    logs = [_make_log(hit_rate=1.0), _make_log(hit_rate=0.5)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_hit_rate"] == round((1.0 + 0.5) / 2, 4)


def test_eval_summary_avg_hit_rate_excludes_none():
    logs = [_make_log(hit_rate=0.75), _make_log(hit_rate=None)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_hit_rate"] == 0.75


def test_eval_summary_avg_mrr_computed():
    logs = [_make_log(mrr=1.0), _make_log(mrr=0.5)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_mrr"] == round((1.0 + 0.5) / 2, 4)


def test_eval_summary_avg_mrr_excludes_none():
    logs = [_make_log(mrr=None), _make_log(mrr=0.3333)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_mrr"] == 0.3333


def test_eval_summary_avg_ndcg_computed():
    logs = [_make_log(ndcg=1.0), _make_log(ndcg=0.6309)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_ndcg"] == round((1.0 + 0.6309) / 2, 4)


def test_eval_summary_avg_ndcg_excludes_none():
    logs = [_make_log(ndcg=0.8), _make_log(ndcg=None)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_ndcg"] == 0.8


def test_eval_summary_avg_ndcg_all_none_returns_none():
    logs = [_make_log(ndcg=None)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_ndcg"] is None


def test_eval_summary_avg_answer_relevance_uses_stored_value():
    logs = [_make_log(
        query="overfitting regularization",
        answer="Overfitting can be reduced by regularization.",
        answer_relevance=0.5,
    )]
    r = _summary_with_logs(logs)
    assert r.json()["avg_answer_relevance"] == 0.5


def test_eval_summary_avg_answer_relevance_falls_back_to_computed():
    from app.evaluator import answer_relevance as _ar
    query = "supervised learning"
    answer = "Supervised learning uses labeled data."
    logs = [_make_log(query=query, answer=answer, answer_relevance=None)]
    r = _summary_with_logs(logs)
    assert r.json()["avg_answer_relevance"] == _ar(query, answer)


def test_eval_summary_avg_answer_relevance_mixes_stored_and_fallback():
    from app.evaluator import answer_relevance as _ar
    query = "gradient descent"
    answer = "Gradient descent minimizes loss."
    computed = _ar(query, answer)
    logs = [
        _make_log(query=query, answer=answer, answer_relevance=0.9),
        _make_log(query=query, answer=answer, answer_relevance=None),
    ]
    r = _summary_with_logs(logs)
    expected = round((0.9 + computed) / 2, 4)
    assert r.json()["avg_answer_relevance"] == expected


def test_eval_summary_single_log_returns_its_own_values():
    logs = [_make_log(faithfulness=0.7, hit_rate=1.0, mrr=1.0, ndcg=1.0, answer_relevance=0.6)]
    r = _summary_with_logs(logs)
    data = r.json()
    assert data["total_queries"] == 1
    assert data["avg_faithfulness"] == 0.7
    assert data["avg_hit_rate"] == 1.0
    assert data["avg_mrr"] == 1.0
    assert data["avg_ndcg"] == 1.0
    assert data["avg_answer_relevance"] == 0.6


def test_eval_summary_averages_rounded_to_four_decimals():
    logs = [_make_log(faithfulness=1 / 3)]
    r = _summary_with_logs(logs)
    avg = r.json()["avg_faithfulness"]
    assert avg == round(1 / 3, 4)
    assert len(str(avg).split(".")[-1]) <= 4


# ── /query (POST) core behavior unit tests ───────────────────

def test_query_empty_index_returns_400():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 0
        r = client.post("/query", json={"query": "What is ML?", "top_k": 3})
    assert r.status_code == 400


def test_query_no_candidates_returns_404():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = []
        r = client.post("/query", json={"query": "nonsense xyzzy 99999", "top_k": 3})
    assert r.status_code == 404


def test_query_response_has_all_top_level_fields():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Supervised learning uses labeled data."
        r = client.post("/query", json={"query": "supervised learning", "top_k": 1})
    assert r.status_code == 200
    data = r.json()
    for field in ("query", "answer", "chunks", "reranked", "citations", "eval"):
        assert field in data, f"missing top-level field: {field}"


def test_query_eval_field_includes_faithfulness():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Supervised learning uses labeled data."
        r = client.post("/query", json={"query": "supervised learning", "top_k": 1})
    assert "faithfulness" in r.json()["eval"]


def test_query_eval_faithfulness_is_bounded():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Supervised learning uses labeled data."
        r = client.post("/query", json={"query": "supervised learning", "top_k": 1})
    score = r.json()["eval"]["faithfulness"]
    assert score is None or 0.0 <= score <= 1.0


def test_query_reranked_false_when_rerank_not_requested():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Some answer."
        r = client.post("/query", json={"query": "q", "top_k": 1, "rerank": False})
    assert r.json()["reranked"] is False


def test_query_reranked_true_when_rerank_requested():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen, \
         patch("app.main.rerank") as mock_rerank:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_rerank.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Some answer."
        r = client.post("/query", json={"query": "q", "top_k": 1, "rerank": True})
    assert r.json()["reranked"] is True


def test_query_calls_reranker_when_rerank_true():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen, \
         patch("app.main.rerank") as mock_rerank:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_rerank.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Some answer."
        client.post("/query", json={"query": "q", "top_k": 1, "rerank": True})
    mock_rerank.assert_called_once()


def test_query_skips_reranker_when_rerank_false():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen, \
         patch("app.main.rerank") as mock_rerank:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Some answer."
        client.post("/query", json={"query": "q", "top_k": 1, "rerank": False})
    mock_rerank.assert_not_called()


def test_query_citations_field_is_list():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Supervised learning uses labeled data."
        r = client.post("/query", json={"query": "supervised learning", "top_k": 1})
    assert isinstance(r.json()["citations"], list)


def test_query_response_query_echoes_request():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Some answer."
        r = client.post("/query", json={"query": "What is supervised learning?", "top_k": 1})
    assert r.json()["query"] == "What is supervised learning?"


def test_query_response_answer_matches_generator_output():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "The specific answer text here."
        r = client.post("/query", json={"query": "q", "top_k": 1})
    assert r.json()["answer"] == "The specific answer text here."


def test_query_chunks_field_is_list():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.generator") as mock_gen:
        mock_vs.count.return_value = 5
        mock_vs.query.return_value = _STREAM_FAKE_CHUNKS
        mock_gen.answer.return_value = "Some answer."
        r = client.post("/query", json={"query": "q", "top_k": 1})
    assert isinstance(r.json()["chunks"], list)


# ── /index/stats unit tests ───────────────────────────────────

def test_index_stats_returns_200():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 7
        r = client.get("/index/stats")
    assert r.status_code == 200


def test_index_stats_has_all_required_fields():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 7
        r = client.get("/index/stats")
    data = r.json()
    for field in ("total_chunks", "embed_model", "gen_model", "reranker_model", "hybrid_alpha"):
        assert field in data, f"missing field: {field}"


def test_index_stats_total_chunks_from_vector_store():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 42
        r = client.get("/index/stats")
    assert r.json()["total_chunks"] == 42


def test_index_stats_hybrid_alpha_is_float():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 5
        r = client.get("/index/stats")
    assert isinstance(r.json()["hybrid_alpha"], float)


def test_index_stats_hybrid_alpha_in_valid_range():
    from unittest.mock import patch
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.count.return_value = 5
        r = client.get("/index/stats")
    alpha = r.json()["hybrid_alpha"]
    assert 0.0 <= alpha <= 1.0


# ── /eval/history unit tests ──────────────────────────────────

def _history_with_logs(logs, limit=20):
    from unittest.mock import MagicMock
    from app.database import get_db
    mock_db = MagicMock()
    mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = logs
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        r = client.get(f"/eval/history?limit={limit}")
    finally:
        app.dependency_overrides.clear()
    return r


def test_eval_history_returns_empty_list_when_no_logs():
    r = _history_with_logs([])
    assert r.status_code == 200
    assert r.json() == []


def test_eval_history_has_all_required_fields():
    log = _make_log(faithfulness=0.7, hit_rate=1.0, mrr=0.5, ndcg=0.8, answer_relevance=0.6)
    r = _history_with_logs([log])
    row = r.json()[0]
    for field in ("query", "answer", "faithfulness", "answer_relevance", "hit_rate", "mrr", "ndcg", "created_at"):
        assert field in row, f"missing field: {field}"


def test_eval_history_short_answer_not_truncated():
    short_answer = "A short answer."
    log = _make_log(answer=short_answer)
    r = _history_with_logs([log])
    assert r.json()[0]["answer"] == short_answer


def test_eval_history_truncates_long_answers():
    long_answer = "x" * 201
    log = _make_log(answer=long_answer)
    r = _history_with_logs([log])
    assert r.json()[0]["answer"] == "x" * 200 + "..."


def test_eval_history_answer_exactly_200_chars_not_truncated():
    answer_200 = "y" * 200
    log = _make_log(answer=answer_200)
    r = _history_with_logs([log])
    assert r.json()[0]["answer"] == answer_200


def test_eval_history_truncated_answer_ends_with_ellipsis():
    log = _make_log(answer="z" * 300)
    r = _history_with_logs([log])
    assert r.json()[0]["answer"].endswith("...")


def test_eval_history_respects_limit_parameter():
    from unittest.mock import MagicMock
    from app.database import get_db
    logs = [_make_log() for _ in range(5)]
    mock_db = MagicMock()
    mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = logs
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        client.get("/eval/history?limit=5")
    finally:
        app.dependency_overrides.clear()
    mock_db.query.return_value.order_by.return_value.limit.assert_called_once_with(5)


def test_eval_history_created_at_is_string():
    log = _make_log()
    r = _history_with_logs([log])
    assert isinstance(r.json()[0]["created_at"], str)


def test_eval_history_null_metric_fields_preserved():
    log = _make_log(faithfulness=None, hit_rate=None, mrr=None, ndcg=None, answer_relevance=None)
    r = _history_with_logs([log])
    row = r.json()[0]
    assert row["faithfulness"] is None
    assert row["hit_rate"] is None
    assert row["mrr"] is None
    assert row["ndcg"] is None
    assert row["answer_relevance"] is None
