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
