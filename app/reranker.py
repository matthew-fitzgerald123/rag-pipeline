"""
Cross-encoder reranker.

Takes the top-K candidates from hybrid retrieval and scores each
(query, chunk) pair through a cross-encoder. Returns results re-sorted
by cross-encoder score, replacing the original fused score.

Set RERANKER_MODEL to override the default model.
Set RERANKER_TOP_K to control how many candidates are passed to the
cross-encoder (default 20); the final top_k is returned after reranking.
"""
from __future__ import annotations

import logging
import os
import threading

from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "20"))

_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                log.info("Loading cross-encoder: %s", RERANKER_MODEL)
                _model = CrossEncoder(RERANKER_MODEL)
                log.info("Cross-encoder ready")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Rerank candidates using the cross-encoder.

    candidates: list of dicts with at least 'text' and 'chunk_id' keys.
    Returns the top_k highest-scoring candidates, each annotated with
    a 'rerank_score' field.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = round(float(scores[i]), 4)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]
