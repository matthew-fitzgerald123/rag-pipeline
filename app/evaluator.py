from __future__ import annotations
import re
from collections import Counter
from math import log2

def hit_rate(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return round(hits / len(relevant_ids), 4)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> float:
    """Normalized discounted cumulative gain over the retrieved ranking.

    Uses binary relevance: a retrieved chunk gains 1 if it is in relevant_ids,
    else 0. Each gain is discounted by log2(rank + 1), so a relevant chunk found
    at rank 1 is worth more than the same chunk found at rank 5. The result is
    normalized by the ideal ranking (all relevant chunks first), giving a score
    in [0, 1] that rewards putting relevant results near the top, not just
    retrieving them at all like hit rate does."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    ranked = retrieved_ids if k is None else retrieved_ids[:k]

    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, rid in enumerate(ranked, start=1)
        if rid in relevant_set
    )

    ideal_hits = min(len(relevant_set), len(ranked)) if k is not None else len(relevant_set)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)

def faithfulness(answer: str, context_chunks: list[dict]) -> float:
    context_text = " ".join(c["text"] for c in context_chunks).lower()
    context_tokens = set(re.findall(r"\w+", context_text))

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        tokens = set(re.findall(r"\w+", sentence.lower()))
        stopwords = {"the","a","an","is","are","was","were","in","on","at","to","of","and","or","it","this","that","i","you","we","they"}
        tokens -= stopwords
        if not tokens:
            continue
        overlap = tokens & context_tokens
        if len(overlap) / len(tokens) > 0.4:
            supported += 1

    return round(supported / len(sentences), 4)

def answer_relevance(query: str, answer: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    stopwords = {"the","a","an","is","are","was","were","in","on","at","to","of","and","or","it","this","that","i","you","we","they"}
    query_tokens -= stopwords
    if not query_tokens:
        return 0.0
    overlap = query_tokens & answer_tokens
    return round(len(overlap) / len(query_tokens), 4)
