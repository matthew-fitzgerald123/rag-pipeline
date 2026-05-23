from __future__ import annotations
import re
from collections import Counter

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
