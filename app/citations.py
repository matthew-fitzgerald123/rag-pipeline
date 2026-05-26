"""
Citation extraction.

Maps each sentence in the generated answer back to the chunk(s) that
best support it, using token-overlap as the grounding signal. Returns
a list of citation objects that the API attaches to the response.
"""
from __future__ import annotations

import re


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "of", "and", "or", "it", "this", "that", "i", "you", "we",
    "they", "be", "have", "has", "had", "do", "does", "did", "not",
    "with", "for", "as", "by", "from", "its", "their", "our",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower())) - _STOPWORDS


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def extract_citations(
    answer: str,
    chunks: list[dict],
    threshold: float = 0.3,
) -> list[dict]:
    """
    For each sentence in answer, find the chunk(s) with token overlap
    above threshold and return citation objects.

    Returns a list of:
      {
        "sentence": str,
        "citations": [{"chunk_id": str, "title": str, "overlap": float}]
      }
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    results = []

    for sentence in sentences:
        sent_tokens = _tokens(sentence)
        if not sent_tokens:
            continue

        grounded = []
        for chunk in chunks:
            chunk_tokens = _tokens(chunk["text"])
            ov = _overlap(sent_tokens, chunk_tokens)
            if ov >= threshold:
                grounded.append({
                    "chunk_id": chunk["chunk_id"],
                    "title":    chunk.get("metadata", {}).get("title", ""),
                    "overlap":  round(ov, 3),
                })

        grounded.sort(key=lambda x: x["overlap"], reverse=True)
        results.append({"sentence": sentence, "citations": grounded})

    return results
