from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict

def chunk_document(
    doc_id: str,
    text: str,
    metadata: dict,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            ordinal = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{ordinal}",
                doc_id=doc_id,
                text=chunk_text,
                metadata={**metadata, "chunk_index": ordinal, "doc_id": doc_id},
            ))
        start += chunk_size - overlap

    return chunks
