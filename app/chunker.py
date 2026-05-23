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
    chunks = []
    start = 0
    idx = 0
    text = text.strip()

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                metadata={**metadata, "chunk_index": idx, "doc_id": doc_id},
            ))
        start += chunk_size - overlap
        idx += 1

    return chunks
