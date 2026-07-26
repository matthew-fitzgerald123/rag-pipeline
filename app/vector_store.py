from __future__ import annotations
import re
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from app.chunker import Chunk
from dotenv import load_dotenv
import os

load_dotenv()

HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))  # weight for dense score


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PATH", "./chroma_db"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(
            os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            device="cpu",
        )
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        n = self.collection.count()
        if n == 0:
            self._bm25 = None
            self._bm25_ids = []
            return
        all_docs = self.collection.get(include=["documents"])
        self._bm25_ids = all_docs["ids"]
        corpus = [_tokenize(t) for t in all_docs["documents"]]
        self._bm25 = BM25Okapi(corpus)

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.metadata for c in chunks],
        )
        self._rebuild_bm25()

    def _dense_query(self, query_text: str, top_k: int) -> dict[str, float]:
        if self.collection.count() == 0:
            return {}
        embedding = self.embedder.encode(
            [query_text], normalize_embeddings=True
        ).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k * 2, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        scores = {}
        for i, cid in enumerate(results["ids"][0]):
            scores[cid] = round(1 - results["distances"][0][i], 4)
        return scores

    def _bm25_query(self, query_text: str, top_k: int) -> dict[str, float]:
        if self._bm25 is None or not self._bm25_ids:
            return {}
        tokens = _tokenize(query_text)
        raw_scores = self._bm25.get_scores(tokens)
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        scored = [
            (cid, round(float(raw_scores[i]) / max_score, 4))
            for i, cid in enumerate(self._bm25_ids)
            if raw_scores[i] > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return dict(scored[: top_k * 2])

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        dense_scores = self._dense_query(query_text, top_k)
        bm25_scores  = self._bm25_query(query_text, top_k)

        all_ids = set(dense_scores) | set(bm25_scores)
        fused = {
            cid: HYBRID_ALPHA * dense_scores.get(cid, 0.0)
                 + (1 - HYBRID_ALPHA) * bm25_scores.get(cid, 0.0)
            for cid in all_ids
        }
        ranked_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]

        if not ranked_ids:
            return []

        fetched = self.collection.get(
            ids=ranked_ids,
            include=["documents", "metadatas"],
        )
        id_to_idx = {cid: i for i, cid in enumerate(fetched["ids"])}

        output = []
        for cid in ranked_ids:
            if cid not in id_to_idx:
                continue
            idx = id_to_idx[cid]
            output.append({
                "chunk_id":    cid,
                "text":        fetched["documents"][idx],
                "metadata":    fetched["metadatas"][idx],
                "score":       round(fused[cid], 4),
                "dense_score": round(dense_scores.get(cid, 0.0), 4),
                "bm25_score":  round(bm25_scores.get(cid, 0.0), 4),
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_ids = []


vector_store = VectorStore()
