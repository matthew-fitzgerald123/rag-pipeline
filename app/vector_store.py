from __future__ import annotations
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.chunker import Chunk
from dotenv import load_dotenv
import os

load_dotenv()

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

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        embedding = self.embedder.encode(
            [query_text],
            normalize_embeddings=True,
        ).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk_id": results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    round(1 - results["distances"][0][i], 4),
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

vector_store = VectorStore()
