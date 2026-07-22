from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Any, AsyncIterator
import os
from dotenv import load_dotenv

from sqlalchemy import text
from app.database import get_db, engine
from app.models import Base, QueryLog
from app.vector_store import vector_store
from app.generator import generator
from app.evaluator import hit_rate, mean_reciprocal_rank, ndcg_at_k, faithfulness, answer_relevance
from app.reranker import rerank, RERANKER_TOP_K
from app.citations import extract_citations

load_dotenv()
Base.metadata.create_all(bind=engine)

with engine.connect() as _conn:
    _conn.execute(text("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS ndcg FLOAT"))
    _conn.execute(text("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS answer_relevance FLOAT"))
    _conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    generator.load_model()
    print(f"Vector store: {vector_store.count()} chunks indexed")
    yield


app = FastAPI(title="RAG Pipeline", version="1.0.0", lifespan=lifespan)

# ── Query ─────────────────────────────────────────────────

class QueryReq(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = False

class EvalQueryReq(BaseModel):
    query: str
    relevant_doc_ids: list[str]
    top_k: int = 5
    rerank: bool = False

@app.post("/query", tags=["rag"])
def query(req: QueryReq, db: Session = Depends(get_db)):
    if vector_store.count() == 0:
        raise HTTPException(400, "No documents indexed. Run: make ingest")

    candidates = vector_store.query(req.query, top_k=max(req.top_k, RERANKER_TOP_K))
    if not candidates:
        raise HTTPException(404, "No relevant chunks found")

    chunks = rerank(req.query, candidates, req.top_k) if req.rerank else candidates[:req.top_k]

    answer = generator.answer(req.query, chunks)

    ar = answer_relevance(req.query, answer)
    log = QueryLog(
        query=req.query,
        answer=answer,
        retrieved_ids=[c["chunk_id"] for c in chunks],
        faithfulness=faithfulness(answer, chunks),
        answer_relevance=ar,
    )
    db.add(log)
    db.commit()

    return {
        "query":     req.query,
        "answer":    answer,
        "chunks":    chunks,
        "reranked":  req.rerank,
        "citations": extract_citations(answer, chunks),
        "eval": {
            "faithfulness":     log.faithfulness,
            "answer_relevance": ar,
        },
    }

@app.post("/query/stream", tags=["rag"])
async def query_stream(req: QueryReq, db: Session = Depends(get_db)):
    """Stream answer tokens as SSE events. Does not log to DB."""
    if vector_store.count() == 0:
        raise HTTPException(400, "No documents indexed -- run: make ingest")

    candidates = vector_store.query(req.query, top_k=max(req.top_k, RERANKER_TOP_K))
    if not candidates:
        raise HTTPException(404, "No relevant chunks found")

    chunks = rerank(req.query, candidates, req.top_k) if req.rerank else candidates[:req.top_k]

    async def event_generator() -> AsyncIterator[str]:
        async for payload in generator.answer_stream(req.query, chunks):
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/query/eval", tags=["rag"])
def query_with_eval(req: EvalQueryReq, db: Session = Depends(get_db)):
    candidates = vector_store.query(req.query, top_k=max(req.top_k, RERANKER_TOP_K))
    if not candidates:
        raise HTTPException(404, "No relevant chunks found")

    chunks = rerank(req.query, candidates, req.top_k) if req.rerank else candidates[:req.top_k]

    answer = generator.answer(req.query, chunks)
    retrieved_ids = [c["chunk_id"] for c in chunks]

    hr   = hit_rate(retrieved_ids, req.relevant_doc_ids)
    mrr  = mean_reciprocal_rank(retrieved_ids, req.relevant_doc_ids)
    ndcg = ndcg_at_k(retrieved_ids, req.relevant_doc_ids, req.top_k)
    f    = faithfulness(answer, chunks)
    ar   = answer_relevance(req.query, answer)

    log = QueryLog(
        query=req.query,
        answer=answer,
        retrieved_ids=retrieved_ids,
        hit_rate=hr,
        mrr=mrr,
        ndcg=ndcg,
        faithfulness=f,
        answer_relevance=ar,
    )
    db.add(log)
    db.commit()

    return {
        "query":  req.query,
        "answer": answer,
        "chunks": chunks,
        "eval": {
            "hit_rate":         hr,
            "mrr":              mrr,
            "ndcg":             ndcg,
            "faithfulness":     f,
            "answer_relevance": ar,
        },
    }

# ── Monitoring ────────────────────────────────────────────

@app.get("/eval/summary", tags=["monitoring"])
def eval_summary(db: Session = Depends(get_db)):
    logs = db.query(QueryLog).all()
    if not logs:
        return {"message": "No queries logged yet"}

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "total_queries":        len(logs),
        "avg_faithfulness":     avg([l.faithfulness for l in logs]),
        "avg_hit_rate":         avg([l.hit_rate for l in logs]),
        "avg_mrr":              avg([l.mrr for l in logs]),
        "avg_ndcg":             avg([l.ndcg for l in logs]),
        "avg_answer_relevance": avg([l.answer_relevance for l in logs]),
    }

@app.get("/eval/history", tags=["monitoring"])
def eval_history(limit: int = 20, db: Session = Depends(get_db)):
    logs = (
        db.query(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "query":            l.query,
            "answer":           l.answer[:200] + "..." if len(l.answer) > 200 else l.answer,
            "faithfulness":     l.faithfulness,
            "hit_rate":         l.hit_rate,
            "mrr":              l.mrr,
            "ndcg":             l.ndcg,
            "answer_relevance": l.answer_relevance,
            "created_at":       str(l.created_at),
        }
        for l in logs
    ]

@app.get("/index/stats", tags=["monitoring"])
def index_stats():
    from app.reranker import RERANKER_MODEL
    return {
        "total_chunks":   vector_store.count(),
        "embed_model":    os.getenv("EMBED_MODEL"),
        "gen_model":      os.getenv("GEN_MODEL"),
        "reranker_model": RERANKER_MODEL,
        "hybrid_alpha":   float(os.getenv("HYBRID_ALPHA", "0.7")),
    }

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "chunks_indexed": vector_store.count(),
        "model_loaded":  generator.model is not None,
    }
