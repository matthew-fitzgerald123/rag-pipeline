from __future__ import annotations
from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    doc_id     = Column(String, unique=True, nullable=False, index=True)
    title      = Column(String, nullable=False)
    content    = Column(Text, nullable=False)
    metadata_  = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

class QueryLog(Base):
    __tablename__ = "query_logs"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    query         = Column(Text, nullable=False)
    answer        = Column(Text, nullable=False)
    retrieved_ids = Column(JSON, default=[])
    top_k         = Column(Integer, nullable=True)
    hit_rate      = Column(Float, nullable=True)
    mrr           = Column(Float, nullable=True)
    ndcg          = Column(Float, nullable=True)
    faithfulness      = Column(Float, nullable=True)
    answer_relevance  = Column(Float, nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow, index=True)
