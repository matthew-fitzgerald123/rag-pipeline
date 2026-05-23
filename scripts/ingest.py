"""
Ingests text files from data/ into ChromaDB + Postgres.
Add any .txt files to data/ before running.
Run: make ingest
"""
from __future__ import annotations
import sys, os, uuid
sys.path.insert(0, ".")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from app.database import engine, SessionLocal
from app.models import Base, Document
from app.chunker import chunk_document
from app.vector_store import vector_store

Base.metadata.create_all(bind=engine)

DATA_DIR = Path("./data")

def ingest_file(path: Path, db):
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return

    doc_id = str(uuid.uuid4())[:8]
    title = path.stem

    existing = db.query(Document).filter_by(title=title).first()
    if existing:
        print(f"  Skipping {title} — already ingested")
        return

    doc = Document(doc_id=doc_id, title=title, content=content)
    db.add(doc)
    db.commit()

    chunks = chunk_document(
        doc_id=doc_id,
        text=content,
        metadata={"title": title, "source": str(path)},
        chunk_size=512,
        overlap=64,
    )
    vector_store.add_chunks(chunks)
    print(f"  Ingested: {title} — {len(chunks)} chunks")

if __name__ == "__main__":
    db = SessionLocal()
    txt_files = list(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in data/ — add some documents first")
        sys.exit(1)
    print(f"Ingesting {len(txt_files)} files...")
    for f in txt_files:
        ingest_file(f, db)
    db.close()
    print(f"\nDone. Vector store contains {vector_store.count()} chunks.")
