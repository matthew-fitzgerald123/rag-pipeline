# RAG Pipeline with Evaluation Framework

A retrieval-augmented generation pipeline with a manual evaluation framework. No RAGAS, no torch conflicts, ARM64 native.

## Stack

| Component | Library |
|---|---|
| Vector store | ChromaDB 0.5.3 (local, persistent) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (CPU) |
| Generation | mlx-lm + Mistral-7B-Instruct-v0.3-4bit (MLX) |
| API | FastAPI + uvicorn (port 8082) |
| Document store + eval logs | PostgreSQL + SQLAlchemy |

## Setup

```bash
# Create database
createdb rag_pipeline

# Install dependencies
pip install -r requirements.txt

# Add documents (plain .txt files)
# data/ml_basics.txt is included as a sample

# Ingest documents into ChromaDB + Postgres
make ingest
```

## Running

```bash
# Start API server (downloads Mistral-7B on first run, ~4GB)
make serve

# Run end-to-end demo (requires server running)
make demo

# Run tests
make test
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/query` | Retrieve + generate, logs faithfulness |
| POST | `/query/eval` | Same + hit rate and MRR against ground truth |
| GET | `/eval/summary` | Aggregate metrics across all logged queries |
| GET | `/eval/history` | Recent query log with per-query metrics |
| GET | `/index/stats` | Chunk count, model names |
| GET | `/health` | Server status |

Interactive docs at `http://localhost:8082/docs`.

## Evaluation Metrics

All metrics are implemented manually: no RAGAS dependency.

- **Hit rate**: fraction of relevant docs that appear in retrieved results
- **MRR**: mean reciprocal rank, rewards finding the right doc early
- **Faithfulness**: fraction of answer sentences grounded in retrieved context (token overlap proxy)
- **Answer relevance**: token overlap between query and answer

## Adding Documents

Drop any `.txt` files into `data/` and re-run `make ingest`. Already-ingested titles are skipped.

## Project Structure

```
app/
  chunker.py       sliding-window text chunker (512 chars, 64 overlap)
  vector_store.py  ChromaDB wrapper + MiniLM embedder
  generator.py     MLX Mistral-7B wrapper
  evaluator.py     hit_rate, MRR, faithfulness, answer_relevance
  main.py          FastAPI app
  models.py        SQLAlchemy models (Document, QueryLog)
  database.py      SQLAlchemy engine + session
scripts/
  ingest.py        bulk ingest .txt files from data/
tests/
  test_rag.py      7 integration tests
  conftest.py      session-scoped generator fixture
notebooks/
  demo.py          end-to-end demo script
data/
  ml_basics.txt    sample ML concepts document
```

## Notes

- SentenceTransformer runs on CPU: MPS has stability issues with this model size on Apple Silicon
- ChromaDB telemetry is disabled (`anonymized_telemetry=False`) to prevent startup hangs
- Port 8082 to avoid conflicts with other local projects
