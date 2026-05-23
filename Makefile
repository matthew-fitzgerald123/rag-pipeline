serve:
	uvicorn app.main:app --reload --port 8082

test:
	pytest tests/ -v

demo:
	python notebooks/demo.py

ingest:
	python scripts/ingest.py
