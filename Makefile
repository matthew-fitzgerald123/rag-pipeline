serve:
	uvicorn app.main:app --reload --port 8082

test:
	pytest tests/ -v

test-unit:
	pytest tests/ -m "not integration" -v

demo:
	python notebooks/demo.py

ingest:
	python scripts/ingest.py
