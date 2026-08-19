import sys
sys.path.insert(0, ".")

import json as _json
import pytest
from unittest.mock import MagicMock, patch
from app.generator import generator
from app.main import app
from app.database import get_db

@pytest.fixture(scope="session")
def ensure_generator_loaded():
    if generator.model is None:
        generator.load_model()

@pytest.fixture(autouse=True)
def mock_db_for_unit_tests(request):
    """Override get_db with a MagicMock for every non-integration test.

    Integration tests are skipped so they can use the real database session.
    This prevents route-level unit tests from failing with "Connection refused"
    when the route logs to the DB, and removes the need to set
    app.dependency_overrides manually in each test.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def mock_generator_for_integration(request):
    """Patch app.main.generator for integration tests so they pass without a loaded model.

    Integration tests exercise real vector store and DB behaviour; the LLM itself
    cannot be loaded in CI, so its answer() / answer_stream() are stubbed with
    deterministic returns while all other route logic runs normally.
    """
    if not request.node.get_closest_marker("integration"):
        yield
        return

    async def _fake_stream(query, chunks, max_tokens=512):
        yield _json.dumps({"token": "Mocked"})
        yield _json.dumps({"token": " answer."})
        yield "[DONE]"

    with patch("app.main.generator") as mock_gen:
        mock_gen.model = MagicMock()
        mock_gen.answer.return_value = "Mocked answer for integration testing."
        mock_gen.answer_stream = _fake_stream
        yield mock_gen
