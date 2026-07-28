import sys
sys.path.insert(0, ".")

import pytest
from unittest.mock import MagicMock
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

    Integration tests opt out via @pytest.mark.integration so they can
    use the real database session.  This prevents route-level unit tests
    from failing with "Connection refused" when the route logs to the DB.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.pop(get_db, None)
