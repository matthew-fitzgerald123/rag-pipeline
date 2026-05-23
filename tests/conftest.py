import sys
sys.path.insert(0, ".")

import pytest
from app.generator import generator

@pytest.fixture(scope="session", autouse=True)
def ensure_generator_loaded():
    if generator.model is None:
        generator.load_model()
