"""
Common test fixtures and configurations for pytest.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import app

pytest_plugins = [
    "tests.fixtures",
]


@pytest.fixture(scope="function")
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client_instance:
        original_overrides = app.dependency_overrides.copy()
        app.dependency_overrides = {}
        try:
            yield test_client_instance
        finally:
            app.dependency_overrides = original_overrides
