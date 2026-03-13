"""
Integration tests for health check routes.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Return a TestClient for the application."""
    return TestClient(app)


def test_health_route(client):
    """
    Test the basic health check endpoint.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
