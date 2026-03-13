"""
Tests for RequestIDMiddleware.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.middleware.request_id_middleware import RequestIDMiddleware


@pytest.fixture
def app_with_middleware():
    """
    Create a FastAPI app with RequestIDMiddleware for testing.
    """
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {"request_id": getattr(request.state, "request_id", "not_set")}

    return app


def test_request_id_middleware_new_id(app_with_middleware):
    """
    Test that RequestIDMiddleware adds a new request ID when none is provided.
    """
    client = TestClient(app_with_middleware)

    response = client.get("/test")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert response.json()["request_id"] != "not_set"
    assert "-" in response.json()["request_id"]
    assert len(response.json()["request_id"]) == 36  # UUID4 standard length


def test_request_id_middleware_existing_id(app_with_middleware):
    """
    Test that RequestIDMiddleware uses the existing request ID when provided.
    """
    client = TestClient(app_with_middleware)
    existing_id = "test-request-id-123"

    response = client.get("/test", headers={"X-Request-ID": existing_id})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == existing_id
    assert response.json()["request_id"] == existing_id


@patch("uuid.uuid4")
def test_request_id_middleware_generates_uuid(mock_uuid4, app_with_middleware):
    """
    Test that RequestIDMiddleware generates a UUID when no ID is provided.
    """
    mock_uuid = "mocked-uuid-1234"
    mock_uuid4.return_value = mock_uuid

    client = TestClient(app_with_middleware)
    response = client.get("/test")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == str(mock_uuid)
    assert response.json()["request_id"] == str(mock_uuid)
