"""
Tests for TimeoutMiddleware.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.middleware.timeout_middleware import TimeoutMiddleware
from src.shared import SafetyCode


@pytest.fixture
def mock_config():
    """Create a mock config for the middleware."""
    mock = MagicMock()
    mock.middleware.timeout.default_timeout = 2
    mock.middleware.timeout.path_timeouts = {
        "/slow": 5,
        "/very-slow": 10,
        "/prefix/*": 3,
    }
    return mock


@pytest.fixture
def app_with_timeout(mock_config):
    """
    Create a FastAPI app with TimeoutMiddleware for testing.
    """
    with patch("src.middleware.timeout_middleware.app_state.config", mock_config):
        app = FastAPI()
        app.add_middleware(TimeoutMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        @app.get("/slow")
        async def slow_endpoint(request: Request):
            if hasattr(request.state, "_simulate_timeout_test"):
                await asyncio.sleep(0.01)
            return {"message": "slow response"}

        @app.get("/very-slow")
        async def very_slow_endpoint(request: Request):
            await asyncio.sleep(0.01)
            return {"message": "very slow response"}

        @app.get("/prefix/test")
        async def prefix_endpoint():
            await asyncio.sleep(0.01)
            return {"message": "prefix response"}

        return app


def test_get_timeout_for_path():
    """
    Test that the middleware correctly determines timeouts for different paths.
    """
    with patch("src.middleware.timeout_middleware.app_state.config") as mock_config:
        mock_config.middleware.timeout.default_timeout = 30
        mock_config.middleware.timeout.path_timeouts = {
            "/specific": 60,
            "/api/*": 45,
            "/safeguard": 120,
        }

        middleware = TimeoutMiddleware(app=MagicMock())

        assert middleware.get_timeout_for_path("/random") == 30

        assert middleware.get_timeout_for_path("/specific") == 60

        assert middleware.get_timeout_for_path("/api/endpoint") == 45
        assert middleware.get_timeout_for_path("/api/other") == 45

        assert middleware.get_timeout_for_path("/safeguard") == 120


def test_fast_response(app_with_timeout):
    """
    Test that fast responses complete normally.
    """
    client = TestClient(app_with_timeout)
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "test"}


def test_slow_response_custom_timeout(app_with_timeout):
    """
    Test that slow responses complete if within custom path timeout.
    """
    client = TestClient(app_with_timeout)
    response = client.get("/slow")
    assert response.status_code == 200
    assert response.json() == {"message": "slow response"}


def test_very_slow_response_timeout(app_with_timeout):
    """
    Test that very slow responses timeout even with custom path timeout.
    """
    middleware = next(
        m
        for m in app_with_timeout.user_middleware
        if isinstance(m.cls, type) and m.cls.__name__ == "TimeoutMiddleware"
    ).cls(app_with_timeout)

    assert middleware.get_timeout_for_path("/very-slow") == 30

    with patch("asyncio.wait_for") as mock_wait_for:
        mock_wait_for.side_effect = asyncio.TimeoutError()

        client = TestClient(app_with_timeout)
        response = client.get("/very-slow")

        assert response.status_code == 504
        assert response.json()["safety_code"] == SafetyCode.TIMEOUT


def test_prefix_timeout(app_with_timeout):
    """
    Test that prefix-based timeouts work correctly.
    """
    client = TestClient(app_with_timeout)
    response = client.get("/prefix/test")
    assert response.status_code == 200
    assert response.json() == {"message": "prefix response"}
