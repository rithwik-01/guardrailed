"""
Tests for SecurityMiddleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.middleware.security_middleware import SecurityMiddleware


@pytest.fixture
def app_with_security():
    """
    Create a FastAPI app with SecurityMiddleware for testing.
    """
    app = FastAPI()
    app.add_middleware(SecurityMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    return app


def test_security_middleware_headers(app_with_security):
    """
    Test that SecurityMiddleware adds security headers.
    """
    client = TestClient(app_with_security)

    response = client.get("/test")

    assert response.status_code == 200

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
    assert "includeSubDomains" in response.headers["Strict-Transport-Security"]
    assert "preload" in response.headers["Strict-Transport-Security"]
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in response.headers["Permissions-Policy"]
    assert "microphone=()" in response.headers["Permissions-Policy"]
    assert "geolocation=()" in response.headers["Permissions-Policy"]
    assert response.headers["Cache-Control"] == "no-store, max-age=0"


def test_security_middleware_removes_server_header(app_with_security):
    """
    Test that SecurityMiddleware removes the Server header if present.
    """
    client = TestClient(app_with_security)

    response = client.get("/test")

    assert response.status_code == 200
    assert "Server" not in response.headers


def test_security_middleware_header_values(app_with_security):
    """
    Test the specific values of certain critical security headers.
    """
    client = TestClient(app_with_security)

    response = client.get("/test")

    assert response.status_code == 200

    csp = response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "style-src 'self'" in csp
    assert "img-src 'self' data:" in csp
    assert "object-src 'none'" in csp

    pp = response.headers["Permissions-Policy"]
    assert "camera=()" in pp
    assert "microphone=()" in pp
    assert "geolocation=()" in pp
    assert "payment=()" in pp

    hsts = response.headers["Strict-Transport-Security"]
    assert "max-age=31536000" in hsts
    assert "includeSubDomains" in hsts
    assert "preload" in hsts
