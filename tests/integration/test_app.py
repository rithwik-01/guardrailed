"""
High-level integration tests for the main FastAPI application (OS Core).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.core import app_state
from src.presentation.dependencies import get_loaded_policies
from src.shared import Action, Policy, PolicyType, Result, SafetyCode


@pytest.fixture
def test_policies():
    """Minimal list of policies for integration testing basic safeguard flow."""
    return [
        Policy(
            id=PolicyType.PROFANITY.value,
            name="Test Tox",
            state=True,
            action=Action.OVERRIDE.value,
            threshold=0.8,
            message="Tox fail int",
            is_user_policy=True,
            is_llm_policy=True,
            pii_entities=None,
            pii_threshold=0.5,
            pii_categories=None,
            protected_prompts=None,
            prompt_leakage_threshold=0.85,
            locations=None,
            persons=None,
            competitors=None,
            metadata={},
        ),
    ]


def test_health_check(client: TestClient):
    """Verify the /health endpoint is reachable and returns OK."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_cors_headers(client: TestClient):
    """Verify that CORS headers are present and allow specified headers."""
    origin = "http://example.com"
    response = client.get("/health", headers={"Origin": origin})
    assert response.status_code == status.HTTP_200_OK
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in [origin, "*"]
    assert "access-control-allow-credentials" in response.headers
    assert response.headers["access-control-allow-credentials"] == "true"

    response_options = client.options(
        f"/{app_state.config.gemini_api_version}/models/gemini-pro:generateContent",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-goog-api-key,authorization",
        },
    )
    assert response_options.status_code == status.HTTP_200_OK
    assert "access-control-allow-origin" in response_options.headers
    assert response_options.headers["access-control-allow-origin"] in [origin, "*"]
    assert "access-control-allow-methods" in response_options.headers
    assert "POST" in response_options.headers["access-control-allow-methods"]
    assert "access-control-allow-headers" in response_options.headers

    allowed_headers_str = response_options.headers["access-control-allow-headers"]
    allowed_headers_list = [h.strip().lower() for h in allowed_headers_str.split(",")]
    allow_all_headers = "*" in allowed_headers_list
    assert allow_all_headers or "content-type" in allowed_headers_list
    assert allow_all_headers or "x-goog-api-key" in allowed_headers_list
    assert allow_all_headers or "authorization" in allowed_headers_list


def test_request_id_middleware(client: TestClient):
    """Verify the RequestIDMiddleware adds the X-Request-ID header."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert "x-request-id" in response.headers
    assert isinstance(response.headers["x-request-id"], str)
    assert len(response.headers["x-request-id"]) > 10


@patch("src.presentation.routes.safeguard.ContentValidator", new_callable=MagicMock)
def test_safeguard_basic_flow(
    mock_validator_cls: MagicMock,
    client: TestClient,
    test_policies: list[Policy],
):
    """Test a basic successful request to the OS safeguard endpoint."""
    mock_validator_instance = MagicMock()
    safe_result_status = Result.safe_result()
    mock_validator_instance.validate_content = AsyncMock(
        return_value=safe_result_status
    )
    mock_validator_cls.return_value = mock_validator_instance

    async def override_policies():
        return test_policies

    original_overrides = client.app.dependency_overrides.copy()
    client.app.dependency_overrides[get_loaded_policies] = override_policies

    request_body = {
        "messages": [{"role": "user", "content": "Integration test message"}]
    }

    try:
        response = client.post("/safeguard", json=request_body)

        print(
            f"Response for safeguard_basic_flow: {response.status_code} - {response.text}"
        )
        assert response.status_code == status.HTTP_200_OK
        response_json = response.json()
        assert response_json["safety_code"] == SafetyCode.SAFE
        assert "validated successfully" in response_json["message"]
        assert "action" in response_json
        assert response_json["action"] is None

        mock_validator_cls.assert_called_once()
        call_args, _ = mock_validator_cls.call_args
        context_arg = call_args[0]
        assert context_arg.policies == test_policies
        assert context_arg.messages == request_body["messages"]
        mock_validator_instance.validate_content.assert_awaited_once()

    finally:
        client.app.dependency_overrides = original_overrides


def test_openai_proxy_endpoint_exists(client: TestClient):
    """Test if the OpenAI proxy endpoint path exists (expects 401 without auth)."""
    response = client.post("/v1/chat/completions", json={})
    assert response.status_code != status.HTTP_404_NOT_FOUND
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_gemini_proxy_endpoint_exists(client: TestClient):
    """Test if the Gemini proxy endpoint path exists (expects 401 without auth)."""
    model_name = "gemini-pro"
    gemini_path = (
        f"/{app_state.config.gemini_api_version}/models/{model_name}:generateContent"
    )
    response = client.post(gemini_path, json={})
    assert response.status_code != status.HTTP_404_NOT_FOUND
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_metrics_endpoint(client: TestClient):
    """Verify the /metrics endpoint is reachable and returns Prometheus format."""
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"].startswith("text/plain")

    metrics_text = response.text

    assert "# HELP http_requests_total Total count of requests" in metrics_text
    assert "# TYPE http_requests_total counter" in metrics_text
    assert "http_requests_total{" in metrics_text

    assert (
        "# HELP python_gc_objects_collected_total Objects collected during gc"
        in metrics_text
    )
    assert "# TYPE python_gc_objects_collected_total counter" in metrics_text
    assert "python_gc_objects_collected_total{" in metrics_text
