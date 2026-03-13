import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.main import app
from src.presentation.proxy_utils import (
    HEADER_GUARDRAILED_BLOCKED,
    HEADER_GUARDRAILED_SAFETY_CODE,
)
from src.shared import (
    Action,
    SafetyCode,
    Status,
)


@pytest.mark.usefixtures(
    "mock_app_state_for_proxy", "mock_http_client_instance_fixture"
)
@pytest.fixture(scope="function")
def client(mock_app_state_for_proxy):
    """Provides a TestClient instance for the main application."""
    app.dependency_overrides = {}
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides = {}


SAMPLE_OAI_REQUEST = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello AI"}],
    "user": "test-user-oai",
}
SAMPLE_OAI_RESPONSE = {
    "id": "chatcmpl-mock123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19},
}
UNSAFE_STATUS_BLOCK = Status(
    status=status.HTTP_400_BAD_REQUEST,
    message="Blocked by Policy X",
    safety_code=SafetyCode.PROFANE,
    action=Action.OVERRIDE.value,
)
INTERNAL_ERROR_STATUS = Status(
    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    message="Internal validation error",
    safety_code=SafetyCode.UNEXPECTED,
    action=Action.OVERRIDE.value,
)


@patch("src.presentation.proxy_utils._validate_messages")
def test_openai_proxy_success(
    mock_validate: AsyncMock,
    mock_http_client_instance_fixture: MagicMock,
    client: TestClient,
):
    """Test successful request flow: Input OK -> Forward -> Output OK -> Return OpenAI response."""
    mock_validate.return_value = None

    mock_backend_response = MagicMock()
    mock_backend_response.status_code = status.HTTP_200_OK
    mock_backend_response.json = MagicMock(return_value=SAMPLE_OAI_RESPONSE)
    mock_backend_response.headers = httpx.Headers({"content-type": "application/json"})
    mock_backend_response.raise_for_status = MagicMock(return_value=None)
    mock_backend_response.content = json.dumps(SAMPLE_OAI_RESPONSE).encode()

    mock_http_client_instance_fixture.post.return_value = mock_backend_response

    headers = {"Authorization": "Bearer valid-key"}

    response = client.post(
        "/v1/chat/completions", headers=headers, json=SAMPLE_OAI_REQUEST
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200, got {response.status_code}. Body: {response.text}"
    assert response.json() == SAMPLE_OAI_RESPONSE
    assert HEADER_GUARDRAILED_BLOCKED not in response.headers

    mock_http_client_instance_fixture.post.assert_called_once()
    mock_validate.assert_called()


@patch("src.presentation.proxy_utils._validate_messages")
def test_openai_proxy_input_blocked(
    mock_validate: AsyncMock,
    mock_http_client_instance_fixture: MagicMock,
    client: TestClient,
):
    """Test input blocked flow: Input FAIL -> Return 200 + Fake Body + Headers."""
    mock_validate.return_value = UNSAFE_STATUS_BLOCK

    headers = {"Authorization": "Bearer valid-key"}

    response = client.post(
        "/v1/chat/completions", headers=headers, json=SAMPLE_OAI_REQUEST
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200, got {response.status_code}. Body: {response.text}"
    response_json = response.json()
    assert response_json["choices"][0]["message"]["content"] == (
        UNSAFE_STATUS_BLOCK.message
    )
    assert response_json["choices"][0]["finish_reason"] == "content_filter"
    assert response_json["model"] == SAMPLE_OAI_REQUEST["model"]
    assert response.headers.get(HEADER_GUARDRAILED_BLOCKED) == "true"
    assert response.headers.get(HEADER_GUARDRAILED_SAFETY_CODE) == str(
        UNSAFE_STATUS_BLOCK.safety_code
    )

    mock_validate.assert_called_once()
    mock_http_client_instance_fixture.post.assert_not_called()


@patch("src.presentation.proxy_utils._validate_messages")
def test_openai_proxy_output_blocked(
    mock_validate: AsyncMock,
    mock_http_client_instance_fixture: MagicMock,
    client: TestClient,
):
    """Test output blocked flow: Input OK -> Forward -> Output FAIL -> Return 200 + Fake Body + Headers."""

    mock_validate.side_effect = [None, UNSAFE_STATUS_BLOCK]

    mock_backend_response = MagicMock()
    mock_backend_response.status_code = status.HTTP_200_OK
    mock_backend_response.json = MagicMock(return_value=SAMPLE_OAI_RESPONSE)
    mock_backend_response.headers = httpx.Headers({"content-type": "application/json"})
    mock_backend_response.raise_for_status = MagicMock(return_value=None)
    mock_backend_response.content = json.dumps(SAMPLE_OAI_RESPONSE).encode()

    mock_http_client_instance_fixture.post.return_value = mock_backend_response

    headers = {"Authorization": "Bearer valid-key"}

    response = client.post(
        "/v1/chat/completions", headers=headers, json=SAMPLE_OAI_REQUEST
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200, got {response.status_code}. Body: {response.text}"
    response_json = response.json()
    assert response_json["choices"][0]["message"]["content"] == (
        UNSAFE_STATUS_BLOCK.message
    )
    assert response_json["choices"][0]["finish_reason"] == "content_filter"
    assert response_json["usage"] == SAMPLE_OAI_RESPONSE["usage"]
    assert response.headers.get(HEADER_GUARDRAILED_BLOCKED) == "true"
    assert response.headers.get(HEADER_GUARDRAILED_SAFETY_CODE) == str(
        UNSAFE_STATUS_BLOCK.safety_code
    )

    assert mock_validate.call_count == 2
    mock_http_client_instance_fixture.post.assert_called_once()


@patch("src.presentation.proxy_utils._validate_messages")
def test_openai_proxy_validation_internal_error(
    mock_validate: AsyncMock,
    mock_http_client_instance_fixture: MagicMock,
    client: TestClient,
):
    """Test flow when validator returns an internal error status."""
    mock_validate.return_value = INTERNAL_ERROR_STATUS

    headers = {"Authorization": "Bearer valid-key"}

    response = client.post(
        "/v1/chat/completions", headers=headers, json=SAMPLE_OAI_REQUEST
    )

    assert (
        response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    ), f"Expected 500, got {response.status_code}. Body: {response.text}"
    response_json = response.json()
    assert response_json["safety_code"] == INTERNAL_ERROR_STATUS.safety_code
    assert response_json["message"] == INTERNAL_ERROR_STATUS.message
    assert response_json["action"] == INTERNAL_ERROR_STATUS.action
    assert HEADER_GUARDRAILED_BLOCKED not in response.headers

    mock_validate.assert_called_once()
    mock_http_client_instance_fixture.post.assert_not_called()


@patch("src.presentation.proxy_utils._validate_messages")
def test_openai_proxy_backend_api_error(
    mock_validate: AsyncMock,
    mock_http_client_instance_fixture: MagicMock,
    client: TestClient,
):
    """Test passthrough of backend OpenAI API errors (e.g., 401, 429)."""
    mock_validate.return_value = None

    mock_request_for_exception = httpx.Request(
        method="POST", url="https://api.openai.com/v1/chat/completions"
    )

    mock_error_response = MagicMock(spec=httpx.Response)
    mock_error_response.status_code = status.HTTP_401_UNAUTHORIZED
    error_body_dict = {
        "error": {
            "message": "Incorrect API key mock",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }
    error_body_bytes = json.dumps(error_body_dict).encode()
    mock_error_response.content = error_body_bytes
    mock_error_response.text = error_body_bytes.decode()
    mock_error_response.json.return_value = error_body_dict
    mock_error_response.headers = httpx.Headers({"content-type": "application/json"})

    http_error = httpx.HTTPStatusError(
        message="Client error '401 Unauthorized' mock",
        request=mock_request_for_exception,
        response=mock_error_response,
    )
    mock_http_client_instance_fixture.post.side_effect = http_error

    headers = {"Authorization": "Bearer invalid-key"}

    response = client.post(
        "/v1/chat/completions", headers=headers, json=SAMPLE_OAI_REQUEST
    )

    assert (
        response.status_code == status.HTTP_401_UNAUTHORIZED
    ), f"Expected 401, got {response.status_code}. Body: {response.text}"
    assert "Incorrect API key mock" in response.text
    assert "invalid_api_key" in response.text
    assert HEADER_GUARDRAILED_BLOCKED not in response.headers

    mock_validate.assert_called_once()
    mock_http_client_instance_fixture.post.assert_called_once()


@patch("src.presentation.proxy_utils._validate_messages")
def test_openai_proxy_backend_network_error(
    mock_validate: AsyncMock,
    mock_http_client_instance_fixture: MagicMock,
    client: TestClient,
):
    """Test handling of network errors when calling backend."""
    mock_validate.return_value = None
    network_exception = httpx.TimeoutException(
        "Connection timed out mock", request=MagicMock()
    )
    mock_http_client_instance_fixture.post.side_effect = network_exception

    headers = {"Authorization": "Bearer valid-key"}

    response = client.post(
        "/v1/chat/completions", headers=headers, json=SAMPLE_OAI_REQUEST
    )

    assert (
        response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
    ), f"Expected 504, got {response.status_code}. Body: {response.text}"
    response_json = response.json()
    assert response_json["safety_code"] == SafetyCode.TIMEOUT
    assert "Request to OpenAI timed out" in response_json["message"]
    assert response_json["action"] == Action.RETRY.value

    mock_validate.assert_called_once()
    mock_http_client_instance_fixture.post.assert_called_once()


def test_openai_proxy_missing_auth_header(client: TestClient):
    """Test request fails with 401 if Authorization header is missing."""
    response = client.post("/v1/chat/completions", json=SAMPLE_OAI_REQUEST)
    assert (
        response.status_code == status.HTTP_401_UNAUTHORIZED
    ), f"Expected 401, got {response.status_code}. Body: {response.text}"
    response_json = response.json()
    assert response_json["message"] == "Missing or invalid OpenAI API Key."
    assert response_json["safety_code"] == SafetyCode.GENERIC_UNSAFE


def test_openai_proxy_invalid_json(client: TestClient):
    """Test request fails with 400 if JSON body is invalid."""
    headers = {"Authorization": "Bearer valid-key", "Content-Type": "application/json"}
    response = client.post(
        "/v1/chat/completions", headers=headers, content="{invalid json"
    )
    assert (
        response.status_code == status.HTTP_400_BAD_REQUEST
    ), f"Expected 400, got {response.status_code}. Body: {response.text}"
    response_json = response.json()
    assert "Invalid JSON payload" in response_json["message"]
    assert response_json["safety_code"] == SafetyCode.GENERIC_UNSAFE


@patch("src.presentation.proxy_utils._validate_messages")
def test_openai_proxy_streaming_not_implemented(
    mock_validate: AsyncMock,
    client: TestClient,
):
    """Test streaming request returns 501 Not Implemented."""
    mock_validate.return_value = None
    headers = {"Authorization": "Bearer valid-key"}
    request_body = {**SAMPLE_OAI_REQUEST, "stream": True}

    response = client.post("/v1/chat/completions", headers=headers, json=request_body)

    assert (
        response.status_code == status.HTTP_501_NOT_IMPLEMENTED
    ), f"Expected 501, got {response.status_code}. Body: {response.text}"
    response_json = response.json()
    assert (
        "Streaming responses are not yet supported by this proxy."
        in response_json["message"]
    )
    assert response_json["safety_code"] == SafetyCode.UNEXPECTED
    assert response_json["action"] == Action.OVERRIDE.value
