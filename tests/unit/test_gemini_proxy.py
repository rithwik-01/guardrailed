import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.presentation.proxy_utils import (
    HEADER_GUARDRAILED_ACTION,
    HEADER_GUARDRAILED_BLOCKED,
    HEADER_GUARDRAILED_MESSAGE,
    HEADER_GUARDRAILED_SAFETY_CODE,
)
from src.shared import Action, Policy, PolicyType, SafetyCode, Status

GEMINI_PATH = "/v1beta/models/gemini-1.5-flash-latest:generateContent"
SAMPLE_GEMINI_REQUEST = {
    "contents": [{"role": "user", "parts": [{"text": "Hello there"}]}],
    "generationConfig": {"temperature": 0.7},
}
SAMPLE_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "General Kenobi!"}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
            "safetyRatings": [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "probability": "NEGLIGIBLE",
                },
            ],
        }
    ],
    "promptFeedback": {
        "safetyRatings": [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "probability": "NEGLIGIBLE",
            },
        ]
    },
}

SAFE_STATUS = Status(
    status=status.HTTP_200_OK, message="Safe", safety_code=SafetyCode.SAFE
)
UNSAFE_STATUS_BLOCK_FROM_VALIDATOR = Status(
    status=status.HTTP_400_BAD_REQUEST,
    message="Blocked Content",
    safety_code=SafetyCode.PROFANE,
    action=Action.OVERRIDE.value,
)
INTERNAL_ERROR_STATUS_FROM_VALIDATOR = Status(
    status=status.HTTP_503_SERVICE_UNAVAILABLE,
    message="Service component not ready: Presidio Analyzer Engine not initialized.",
    safety_code=SafetyCode.UNEXPECTED,
    action=Action.RETRY.value,
)

DUMMY_POLICIES = [
    Policy(
        id=PolicyType.PROFANITY,
        name="Toxicity Check",
        state=True,
        action=Action.OVERRIDE.value,
    ),
    Policy(
        id=PolicyType.PII_LEAKAGE,
        name="PII Check",
        state=True,
        action=Action.RETRY.value,
    ),
]

VALIDATE_MESSAGES_PATH = "src.presentation.routes.gemini_proxy._validate_messages"
CREATE_BLOCKED_RESPONSE_PATH = (
    "src.presentation.routes.gemini_proxy.create_blocked_response"
)
HTTP_CLIENT_POST_PATH = "src.presentation.proxy_utils.http_client.post"


@pytest.mark.usefixtures("mock_app_state_for_proxy")
@pytest.fixture(scope="function")
def mock_httpx_post():
    with patch(HTTP_CLIENT_POST_PATH, new_callable=AsyncMock) as mock_post:
        mock_post.reset_mock(return_value=True, side_effect=True)
        yield mock_post


@pytest.mark.usefixtures(
    "mock_policies_dependency",
    "mock_app_state_for_proxy",
)
class TestGeminiProxy:
    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_gemini_proxy_success(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_httpx_post: AsyncMock,
    ):
        """Test successful flow."""
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_200_OK
        mock_backend_response.json = MagicMock(return_value=SAMPLE_GEMINI_RESPONSE)
        mock_backend_response.headers = httpx.Headers(
            {"content-type": "application/json"}
        )
        mock_backend_response.raise_for_status = MagicMock(return_value=None)
        mock_backend_response.content = json.dumps(SAMPLE_GEMINI_RESPONSE).encode()
        mock_httpx_post.return_value = mock_backend_response

        headers = {"x-goog-api-key": "valid-key"}
        response = client.post(GEMINI_PATH, headers=headers, json=SAMPLE_GEMINI_REQUEST)

        assert response.status_code == status.HTTP_200_OK, f"Body: {response.text}"
        assert response.json() == SAMPLE_GEMINI_RESPONSE
        assert HEADER_GUARDRAILED_BLOCKED not in response.headers

        assert mock_validate.call_count == 2
        mock_httpx_post.assert_called_once()
        call_args, call_kwargs = mock_httpx_post.call_args
        assert call_kwargs["json"] == SAMPLE_GEMINI_REQUEST
        assert isinstance(call_kwargs.get("params"), dict)
        assert call_kwargs["params"].get("key") == "valid-key"

    @patch(
        VALIDATE_MESSAGES_PATH,
        new_callable=AsyncMock,
        return_value=UNSAFE_STATUS_BLOCK_FROM_VALIDATOR,
    )
    def test_gemini_proxy_input_blocked(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_httpx_post: AsyncMock,
    ):
        """Test input blocked flow."""
        headers = {"x-goog-api-key": "valid-key"}
        response = client.post(GEMINI_PATH, headers=headers, json=SAMPLE_GEMINI_REQUEST)

        assert response.status_code == status.HTTP_200_OK, f"Body: {response.text}"
        response_data = response.json()
        assert "candidates" in response_data and len(response_data["candidates"]) == 1
        assert response_data["candidates"][0]["finishReason"] == "SAFETY"
        assert "promptFeedback" in response_data
        assert response_data["promptFeedback"].get("blockReason") == "OTHER"
        assert (
            response_data["candidates"][0]["content"]["parts"][0]["text"]
            == UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.message
        )

        assert response.headers.get(HEADER_GUARDRAILED_BLOCKED) == "true"
        assert response.headers.get(HEADER_GUARDRAILED_SAFETY_CODE) == str(
            UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.safety_code
        )
        assert response.headers.get(HEADER_GUARDRAILED_ACTION) == str(
            UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.action
        )
        assert (
            response.headers.get(HEADER_GUARDRAILED_MESSAGE)
            == UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.message
        )

        mock_validate.assert_called_once()
        mock_httpx_post.assert_not_called()

    @patch(CREATE_BLOCKED_RESPONSE_PATH)
    @patch(
        VALIDATE_MESSAGES_PATH,
        new_callable=AsyncMock,
        side_effect=[None, UNSAFE_STATUS_BLOCK_FROM_VALIDATOR],
    )
    def test_gemini_proxy_output_blocked(
        self,
        mock_validate: AsyncMock,
        mock_create_blocked_response: MagicMock,
        client: TestClient,
        mock_httpx_post: AsyncMock,
    ):
        """Test output blocked flow."""
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_200_OK
        mock_backend_response.json = MagicMock(return_value=SAMPLE_GEMINI_RESPONSE)
        mock_backend_response.headers = httpx.Headers(
            {"content-type": "application/json"}
        )
        mock_backend_response.raise_for_status = MagicMock(return_value=None)
        mock_backend_response.content = json.dumps(SAMPLE_GEMINI_RESPONSE).encode()
        mock_httpx_post.return_value = mock_backend_response

        fake_blocked_body = {
            "candidates": [
                {
                    "finishReason": "SAFETY",
                    "content": {
                        "parts": [{"text": UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.message}]
                    },
                }
            ],
            "promptFeedback": {"blockReason": "OTHER"},
        }
        fake_headers = {
            HEADER_GUARDRAILED_BLOCKED: "true",
            HEADER_GUARDRAILED_SAFETY_CODE: str(
                UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.safety_code
            ),
            HEADER_GUARDRAILED_ACTION: str(UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.action),
            HEADER_GUARDRAILED_MESSAGE: UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.message,
        }
        mock_create_blocked_response.return_value = JSONResponse(
            status_code=status.HTTP_200_OK,
            content=fake_blocked_body,
            headers=fake_headers,
        )

        headers = {"x-goog-api-key": "valid-key"}
        response = client.post(GEMINI_PATH, headers=headers, json=SAMPLE_GEMINI_REQUEST)

        assert response.status_code == status.HTTP_200_OK, f"Body: {response.text}"
        assert response.json() == fake_blocked_body
        assert response.headers.get(HEADER_GUARDRAILED_BLOCKED) == "true"
        assert response.headers.get(HEADER_GUARDRAILED_SAFETY_CODE) == str(
            UNSAFE_STATUS_BLOCK_FROM_VALIDATOR.safety_code
        )

        assert mock_validate.call_count == 2
        mock_httpx_post.assert_called_once()
        mock_create_blocked_response.assert_called_once_with(
            provider="gemini",
            block_status=UNSAFE_STATUS_BLOCK_FROM_VALIDATOR,
            original_request_data=SAMPLE_GEMINI_REQUEST,
            original_response_data=SAMPLE_GEMINI_RESPONSE,
        )

    @patch(
        VALIDATE_MESSAGES_PATH,
        new_callable=AsyncMock,
        return_value=INTERNAL_ERROR_STATUS_FROM_VALIDATOR,
    )
    def test_gemini_proxy_validation_internal_error(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_httpx_post: AsyncMock,
    ):
        """Test flow when validator returns an internal error status."""
        headers = {"x-goog-api-key": "valid-key"}
        response = client.post(GEMINI_PATH, headers=headers, json=SAMPLE_GEMINI_REQUEST)

        assert (
            response.status_code == INTERNAL_ERROR_STATUS_FROM_VALIDATOR.status
        ), f"Body: {response.text}"
        response_data = response.json()
        assert (
            response_data["safety_code"]
            == INTERNAL_ERROR_STATUS_FROM_VALIDATOR.safety_code
        )
        assert response_data["message"] == INTERNAL_ERROR_STATUS_FROM_VALIDATOR.message
        assert response_data["action"] == INTERNAL_ERROR_STATUS_FROM_VALIDATOR.action

        mock_validate.assert_called_once()
        mock_httpx_post.assert_not_called()

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_gemini_proxy_backend_api_error(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_httpx_post: AsyncMock,
    ):
        """Test passthrough of backend Gemini API errors (e.g., 400 bad key)."""
        mock_request_for_exception = httpx.Request("POST", "http://fakegemini/generate")
        error_body = {
            "error": {
                "code": 400,
                "message": "API key not valid.",
                "status": "INVALID_ARGUMENT",
            }
        }
        mock_response = httpx.Response(
            status.HTTP_400_BAD_REQUEST,
            json=error_body,
            request=mock_request_for_exception,
            headers={"content-type": "application/json"},
        )
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_400_BAD_REQUEST
        mock_backend_response.json = MagicMock(return_value=error_body)
        mock_backend_response.headers = httpx.Headers(
            {"content-type": "application/json"}
        )
        mock_backend_response.content = json.dumps(error_body).encode()
        mock_backend_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                message="Bad Request",
                request=mock_request_for_exception,
                response=mock_response,
            )
        )
        mock_httpx_post.return_value = mock_backend_response

        headers = {"x-goog-api-key": "invalid-key"}
        response = client.post(GEMINI_PATH, headers=headers, json=SAMPLE_GEMINI_REQUEST)

        assert (
            response.status_code == status.HTTP_400_BAD_REQUEST
        ), f"Body: {response.text}"
        assert response.json() == error_body
        assert HEADER_GUARDRAILED_BLOCKED not in response.headers

        mock_validate.assert_called_once()
        mock_httpx_post.assert_called_once()
        mock_backend_response.raise_for_status.assert_called_once()

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_gemini_proxy_backend_network_error(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_httpx_post: AsyncMock,
    ):
        """Test handling of network errors when calling backend."""
        mock_request = MagicMock(spec=httpx.Request)
        mock_httpx_post.side_effect = httpx.ConnectError(
            "Connection failed", request=mock_request
        )

        headers = {"x-goog-api-key": "valid-key"}
        response = client.post(GEMINI_PATH, headers=headers, json=SAMPLE_GEMINI_REQUEST)

        assert (
            response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        ), f"Body: {response.text}"
        response_data = response.json()
        assert response_data["safety_code"] == SafetyCode.UNEXPECTED
        assert "Network error communicating with Gemini" in response_data["message"]
        assert response_data["action"] == Action.RETRY.value

        mock_validate.assert_called_once()
        mock_httpx_post.assert_called_once()

    def test_gemini_proxy_missing_key(self, client: TestClient):
        response = client.post(GEMINI_PATH, headers={}, json=SAMPLE_GEMINI_REQUEST)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing Gemini API Key" in response.json()["message"]

    def test_gemini_proxy_invalid_json(self, client: TestClient):
        headers = {"x-goog-api-key": "valid-key"}
        response = client.post(GEMINI_PATH, headers=headers, content=b"{invalid json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid JSON payload" in response.json()["message"]
        assert response.json()["safety_code"] == SafetyCode.GENERIC_UNSAFE

    @patch(
        "src.presentation.routes.gemini_proxy._extract_input_messages_from_gemini",
        return_value=[],
    )
    def test_gemini_proxy_no_valid_input_messages(
        self, mock_extract_input, client: TestClient
    ):
        headers = {"x-goog-api-key": "valid-key"}
        request_data = {"contents": [{"role": "user", "parts": [{"imageData": "..."}]}]}
        response = client.post(GEMINI_PATH, headers=headers, json=request_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert (
            "Could not extract valid message content from 'contents'"
            in response.json()["message"]
        )
        assert response.json()["safety_code"] == SafetyCode.GENERIC_UNSAFE
        mock_extract_input.assert_called_once_with(request_data)

    def test_gemini_proxy_streaming_not_implemented(self, client: TestClient):
        """Test streaming path handling (checks 404 for non-existent path)."""
        headers = {"x-goog-api-key": "valid-key"}
        non_existent_streaming_path = GEMINI_PATH.replace(
            ":generateContent", ":streamGenerateContent"
        )
        response_404 = client.post(
            non_existent_streaming_path, headers=headers, json=SAMPLE_GEMINI_REQUEST
        )
        assert response_404.status_code == status.HTTP_404_NOT_FOUND
        response_json = response_404.json()
        assert "message" in response_json and response_json["message"] == "Not Found"
        assert (
            "safety_code" in response_json
            and response_json["safety_code"] == SafetyCode.GENERIC_UNSAFE
        )
        assert (
            "action" in response_json
            and response_json["action"] == Action.OVERRIDE.value
        )
