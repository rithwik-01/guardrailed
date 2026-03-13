import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.presentation.proxy_utils import (
    HEADER_ANTHROPIC_API_KEY,
    HEADER_GUARDRAILED_BLOCKED,
)
from src.presentation.routes.claude_proxy import (
    _extract_claude_api_key,
    _extract_input_messages_from_claude,
    _extract_output_message_from_claude,
)
from src.shared import Action, Agent, SafetyCode

CLAUDE_PATH = "/anthropic/v1/messages"
CLAUDE_API_VERSION = "2023-06-01"

VALIDATE_MESSAGES_PATH = "src.presentation.proxy_utils._validate_messages"
HTTP_CLIENT_POST_PATH = "src.presentation.proxy_utils.http_client.post"

pytestmark = pytest.mark.usefixtures("mock_app_state_for_proxy")


class TestClaudeProxyHelperFunctions:
    """Tests for the Claude proxy helper functions."""

    def test_extract_claude_api_key(self):
        """Test extracting Claude API key from request headers."""
        mock_request = MagicMock()
        mock_request.headers = {HEADER_ANTHROPIC_API_KEY: "test-key-123"}
        assert _extract_claude_api_key(mock_request) == "test-key-123"

        mock_request.headers = {}
        assert _extract_claude_api_key(mock_request) is None

    def test_extract_input_messages_basic(self):
        """Test basic extraction of input messages from Claude request."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello Claude"},
                {"role": "assistant", "content": "Hi there, how can I help?"},
            ]
        }

        expected = [
            {"role": Agent.USER, "content": "Hello Claude"},
            {"role": Agent.ASSISTANT, "content": "Hi there, how can I help?"},
        ]

        result = _extract_input_messages_from_claude(request_data)
        assert result == expected

    def test_extract_input_messages_content_list(self):
        """Test extraction of messages with content as list of blocks."""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello Claude"},
                        {"type": "text", "text": "This is a follow-up"},
                    ],
                }
            ]
        }

        expected = [
            {"role": Agent.USER, "content": "Hello Claude\nThis is a follow-up"}
        ]

        result = _extract_input_messages_from_claude(request_data)
        assert result == expected

    def test_extract_input_messages_mixed_content(self):
        """Test extraction with mixed content types (text and non-text)."""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {"type": "image", "source": {"data": "base64-data"}},
                        {"type": "text", "text": "Focus on the main subject."},
                    ],
                }
            ]
        }

        expected = [
            {
                "role": Agent.USER,
                "content": "Describe this image:\nFocus on the main subject.",
            }
        ]

        result = _extract_input_messages_from_claude(request_data)
        assert result == expected

    def test_extract_input_messages_edge_cases(self):
        """Test extraction with various edge cases."""
        assert _extract_input_messages_from_claude({"messages": []}) == []

        assert _extract_input_messages_from_claude({"messages": "not a list"}) == []

        assert _extract_input_messages_from_claude({}) == []

        assert _extract_input_messages_from_claude({"messages": ["not a dict"]}) == []

        result = _extract_input_messages_from_claude(
            {"messages": [{"role": "unknown", "content": "Hello"}]}
        )
        assert result == []

        result = _extract_input_messages_from_claude({"messages": [{"role": "user"}]})
        assert result == []

        result = _extract_input_messages_from_claude(
            {"messages": [{"role": "user", "content": 123}]}
        )
        assert result == []

        result = _extract_input_messages_from_claude(
            {"messages": [{"role": "user", "content": []}]}
        )
        assert result == []

        result = _extract_input_messages_from_claude(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"data": "base64"}}],
                    }
                ]
            }
        )
        assert result == []

        result = _extract_input_messages_from_claude(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": ""}]}]}
        )
        assert result == []

    def test_extract_output_message_basic(self):
        """Test basic extraction of output message from Claude response."""
        response_data = {
            "content": [{"type": "text", "text": "This is a test response."}]
        }

        expected = {"role": Agent.ASSISTANT, "content": "This is a test response."}

        result = _extract_output_message_from_claude(response_data)
        assert result == expected

    def test_extract_output_message_multiple_blocks(self):
        """Test extraction with multiple text blocks."""
        response_data = {
            "content": [
                {"type": "text", "text": "First paragraph."},
                {"type": "text", "text": "Second paragraph."},
            ]
        }

        expected = {
            "role": Agent.ASSISTANT,
            "content": "First paragraph.\nSecond paragraph.",
        }

        result = _extract_output_message_from_claude(response_data)
        assert result == expected

    def test_extract_output_message_edge_cases(self):
        """Test output extraction with various edge cases."""
        assert _extract_output_message_from_claude({"content": []}) is None

        assert _extract_output_message_from_claude({"content": "not a list"}) is None

        assert _extract_output_message_from_claude({}) is None

        assert _extract_output_message_from_claude({"content": ["not a dict"]}) is None

        result = _extract_output_message_from_claude(
            {"content": [{"type": "image", "source": "data"}]}
        )
        assert result is None

        result = _extract_output_message_from_claude(
            {
                "content": [
                    {"type": "image", "source": "data"},
                    {"type": "text", "text": "Valid text"},
                    {"type": "tool_use", "id": "tool1"},
                ]
            }
        )
        expected = {"role": Agent.ASSISTANT, "content": "Valid text"}
        assert result == expected

        result = _extract_output_message_from_claude(
            {"content": [{"type": "text", "text": ""}]}
        )
        assert result is None


@pytest.mark.usefixtures(
    "mock_policies_dependency",
    "mock_app_state_for_proxy",
)
class TestClaudeProxyAdditionalCases:
    """Additional tests for Claude proxy route edge cases."""

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_claude_proxy_with_beta_header(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_http_client_instance_fixture: MagicMock,
    ):
        """Test that anthropic-beta header is preserved when forwarding."""
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_200_OK
        mock_backend_response.json = MagicMock(
            return_value={"content": [{"type": "text", "text": "Response"}]}
        )
        mock_backend_response.headers = httpx.Headers(
            {"content-type": "application/json"}
        )
        mock_backend_response.raise_for_status = MagicMock(return_value=None)
        mock_backend_response.content = json.dumps(
            {"content": [{"type": "text", "text": "Response"}]}
        ).encode()
        mock_http_client_instance_fixture.post.return_value = mock_backend_response

        headers = {
            HEADER_ANTHROPIC_API_KEY: "valid-key",
            "anthropic-beta": "claude-3-haiku-20240307",
        }
        request_data = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(CLAUDE_PATH, headers=headers, json=request_data)

        assert response.status_code == status.HTTP_200_OK

        call_args, call_kwargs = mock_http_client_instance_fixture.post.call_args
        sent_headers = call_kwargs.get("headers", {})
        assert sent_headers.get("anthropic-beta") == "claude-3-haiku-20240307"

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_claude_proxy_timeout_handling(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_http_client_instance_fixture: MagicMock,
    ):
        """Test handling of timeouts when calling Claude API."""
        timeout_exception = httpx.TimeoutException(
            "Request timed out", request=MagicMock()
        )
        mock_http_client_instance_fixture.post.side_effect = timeout_exception

        headers = {HEADER_ANTHROPIC_API_KEY: "valid-key"}
        request_data = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(CLAUDE_PATH, headers=headers, json=request_data)

        assert response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        response_data = response.json()
        assert response_data["safety_code"] == SafetyCode.TIMEOUT
        assert "Request to Anthropic Claude timed out" in response_data["message"]
        assert response_data["action"] == Action.RETRY.value

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_claude_proxy_malformed_backend_response(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_http_client_instance_fixture: MagicMock,
    ):
        """Test handling when Claude returns malformed JSON response."""
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_200_OK
        mock_backend_response.json.side_effect = json.JSONDecodeError(
            "Invalid JSON", "", 0
        )
        mock_backend_response.headers = httpx.Headers(
            {"content-type": "application/json"}
        )
        mock_backend_response.raise_for_status = MagicMock(return_value=None)
        mock_backend_response.content = b"Not valid JSON"
        mock_http_client_instance_fixture.post.return_value = mock_backend_response

        headers = {HEADER_ANTHROPIC_API_KEY: "valid-key"}
        request_data = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(CLAUDE_PATH, headers=headers, json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_claude_proxy_missing_content_in_response(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_http_client_instance_fixture: MagicMock,
    ):
        """Test handling when Claude response is missing the content field."""
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_200_OK
        mock_backend_response.json = MagicMock(
            return_value={"id": "msg_123", "model": "claude-3"}
        )
        mock_backend_response.headers = httpx.Headers(
            {"content-type": "application/json"}
        )
        mock_backend_response.raise_for_status = MagicMock(return_value=None)
        mock_backend_response.content = json.dumps(
            {"id": "msg_123", "model": "claude-3"}
        ).encode()
        mock_http_client_instance_fixture.post.return_value = mock_backend_response

        headers = {HEADER_ANTHROPIC_API_KEY: "valid-key"}
        request_data = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(CLAUDE_PATH, headers=headers, json=request_data)

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data == {"id": "msg_123", "model": "claude-3"}
        assert HEADER_GUARDRAILED_BLOCKED not in response.headers

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_claude_proxy_user_id_extraction(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_http_client_instance_fixture: MagicMock,
    ):
        """Test that user_id is correctly extracted from metadata."""
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_200_OK
        mock_backend_response.json = MagicMock(
            return_value={"content": [{"type": "text", "text": "Response"}]}
        )
        mock_backend_response.headers = httpx.Headers(
            {"content-type": "application/json"}
        )
        mock_backend_response.raise_for_status = MagicMock(return_value=None)
        mock_backend_response.content = json.dumps(
            {"content": [{"type": "text", "text": "Response"}]}
        ).encode()
        mock_http_client_instance_fixture.post.return_value = mock_backend_response

        headers = {HEADER_ANTHROPIC_API_KEY: "valid-key"}
        request_data = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"user_id": "test-user-123"},
        }

        response = client.post(CLAUDE_PATH, headers=headers, json=request_data)

        assert response.status_code == status.HTTP_200_OK

        call_args = mock_validate.call_args_list[0][1]
        assert call_args.get("user_id") == "test-user-123"

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_claude_proxy_response_headers_forwarding(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_http_client_instance_fixture: MagicMock,
    ):
        """Test that response headers are properly forwarded with problematic ones removed."""
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_200_OK
        mock_backend_response.json = MagicMock(
            return_value={"content": [{"type": "text", "text": "Response"}]}
        )
        mock_backend_response.headers = httpx.Headers(
            {
                "content-type": "application/json",
                "content-encoding": "gzip",
                "content-length": "1024",
                "transfer-encoding": "chunked",
                "anthropic-version": CLAUDE_API_VERSION,
                "request-id": "req_123456",
                "x-request-id": "xyz-123",
            }
        )
        mock_backend_response.raise_for_status = MagicMock(return_value=None)
        mock_backend_response.content = json.dumps(
            {"content": [{"type": "text", "text": "Response"}]}
        ).encode()
        mock_http_client_instance_fixture.post.return_value = mock_backend_response

        headers = {HEADER_ANTHROPIC_API_KEY: "valid-key"}
        request_data = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(CLAUDE_PATH, headers=headers, json=request_data)

        assert response.status_code == status.HTTP_200_OK

        assert "content-encoding" not in response.headers
        # Note: content-length is added by FastAPI/Starlette when sending the response
        # so we don't check for its absence
        assert "transfer-encoding" not in response.headers

        assert response.headers.get("anthropic-version") == CLAUDE_API_VERSION
        assert response.headers.get("request-id") == "req_123456"
        # The x-request-id is being overwritten with a new UUID by the test client or middleware
        assert "x-request-id" in response.headers

    @patch(VALIDATE_MESSAGES_PATH, new_callable=AsyncMock, return_value=None)
    def test_claude_proxy_non_json_content_type(
        self,
        mock_validate: AsyncMock,
        client: TestClient,
        mock_http_client_instance_fixture: MagicMock,
    ):
        """Test handling when Claude returns a non-JSON content type."""
        mock_backend_response = MagicMock(spec=httpx.Response)
        mock_backend_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        mock_backend_response.headers = httpx.Headers({"content-type": "text/html"})
        mock_backend_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "500 error", request=MagicMock(), response=mock_backend_response
            )
        )
        mock_backend_response.content = (
            b"<html><body>Internal Server Error</body></html>"
        )
        mock_http_client_instance_fixture.post.return_value = mock_backend_response

        headers = {HEADER_ANTHROPIC_API_KEY: "valid-key"}
        request_data = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(CLAUDE_PATH, headers=headers, json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.headers.get("content-type") == "text/html"
