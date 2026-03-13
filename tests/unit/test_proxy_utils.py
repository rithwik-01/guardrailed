import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.responses import JSONResponse

from src.exceptions import NotInitializedError, ValidationError
from src.presentation.proxy_utils import (
    HEADER_GUARDRAILED_ACTION,
    HEADER_GUARDRAILED_BLOCKED,
    HEADER_GUARDRAILED_MESSAGE,
    HEADER_GUARDRAILED_SAFETY_CODE,
    _create_blocked_response_headers,
    _create_gemini_blocked_response_body,
    _create_openai_blocked_response_body,
    _validate_messages,
    create_blocked_response,
)
from src.shared import (
    Action,
    Policy,
    PolicyType,
    Result,
    SafetyCode,
    Status,
)


@pytest.fixture
def mock_content_validator():
    """Fixture for a mocked ContentValidator instance."""
    validator = MagicMock(spec=True)
    validator.validate_content = AsyncMock()
    return validator


@pytest.fixture
def sample_policies():
    """Basic list of policies for context."""
    return [
        Policy(
            id=PolicyType.PROFANITY.value,
            name="Profanity",
            state=True,
            action=Action.OVERRIDE.value,
            is_user_policy=True,
            is_llm_policy=True,
            message="Profanity detected",
            pii_entities=None,
            pii_threshold=0.5,
            pii_categories=None,
            threshold=0.7,
            protected_prompts=None,
            prompt_leakage_threshold=0.85,
            locations=None,
            persons=None,
            competitors=None,
            metadata={},
        )
    ]


@pytest.fixture
def sample_messages():
    """Sample messages list."""
    return [{"role": "user", "content": "Test content"}]


@pytest.mark.asyncio
@patch("src.presentation.proxy_utils.ContentValidator")
async def test_validate_messages_safe(
    mock_validator_cls,
    mock_content_validator,
    sample_policies,
    sample_messages,
):
    """Test _validate_messages returns None when validation is safe."""
    mock_content_validator.validate_content.return_value = Result.safe_result()
    mock_validator_cls.return_value = mock_content_validator

    result = await _validate_messages(
        messages_to_validate=sample_messages,
        policies=sample_policies,
        user_id="u1",
        request_id="req1",
        validation_stage="input",
    )

    assert result is None
    mock_validator_cls.assert_called_once()
    mock_content_validator.validate_content.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.presentation.proxy_utils.ContentValidator")
async def test_validate_messages_policy_violation(
    mock_validator_cls,
    mock_content_validator,
    sample_policies,
    sample_messages,
):
    """Test _validate_messages returns Status with 400 hint for policy violation."""
    unsafe_status = Status(
        status=status.HTTP_200_OK,
        message="Profanity detected",
        safety_code=SafetyCode.PROFANE,
        action=Action.OVERRIDE.value,
    )
    mock_content_validator.validate_content.return_value = unsafe_status
    mock_validator_cls.return_value = mock_content_validator

    result = await _validate_messages(
        messages_to_validate=sample_messages,
        policies=sample_policies,
        user_id="u1",
        request_id="req2",
        validation_stage="output",
    )

    assert isinstance(result, Status)
    assert result.safety_code == SafetyCode.PROFANE
    assert result.action == Action.OVERRIDE.value
    assert result.status == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
@patch("src.presentation.proxy_utils.ContentValidator")
async def test_validate_messages_internal_error_status(
    mock_validator_cls,
    mock_content_validator,
    sample_policies,
    sample_messages,
):
    """Test _validate_messages returns Status with 500 hint for internal errors."""
    internal_error_status = Status(
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="Validator internal error",
        safety_code=SafetyCode.UNEXPECTED,
        action=Action.OVERRIDE.value,
    )
    mock_content_validator.validate_content.return_value = internal_error_status
    mock_validator_cls.return_value = mock_content_validator

    result = await _validate_messages(
        messages_to_validate=sample_messages,
        policies=sample_policies,
        user_id="u1",
        request_id="req3",
        validation_stage="input",
    )

    assert isinstance(result, Status)
    assert result.safety_code == SafetyCode.UNEXPECTED
    assert result.status == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
@patch("src.presentation.proxy_utils.ContentValidator")
async def test_validate_messages_raises_validation_error(
    mock_validator_cls,
    mock_content_validator,
    sample_policies,
    sample_messages,
):
    """Test _validate_messages re-raises ValidationError from validator."""
    mock_content_validator.validate_content.side_effect = ValidationError("Bad context")
    mock_validator_cls.return_value = mock_content_validator

    with pytest.raises(ValidationError, match="Bad context"):
        await _validate_messages(
            messages_to_validate=sample_messages,
            policies=sample_policies,
            user_id="u1",
            request_id="req4",
            validation_stage="input",
        )
    mock_content_validator.validate_content.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.presentation.proxy_utils.ContentValidator")
async def test_validate_messages_raises_not_initialized(
    mock_validator_cls,
    mock_content_validator,
    sample_policies,
    sample_messages,
):
    """Test _validate_messages re-raises NotInitializedError from validator."""
    mock_content_validator.validate_content.side_effect = NotInitializedError("Model")
    mock_validator_cls.return_value = mock_content_validator

    with pytest.raises(NotInitializedError, match="Model"):
        await _validate_messages(
            messages_to_validate=sample_messages,
            policies=sample_policies,
            user_id="u1",
            request_id="req5",
            validation_stage="output",
        )
    mock_content_validator.validate_content.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.presentation.proxy_utils.ContentValidator")
async def test_validate_messages_unexpected_exception(
    mock_validator_cls,
    mock_content_validator,
    sample_policies,
    sample_messages,
):
    """Test _validate_messages returns internal error status on unexpected exceptions."""
    mock_content_validator.validate_content.side_effect = Exception(
        "Something went wrong"
    )
    mock_validator_cls.return_value = mock_content_validator

    result = await _validate_messages(
        messages_to_validate=sample_messages,
        policies=sample_policies,
        user_id="u1",
        request_id="req6",
        validation_stage="input",
    )

    assert isinstance(result, Status)
    assert result.safety_code == SafetyCode.UNEXPECTED
    assert result.status == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Internal error during content validation process" in result.message
    assert result.action == Action.OVERRIDE.value
    mock_content_validator.validate_content.assert_awaited_once()


def test_create_blocked_response_headers():
    """Test creation of standard block headers."""
    block_status = Status(
        status=400,
        message="PII Detected\nHandle with care.",
        safety_code=SafetyCode.PII_DETECTED,
        action=Action.OVERRIDE.value,
    )
    headers = _create_blocked_response_headers(block_status)

    assert headers[HEADER_GUARDRAILED_BLOCKED] == "true"
    assert headers[HEADER_GUARDRAILED_SAFETY_CODE] == str(SafetyCode.PII_DETECTED)
    assert headers[HEADER_GUARDRAILED_ACTION] == str(Action.OVERRIDE.value)
    assert headers[HEADER_GUARDRAILED_MESSAGE] == "PII Detected Handle with care."


def test_create_openai_blocked_response_body():
    """Test creation of OpenAI-compatible blocked response body."""
    block_status = Status(
        status=400,
        message="Toxic input",
        safety_code=SafetyCode.PROFANE,
        action=Action.OVERRIDE.value,
    )
    request_data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "..."}]}
    original_resp = {
        "id": "chatcmpl-original123",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    body_input_block = _create_openai_blocked_response_body(block_status, request_data)
    assert body_input_block["object"] == "chat.completion"
    assert body_input_block["model"] == "gpt-4o"  # From request
    assert body_input_block["choices"][0]["message"]["content"] == "Toxic input"
    assert body_input_block["choices"][0]["finish_reason"] == "content_filter"
    assert body_input_block["usage"]["total_tokens"] == 0
    assert body_input_block["id"].startswith("guardrailed-blocked-")
    assert body_input_block["system_fingerprint"] == "fp_guardrailed_blocked"

    body_output_block = _create_openai_blocked_response_body(
        block_status, request_data, original_resp
    )
    assert body_output_block["model"] == "gpt-4o"
    assert body_output_block["choices"][0]["message"]["content"] == "Toxic input"
    assert body_output_block["usage"]["total_tokens"] == 15
    assert body_output_block["id"] == "chatcmpl-original123"
    assert body_output_block["created"] == 1677652288


def test_create_gemini_blocked_response_body():
    """Test creation of Gemini-compatible blocked response body."""
    block_message_text = "PII problem"
    block_status_pii = Status(
        status=400,
        message=block_message_text,
        safety_code=SafetyCode.PII_DETECTED,
        action=Action.OVERRIDE.value,
    )
    block_status_profane = Status(
        status=400,
        message="Toxic problem",
        safety_code=SafetyCode.PROFANE,
        action=Action.OVERRIDE.value,
    )
    original_resp_blocked = {
        "candidates": [{"finishReason": "STOP", "index": 0, "safetyRatings": []}],
        "promptFeedback": {
            "blockReason": "SAFETY",
            "safetyRatings": [],
        },  # Has blockReason
    }
    original_resp_ok = {
        "candidates": [{"finishReason": "STOP", "index": 0, "safetyRatings": []}],
        "promptFeedback": {"safetyRatings": []},  # No blockReason
    }

    body_input_block = _create_gemini_blocked_response_body(block_status_pii)
    assert (
        body_input_block["candidates"][0]["content"]["parts"][0]["text"]
        == block_message_text
    )
    assert body_input_block["candidates"][0]["finishReason"] == "SAFETY"
    assert (
        body_input_block["candidates"][0]["safetyRatings"][0]["category"]
        == "HARM_CATEGORY_DANGEROUS_CONTENT"
    )
    assert body_input_block["promptFeedback"]["blockReason"] == "OTHER"

    body_input_block_profane = _create_gemini_blocked_response_body(
        block_status_profane
    )
    assert (
        body_input_block_profane["candidates"][0]["safetyRatings"][0]["category"]
        == "HARM_CATEGORY_HARASSMENT"
    )
    assert (
        body_input_block_profane["candidates"][0]["content"]["parts"][0]["text"]
        == "Toxic problem"
    )

    body_output_block_orig_ok = _create_gemini_blocked_response_body(
        block_status_pii, original_resp_ok
    )
    assert (
        body_output_block_orig_ok["candidates"][0]["content"]["parts"][0]["text"]
        == block_message_text
    )
    assert body_output_block_orig_ok["promptFeedback"]["blockReason"] == "OTHER"

    body_output_block_orig_blocked = _create_gemini_blocked_response_body(
        block_status_pii, original_resp_blocked
    )
    assert (
        body_output_block_orig_blocked["candidates"][0]["content"]["parts"][0]["text"]
        == block_message_text
    )
    assert body_output_block_orig_blocked["promptFeedback"]["blockReason"] == "SAFETY"


@patch("src.presentation.proxy_utils._create_openai_blocked_response_body")
@patch("src.presentation.proxy_utils._create_blocked_response_headers")
def test_create_blocked_response_openai(mock_create_headers, mock_create_body):
    """Test create_blocked_response for OpenAI provider."""
    block_status = Status(status=200, message="Msg", safety_code=10, action=0)
    req_data = {"model": "m"}
    mock_headers = {"X-Guardrailed-Blocked": "true"}
    mock_body = {"id": "1", "choices": []}
    mock_create_headers.return_value = mock_headers
    mock_create_body.return_value = mock_body

    response = create_blocked_response("openai", block_status, req_data)

    assert isinstance(response, JSONResponse)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.body.decode()) == mock_body
    assert response.headers["content-type"] == "application/json"
    for k, v in mock_headers.items():
        assert response.headers[k] == v

    mock_create_headers.assert_called_once_with(block_status)
    mock_create_body.assert_called_once_with(block_status, req_data, None)


@patch("src.presentation.proxy_utils._create_gemini_blocked_response_body")
@patch("src.presentation.proxy_utils._create_blocked_response_headers")
def test_create_blocked_response_gemini(mock_create_headers, mock_create_body):
    """Test create_blocked_response for Gemini provider."""
    block_status = Status(status=200, message="Msg", safety_code=10, action=0)
    req_data = {"contents": []}
    orig_resp = {"candidates": []}
    mock_headers = {"X-Guardrailed-Blocked": "true"}
    mock_body = {"candidates": []}
    mock_create_headers.return_value = mock_headers
    mock_create_body.return_value = mock_body

    response = create_blocked_response("gemini", block_status, req_data, orig_resp)
    assert isinstance(response, JSONResponse)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.body.decode()) == mock_body
    for k, v in mock_headers.items():
        assert response.headers[k] == v

    mock_create_headers.assert_called_once_with(block_status)
    mock_create_body.assert_called_once_with(block_status, orig_resp)


@patch("src.presentation.proxy_utils.helper_logger")
def test_create_blocked_response_unsupported(mock_logger):
    """Test create_blocked_response handles unsupported providers."""
    block_status = Status(status=400, message="Msg", safety_code=10, action=0)
    req_data = {}

    response = create_blocked_response("unknown_provider", block_status, req_data)

    assert isinstance(response, JSONResponse)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.headers[HEADER_GUARDRAILED_BLOCKED] == "true"
    response_body = json.loads(response.body.decode())
    assert response_body["safety_code"] == SafetyCode.UNEXPECTED
    assert "Unsupported provider" in response_body["message"]
    assert response_body["action"] == Action.OVERRIDE.value

    mock_logger.error.assert_called_once()
    assert (
        "Unsupported provider 'unknown_provider'" in mock_logger.error.call_args[0][0]
    )
