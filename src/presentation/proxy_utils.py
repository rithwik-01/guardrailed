import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import status
from fastapi.responses import JSONResponse

from src.domain.validators import ContentValidator, ValidationContext
from src.exceptions import (
    NotInitializedError,
    ValidationError,
)
from src.shared import Action, Policy, SafetyCode, Status

http_client = httpx.AsyncClient(timeout=60.0)
helper_logger = logging.getLogger("src.presentation.proxy_utils")
HEADER_GUARDRAILED_BLOCKED = "X-Guardrailed-Blocked"
HEADER_GUARDRAILED_SAFETY_CODE = "X-Guardrailed-Safety-Code"
HEADER_GUARDRAILED_ACTION = "X-Guardrailed-Action"
HEADER_GUARDRAILED_MESSAGE = "X-Guardrailed-Message"
HEADER_ANTHROPIC_API_KEY = "X-Api-Key"
HEADER_ANTHROPIC_VERSION = "anthropic-version"


async def _validate_messages(
    messages_to_validate: List[Dict[str, str]],
    policies: List[Policy],
    user_id: Optional[str],
    request_id: str,
    validation_stage: str,
) -> Optional[Status]:
    """
    Helper to run validation and return Status if unsafe.
    Determines appropriate HTTP status code hint within the Status object.
    Returns None if validation passes.
    Raises exceptions for validation dependency errors.
    """
    if not messages_to_validate:
        helper_logger.debug(
            f"No messages to validate for stage: {validation_stage}",
            extra={"request_id": request_id},
        )
        return None

    context = ValidationContext(
        policies=policies, messages=messages_to_validate, user_id=user_id
    )
    content_validator = ContentValidator(context)

    try:
        validation_status: Status = await content_validator.validate_content()
        if validation_status.safety_code != SafetyCode.SAFE:
            final_status_code = validation_status.status
            if not (500 <= final_status_code < 600):
                if validation_status.safety_code in [
                    SafetyCode.UNEXPECTED,
                    SafetyCode.TIMEOUT,
                ]:
                    final_status_code = (
                        status.HTTP_503_SERVICE_UNAVAILABLE
                        if validation_status.safety_code == SafetyCode.TIMEOUT
                        else status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                else:
                    final_status_code = status.HTTP_400_BAD_REQUEST

            unsafe_status_with_code = Status(
                status=final_status_code,
                message=validation_status.message,
                safety_code=validation_status.safety_code,
                action=validation_status.action,
                processed_content=validation_status.processed_content,
            )

            helper_logger.warning(
                f"Validation failed at stage '{validation_stage}'. Code: {unsafe_status_with_code.safety_code}, Msg: {unsafe_status_with_code.message}, HTTP Status Hint: {final_status_code}",
                extra={"request_id": request_id},
            )
            return unsafe_status_with_code
        else:
            helper_logger.debug(
                f"Validation successful for stage: {validation_stage}",
                extra={"request_id": request_id},
            )
            return None
    except (ValidationError, NotInitializedError) as val_err:
        helper_logger.error(
            f"Validation dependency error at stage '{validation_stage}': {val_err}",
            extra={"request_id": request_id},
            exc_info=False,
        )
        raise
    except Exception as e:
        helper_logger.error(
            f"Unexpected error during validation process at stage '{validation_stage}': {e}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        return Status(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"Internal error during content validation process ({validation_stage}).",
            safety_code=SafetyCode.UNEXPECTED,
            action=Action.OVERRIDE.value,
        )


def _create_blocked_response_headers(block_status: Status) -> Dict[str, str]:
    """Creates standard headers for a blocked response."""
    sanitized_message = block_status.message.replace("\n", " ").replace("\r", "")
    headers = {
        HEADER_GUARDRAILED_BLOCKED: "true",
        HEADER_GUARDRAILED_SAFETY_CODE: str(block_status.safety_code),
        HEADER_GUARDRAILED_ACTION: str(
            block_status.action
            if block_status.action is not None
            else Action.OVERRIDE.value
        ),
        HEADER_GUARDRAILED_MESSAGE: sanitized_message,
    }
    return headers


def _create_openai_blocked_response_body(
    block_status: Status,
    original_request_data: Dict[str, Any],
    original_response_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Creates an OpenAI-compatible JSON body for a blocked request."""
    model_used = original_request_data.get("model", "unknown_model")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    response_id = f"guardrailed-blocked-{int(time.time())}"
    created = int(time.time())
    if original_response_data:
        usage = original_response_data.get("usage", usage)
        response_id = original_response_data.get("id", response_id)
        created = original_response_data.get("created", created)
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": block_status.message,
                },
                "logprobs": None,
                "finish_reason": "content_filter",
            }
        ],
        "usage": usage,
        "system_fingerprint": "fp_guardrailed_blocked",
    }


def _create_gemini_blocked_response_body(
    block_status: Status,
    original_response_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Creates a Gemini-compatible JSON body for a blocked request."""
    block_reason = "OTHER"
    if original_response_data:
        block_reason = original_response_data.get("promptFeedback", {}).get(
            "blockReason", "OTHER"
        )
    gemini_category = "HARM_CATEGORY_DANGEROUS_CONTENT"
    if block_status.safety_code == SafetyCode.PROFANE:
        gemini_category = "HARM_CATEGORY_HARASSMENT"
    elif block_status.safety_code == SafetyCode.PII_DETECTED:
        gemini_category = "HARM_CATEGORY_DANGEROUS_CONTENT"
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": block_status.message}], "role": "model"},
                "finishReason": "SAFETY",
                "index": 0,
                "safetyRatings": [{"category": gemini_category, "probability": "HIGH"}],
            }
        ],
        "promptFeedback": {
            "blockReason": block_reason,
            "safetyRatings": [{"category": gemini_category, "probability": "HIGH"}],
        },
    }


def _create_claude_blocked_response_body(block_status: Status) -> Dict[str, Any]:
    """Creates an Anthropic-compatible JSON body for a blocked request."""
    error_type = "invalid_request_error"
    if block_status.safety_code == SafetyCode.UNEXPECTED:
        error_type = "api_error"
    elif block_status.safety_code == SafetyCode.TIMEOUT:
        error_type = "timeout_error"
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": block_status.message,
        },
    }


def create_blocked_response(
    provider: str,
    block_status: Status,
    original_request_data: Dict[str, Any],
    original_response_data: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """
    Creates a JSONResponse for blocked requests based on provider API format.

    Determines the final HTTP status code based on the block_status:
        - Returns 5xx if block_status.status indicates an internal validation
          error or timeout during validation itself.
        - Returns 200 OK for standard policy violations (e.g., PII, Profanity)
          detected by Guardrailed guardrails, mimicking native filter behavior.
        - Returns 500 for unsupported providers.

    Includes standard 'X-Guardrailed-*' headers in the response.
    The response body is modified to indicate the block according to the provider's API.
    """
    headers = _create_blocked_response_headers(block_status)
    body: Dict[str, Any]
    http_status_code = status.HTTP_200_OK

    if provider == "openai":
        body = _create_openai_blocked_response_body(
            block_status, original_request_data, original_response_data
        )
    elif provider == "gemini":
        body = _create_gemini_blocked_response_body(
            block_status, original_response_data
        )
    elif provider == "claude":
        body = _create_claude_blocked_response_body(block_status)
    else:
        helper_logger.error(
            f"Unsupported provider '{provider}' for blocked response creation."
        )
        body = {
            "safety_code": SafetyCode.UNEXPECTED,
            "message": f"Internal configuration error: Unsupported provider '{provider}'.",
            "action": Action.OVERRIDE.value,
        }
        http_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return JSONResponse(
            status_code=http_status_code,
            content=body,
            headers=headers,
        )

    if status.HTTP_500_INTERNAL_SERVER_ERROR <= block_status.status < 600:
        http_status_code = block_status.status
        helper_logger.info(
            f"Overriding blocked response status to {http_status_code} based on internal validation error hint."
        )

    elif block_status.status == status.HTTP_400_BAD_REQUEST:
        helper_logger.info("Policy violation detected.")

    helper_logger.info(f"Final status for blocked response: {http_status_code}")
    return JSONResponse(
        status_code=http_status_code,
        content=body,
        headers=headers,
    )
