import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse

import src.presentation.proxy_utils as proxy_utils
from src.core import app_state
from src.exceptions import (
    AuthenticationError,
    NotInitializedError,
    GuardrailedHTTPException,
    ValidationError,
)
from src.presentation.dependencies import get_loaded_policies
from src.shared import Action, Agent, Policy, SafetyCode, Status

router = APIRouter()
logger = logging.getLogger(__name__)


def _extract_claude_api_key(request: Request) -> Optional[str]:
    """Extracts Claude API key from the 'x-api-key' header."""
    api_key = request.headers.get(proxy_utils.HEADER_ANTHROPIC_API_KEY)
    return api_key


def _extract_input_messages_from_claude(
    request_data: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Converts Claude's 'messages' structure to Guardrailed's message list format.
    Handles both string content and list-of-content-blocks format, extracting text.
    """
    messages_for_validation = []
    claude_messages = request_data.get("messages")
    if not isinstance(claude_messages, list):
        logger.warning("Claude request data missing or invalid 'messages' list.")
        return []

    for msg in claude_messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")

        validator_role: Optional[str] = None
        if role == "user":
            validator_role = Agent.USER
        elif role == "assistant":
            validator_role = Agent.ASSISTANT
        else:
            logger.debug(f"Ignoring message with unknown Claude role: {role}")
            continue

        text_content = ""
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        text_content += text + "\n"
            text_content = text_content.strip()
        else:
            logger.warning(
                f"Unsupported content type in Claude message: {type(content)}"
            )
            continue

        if text_content:
            messages_for_validation.append(
                {"role": validator_role, "content": text_content}
            )
        else:
            logger.debug(f"No text content extracted for role {role}")

    if not messages_for_validation:
        logger.warning(
            "Could not extract any valid text content from Claude 'messages'."
        )

    return messages_for_validation


def _extract_output_message_from_claude(
    response_data: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """
    Extracts the primary text response from Claude's output structure ('content' array).
    """
    try:
        content_blocks = response_data.get("content")
        if not isinstance(content_blocks, list):
            logger.warning("Claude response data missing or invalid 'content' list.")
            return None

        text_content = ""
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_content += text + "\n"

        text_content = text_content.strip()
        if text_content:
            return {"role": Agent.ASSISTANT, "content": text_content}

    except Exception as e:
        logger.error(
            f"Error parsing Claude response structure for output validation: {e}",
            exc_info=True,
        )

    logger.warning(
        "Could not extract text content from Claude response 'content' blocks."
    )
    return None


@router.post(
    "/messages",
    tags=["Claude Proxy"],
    summary="Proxy endpoint for Anthropic Claude Messages API with Guardrails",
    description=(
        "Accepts Anthropic-compatible requests, validates input/output using policies, "
        "forwards if safe, and returns 200 OK with modified content/headers on block."
    ),
)
async def claude_messages_proxy(
    request: Request,
    loaded_policies: List[Policy] = Depends(get_loaded_policies),
) -> Response:
    start_time = time.time()
    request_id = getattr(
        request.state,
        "request_id",
        getattr(asyncio.current_task(), "request_id", "unknown-claude-proxy"),
    )
    log_extra = {"request_id": request_id}
    logger.info("Claude Proxy request received", extra=log_extra)

    request_data: Dict[str, Any] = {}
    claude_response_data: Optional[Dict[str, Any]] = None
    backend_response: Optional[httpx.Response] = None

    try:
        claude_api_key = _extract_claude_api_key(request)
        if not claude_api_key:
            logger.warning("Missing Claude API Key (x-api-key header)", extra=log_extra)
            raise AuthenticationError("Missing Anthropic API Key.")

        try:
            request_data = await request.json()
            if not isinstance(request_data, dict):
                raise ValidationError("Request body must be a JSON object.")
        except json.JSONDecodeError as json_exc:
            logger.error(f"Failed to parse request JSON: {json_exc}", extra=log_extra)
            raise json_exc

        input_messages = _extract_input_messages_from_claude(request_data)
        # Claude API doesn't typically use user ID in the same way as OpenAI
        user_id = (
            request_data.get("metadata", {}).get("user_id")
            if isinstance(request_data.get("metadata"), dict)
            else None
        )
        is_streaming = request_data.get("stream", False)

        model_requested = request_data.get("model", "unknown")
        log_extra["model_requested"] = model_requested

        if not input_messages:
            raise ValidationError(
                "Could not extract valid message content from 'messages'."
            )

        logger.debug(
            f"Validating {len(input_messages)} input messages.", extra=log_extra
        )

        input_status: Optional[Status] = await proxy_utils._validate_messages(
            messages_to_validate=input_messages,
            policies=loaded_policies,
            user_id=user_id,
            request_id=request_id,
            validation_stage="input",
        )

        if input_status:
            status_code_hint = input_status.status
            log_extra_blocked = (
                proxy_utils._merge_log_extra(log_extra, input_status)
                if hasattr(proxy_utils, "_merge_log_extra")
                else {
                    **log_extra,
                    "safety_code": input_status.safety_code,
                    "action": input_status.action,
                }
            )

            if 500 <= status_code_hint < 600:
                logger.error(
                    f"Internal error during input validation: {input_status.message}",
                    extra=log_extra_blocked,
                )
                return JSONResponse(
                    status_code=status_code_hint,
                    content={
                        "safety_code": input_status.safety_code,
                        "message": input_status.message,
                        "action": input_status.action,
                    },
                )
            else:
                logger.warning("Input blocked by guardrail", extra=log_extra_blocked)
                return proxy_utils.create_blocked_response(
                    provider="claude",
                    block_status=input_status,
                    original_request_data=request_data,
                )

        if is_streaming:
            logger.warning("Streaming requested but not supported.", extra=log_extra)
            raise GuardrailedHTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                message="Streaming responses are not yet supported by this proxy.",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.OVERRIDE.value,
            )

        logger.debug("Forwarding request to Claude.", extra=log_extra)

        if app_state.config is None:
            raise RuntimeError("App config is not initialized")

        claude_url = f"{app_state.config.claude_api_base_url}/v1/messages"
        headers = {
            proxy_utils.HEADER_ANTHROPIC_API_KEY: claude_api_key,
            proxy_utils.HEADER_ANTHROPIC_VERSION: app_state.config.claude_api_version,
            "Content-Type": "application/json",
        }
        beta_header = request.headers.get("anthropic-beta")
        if beta_header:
            headers["anthropic-beta"] = beta_header

        try:
            backend_response = await proxy_utils.http_client.post(
                claude_url, json=request_data, headers=headers
            )
            backend_response.raise_for_status()
            claude_response_data = backend_response.json()
            model_used = claude_response_data.get("model", "unknown")  # type: ignore[union-attr]
            log_extra["model_used"] = model_used
            logger.debug(
                f"Received OK response from Claude (Model: {model_used}).",
                extra=log_extra,
            )

        except httpx.TimeoutException:
            logger.error("Request to Claude timed out.", extra=log_extra)
            raise GuardrailedHTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                message="Request to Anthropic Claude timed out.",
                safety_code=SafetyCode.TIMEOUT,
                action=Action.RETRY.value,
            )
        except httpx.RequestError as req_err:
            logger.error(f"Network error contacting Claude: {req_err}", extra=log_extra)
            raise GuardrailedHTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                message=f"Network error communicating with Anthropic Claude: {req_err}",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.RETRY.value,
            )
        except json.JSONDecodeError as json_err:
            logger.error(
                f"Failed to parse Claude response as JSON: {json_err}",
                extra=log_extra,
            )
            raise GuardrailedHTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to parse response from Claude API: The response was not valid JSON.",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.RETRY.value,
            )

        llm_output_message = (
            _extract_output_message_from_claude(claude_response_data)
            if claude_response_data
            else None
        )

        if llm_output_message:
            logger.debug("Validating Claude output message.", extra=log_extra)
            output_status: Optional[Status] = await proxy_utils._validate_messages(
                messages_to_validate=[llm_output_message],
                policies=loaded_policies,
                user_id=user_id,
                request_id=request_id,
                validation_stage="output",
            )

            if output_status:
                status_code_hint = output_status.status
                log_extra_blocked = (
                    proxy_utils._merge_log_extra(log_extra, output_status)
                    if hasattr(proxy_utils, "_merge_log_extra")
                    else {
                        **log_extra,
                        "safety_code": output_status.safety_code,
                        "action": output_status.action,
                    }
                )

                if 500 <= status_code_hint < 600:
                    logger.error(
                        f"Internal error during output validation: {output_status.message}",
                        extra=log_extra_blocked,
                    )
                    return JSONResponse(
                        status_code=status_code_hint,
                        content={
                            "safety_code": output_status.safety_code,
                            "message": output_status.message,
                            "action": output_status.action,
                        },
                    )
                else:
                    logger.warning(
                        "Output blocked by guardrail", extra=log_extra_blocked
                    )
                    return proxy_utils.create_blocked_response(
                        provider="claude",
                        block_status=output_status,
                        original_request_data=request_data,
                        original_response_data=claude_response_data,
                    )
        else:
            if claude_response_data:
                logger.warning(
                    "Could not extract valid text from Claude response for output validation. Passing through original response.",
                    extra=log_extra,
                )
            else:
                logger.error(
                    "Claude response data is None despite successful HTTP call.",
                    extra=log_extra,
                )
                raise GuardrailedHTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    message="Internal server error processing Claude response.",
                    safety_code=SafetyCode.UNEXPECTED,
                )

        if backend_response is None or claude_response_data is None:
            logger.error(
                "Internal logic error: Backend response/data is None before successful return.",
                extra=log_extra,
            )
            raise GuardrailedHTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Internal server error processing response.",
                safety_code=SafetyCode.UNEXPECTED,
            )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"Claude Proxy request finished successfully or passed through. Elapsed: {elapsed_ms}ms",
            extra=log_extra,
        )
        forward_headers = dict(backend_response.headers)
        forward_headers.pop("content-encoding", None)
        forward_headers.pop("Content-Encoding", None)
        forward_headers.pop("content-length", None)
        forward_headers.pop("Content-Length", None)
        forward_headers.pop("transfer-encoding", None)
        forward_headers.pop("Transfer-Encoding", None)
        return Response(
            content=backend_response.content,
            status_code=backend_response.status_code,
            headers=forward_headers,
            media_type=forward_headers.get("content-type", "application/json"),
        )

    except (ValidationError, AuthenticationError, NotInitializedError) as client_err:
        logger.warning(
            f"Client/Setup error in Claude proxy: {client_err}",
            extra=log_extra,
            exc_info=False,
        )
        raise client_err
    except GuardrailedHTTPException as http_exc:
        logger.warning(
            f"GuardrailedHTTPException in Claude proxy route: {http_exc.detail} (Status: {http_exc.status_code})",
            extra=log_extra,
            exc_info=False,
        )
        raise http_exc
    except Exception as e:
        logger.critical(
            f"Unexpected error in Claude proxy route: {e}",
            extra=log_extra,
            exc_info=True,
        )
        raise
