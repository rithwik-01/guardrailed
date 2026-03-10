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
    GuardrailedHTTPException,
    ValidationError,
)
from src.presentation.dependencies import get_loaded_policies
from src.shared import Action, Policy, SafetyCode, Status

router = APIRouter()
logger = logging.getLogger(__name__)
_RESERVED_LOG_KEYS = {
    "message",
    "msg",
    "levelname",
    "levelno",
}


def _merge_log_extra(base: Dict[str, Any], status_obj: Status) -> Dict[str, Any]:
    """Combine base logging context with fields from a Status object, omitting
    any key that would collide with reserved LogRecord attributes."""
    status_dict = {
        k: v for k, v in status_obj.__dict__.items() if k not in _RESERVED_LOG_KEYS
    }
    return {**base, **status_dict}


@router.post(
    "/chat/completions",
    tags=["OpenAI Proxy"],
    summary="Proxy endpoint for OpenAI Chat Completions with Guardrails",
    description=(
        "Accepts OpenAI‑compatible requests, validates input/output with guardrail "
        "policies, forwards if safe, and can return a 200‑OK stub when content "
        "is blocked."
    ),
)
async def openai_chat_completions_proxy(
    request: Request,
    loaded_policies: List[Policy] = Depends(get_loaded_policies),
) -> Response:
    start_time = time.time()
    request_id = getattr(
        request.state,
        "request_id",
        getattr(asyncio.current_task(), "request_id", "unknown-openai-proxy"),
    )
    log_extra = {"request_id": request_id}
    logger.info("OpenAI Proxy request received", extra=log_extra)

    request_data: Dict[str, Any] = {}
    openai_response_data: Optional[Dict[str, Any]] = None

    try:
        auth_header = request.headers.get("Authorization")
        openai_api_key: Optional[str] = None
        if auth_header and auth_header.startswith("Bearer "):
            openai_api_key = auth_header[len("Bearer ") :]

        if not openai_api_key:
            logger.warning("Missing or invalid Authorization header", extra=log_extra)
            raise AuthenticationError("Missing or invalid OpenAI API Key.")

        try:
            request_data = await request.json()
            if not isinstance(request_data, dict):
                raise ValidationError("Request body must be a JSON object.")
        except json.JSONDecodeError as json_exc:
            logger.error(f"Failed to parse request JSON: {json_exc}", extra=log_extra)
            raise json_exc

        input_messages = request_data.get("messages", [])
        user_id = request_data.get("user")
        is_streaming = request_data.get("stream", False)

        if not input_messages or not isinstance(input_messages, list):
            raise ValidationError("Invalid or missing 'messages' list.")

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
            if 500 <= input_status.status < 600:
                logger.error(
                    f"Internal error during input validation: {input_status.message}",
                    extra=log_extra,
                )
                return JSONResponse(
                    status_code=input_status.status,
                    content={
                        "safety_code": input_status.safety_code,
                        "message": input_status.message,
                        "action": input_status.action,
                    },
                )
            logger.warning(
                "Input blocked by guardrail",
                extra=_merge_log_extra(log_extra, input_status),
            )
            return proxy_utils.create_blocked_response(
                provider="openai",
                block_status=input_status,
                original_request_data=request_data,
                original_response_data=None,
            )

        if is_streaming:
            logger.warning("Streaming requested but not supported.", extra=log_extra)
            raise GuardrailedHTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                message="Streaming responses are not yet supported by this proxy.",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.OVERRIDE.value,
            )

        logger.debug("Forwarding request to OpenAI", extra=log_extra)

        if app_state.config is None:
            raise RuntimeError("App config is not initialized")

        openai_url = f"{app_state.config.openai_api_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

        backend_response: Optional[httpx.Response] = None
        try:
            backend_response = await proxy_utils.http_client.post(
                openai_url, json=request_data, headers=headers
            )
            backend_response.raise_for_status()
            openai_response_data = backend_response.json()
            logger.debug("Received OK response from OpenAI", extra=log_extra)

        except httpx.TimeoutException:
            logger.error("Request to OpenAI timed out", extra=log_extra)
            raise GuardrailedHTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                message="Request to OpenAI timed out.",
                safety_code=SafetyCode.TIMEOUT,
                action=Action.RETRY.value,
            )
        except httpx.RequestError as req_err:
            logger.error(f"Network error contacting OpenAI: {req_err}", extra=log_extra)
            raise GuardrailedHTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                message=f"Network error communicating with OpenAI: {req_err}",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.RETRY.value,
            )
        llm_msg: Optional[Dict[str, str]] = None
        if openai_response_data:
            try:
                if (
                    "choices" in openai_response_data
                    and isinstance(openai_response_data["choices"], list)
                    and openai_response_data["choices"]
                    and "message" in openai_response_data["choices"][0]
                ):
                    msg = openai_response_data["choices"][0]["message"]
                    if isinstance(msg, dict) and {"role", "content"} <= msg.keys():
                        llm_msg = msg
            except Exception as parse_err:
                logger.error(
                    f"Error parsing OpenAI response structure for output validation: {parse_err}",
                    extra=log_extra,
                    exc_info=True,
                )
                raise GuardrailedHTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    message="Failed to parse successful LLM response.",
                    safety_code=SafetyCode.UNEXPECTED,
                    action=Action.OVERRIDE.value,
                )

        if llm_msg:
            output_status: Optional[Status] = await proxy_utils._validate_messages(
                messages_to_validate=[llm_msg],
                policies=loaded_policies,
                user_id=user_id,
                request_id=request_id,
                validation_stage="output",
            )

            if output_status:
                if 500 <= output_status.status < 600:
                    logger.error(
                        f"Internal error during output validation: "
                        f"{output_status.message}",
                        extra=log_extra,
                    )
                    return JSONResponse(
                        status_code=output_status.status,
                        content={
                            "safety_code": output_status.safety_code,
                            "message": output_status.message,
                            "action": output_status.action,
                        },
                    )

                logger.warning(
                    "Output blocked by guardrail",
                    extra=_merge_log_extra(log_extra, output_status),
                )
                return proxy_utils.create_blocked_response(
                    provider="openai",
                    block_status=output_status,
                    original_request_data=request_data,
                    original_response_data=openai_response_data,
                )
        else:
            if (
                backend_response is not None
                and backend_response.status_code >= 200
                and backend_response.status_code < 300
            ):
                if openai_response_data is not None:
                    logger.debug(
                        "Could not extract LLM message for output validation. Passing original response.",
                        extra=log_extra,
                    )
            else:
                pass

        if backend_response is None or openai_response_data is None:
            logger.error(
                "Internal logic error: backend_response or response_data is None after processing",
                extra=log_extra,
            )
            raise GuardrailedHTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Internal server error processing response.",
            )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"OpenAI Proxy finished successfully or passed through in {elapsed_ms}ms",
            extra=log_extra,
        )
        return Response(
            content=json.dumps(openai_response_data),
            status_code=backend_response.status_code,
            media_type=backend_response.headers.get("content-type", "application/json"),
        )

    except GuardrailedHTTPException as th_err:
        logger.warning(
            f"GuardrailedHTTPException raised in OpenAI proxy route: {th_err.detail}",
            extra=log_extra,
        )
        raise th_err
