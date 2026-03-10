import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse

from src.core import app_state
from src.exceptions import (
    AuthenticationError,
    NotInitializedError,
    GuardrailedHTTPException,
    ValidationError,
)
from src.presentation.dependencies import get_loaded_policies
from src.presentation.proxy_utils import (
    _validate_messages,
    create_blocked_response,
    http_client,
)
from src.shared import Action, Agent, Policy, SafetyCode, Status

router = APIRouter()
logger = logging.getLogger(__name__)


def _extract_gemini_api_key(request: Request) -> Optional[str]:
    """Extracts Gemini API key from header or query param."""
    api_key = request.headers.get("x-goog-api-key")
    if api_key:
        return api_key
    api_key = request.query_params.get("key")
    return api_key


def _extract_input_messages_from_gemini(
    request_data: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Converts Gemini's 'contents' structure to Guardrailed's message list for validation."""
    messages = []
    contents = request_data.get("contents")
    if not isinstance(contents, list):
        logger.warning("Gemini request data missing 'contents' list.")
        return []

    for content_item in contents:
        if not isinstance(content_item, dict):
            continue
        role = content_item.get("role", Agent.USER)
        if role == "model":
            role = Agent.ASSISTANT
        elif role not in [Agent.USER, Agent.ASSISTANT]:
            logger.debug(
                f"Treating unknown Gemini role '{role}' as '{Agent.USER}' for validation."
            )
            role = Agent.USER

        parts = content_item.get("parts")
        if isinstance(parts, list):
            full_text = ""
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    full_text += part["text"] + "\n"
            full_text = full_text.strip()
            if full_text:
                messages.append({"role": role, "content": full_text})
        elif isinstance(content_item.get("text"), str):
            messages.append({"role": role, "content": content_item["text"]})

    if not messages:
        logger.warning("Could not extract any valid text parts from Gemini 'contents'.")
    return messages


def _extract_output_message_from_gemini(
    response_data: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """Extracts the primary text response from Gemini's output structure."""
    try:
        candidates = response_data.get("candidates")
        if isinstance(candidates, list) and len(candidates) > 0:
            first_candidate = candidates[0]
            finish_reason = first_candidate.get("finishReason")
            if finish_reason in ["SAFETY", "RECITATION", "OTHER"]:
                logger.debug(
                    f"Gemini candidate finished due to '{finish_reason}', no content extracted for validation."
                )
                return None

            if isinstance(first_candidate, dict):
                content = first_candidate.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, list):
                        full_text = ""
                        for part in parts:
                            if isinstance(part, dict) and isinstance(
                                part.get("text"), str
                            ):
                                full_text += part["text"] + "\n"
                        full_text = full_text.strip()
                        if full_text:
                            return {"role": Agent.ASSISTANT, "content": full_text}
    except Exception as e:
        logger.error(
            f"Error parsing Gemini response structure for output validation: {e}",
            exc_info=True,
        )
    logger.warning("Could not extract text content from Gemini candidate.")
    return None


@router.post(
    "/models/{model_name:path}:generateContent",
    tags=["Gemini Proxy"],
    summary="Proxy endpoint for Google Gemini generateContent with Guardrails",
    description="Accepts Gemini SDK requests, validates input/output using policies, forwards if safe, returns 200 OK with modified content/headers on block.",
)
async def gemini_generate_content_proxy(
    request: Request,
    model_name: str,
    loaded_policies: List[Policy] = Depends(get_loaded_policies),
) -> Response:
    start_time = time.time()
    request_id = getattr(
        request.state,
        "request_id",
        getattr(asyncio.current_task(), "request_id", "unknown-gemini-proxy"),
    )
    log_extra = {"request_id": request_id, "model_name": model_name}
    logger.info("Gemini Proxy request received", extra=log_extra)
    request_data: Dict[str, Any] = {}
    gemini_response_data: Optional[Dict[str, Any]] = None
    backend_response: Optional[httpx.Response] = None

    try:
        gemini_api_key = _extract_gemini_api_key(request)
        if not gemini_api_key:
            logger.warning("Missing Gemini API Key", extra=log_extra)
            raise AuthenticationError("Missing Gemini API Key.")

        try:
            request_data = await request.json()
            if not isinstance(request_data, dict):
                raise ValidationError("Request body must be a JSON object.")
        except json.JSONDecodeError as json_exc:
            logger.error(f"Failed to parse request JSON: {json_exc}", extra=log_extra)
            raise ValidationError(f"Invalid JSON payload: {json_exc}") from json_exc

        input_messages = _extract_input_messages_from_gemini(request_data)
        user_id = None
        is_streaming = ":streamGenerateContent" in request.url.path

        if not input_messages:
            raise ValidationError(
                "Could not extract valid message content from 'contents'."
            )

        logger.debug(
            f"Validating {len(input_messages)} input messages.", extra=log_extra
        )

        input_validation_status: Optional[Status] = await _validate_messages(
            messages_to_validate=input_messages,
            policies=loaded_policies,
            user_id=user_id,
            request_id=request_id,
            validation_stage="input",
        )

        if input_validation_status:
            if 500 <= input_validation_status.status < 600:
                logger.error(
                    f"Internal error during input validation: {input_validation_status.message}",
                    extra=log_extra,
                )
                return JSONResponse(
                    status_code=input_validation_status.status,
                    content={
                        "safety_code": input_validation_status.safety_code,
                        "message": input_validation_status.message,
                        "action": input_validation_status.action,
                    },
                )
            else:
                logger.warning(
                    f"Input blocked by guardrail. Policy msg: {input_validation_status.message}",
                    extra=log_extra,
                )
                return create_blocked_response(
                    provider="gemini",
                    block_status=input_validation_status,
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

        logger.debug("Forwarding request to Gemini.", extra=log_extra)
        if app_state.config is None:
            raise RuntimeError("App config is not initialized")
        gemini_url = f"{app_state.config.gemini_api_base_url}/{app_state.config.gemini_api_version}/models/{model_name}:generateContent"
        params = {"key": gemini_api_key}
        headers = {"Content-Type": "application/json"}

        try:
            backend_response = await http_client.post(
                gemini_url, json=request_data, params=params, headers=headers
            )
            backend_response.raise_for_status()
            gemini_response_data = backend_response.json()
            logger.debug("Received OK response from Gemini.", extra=log_extra)

        except httpx.TimeoutException:
            logger.error("Request to Gemini timed out.", extra=log_extra)
            raise GuardrailedHTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                message="Request to Gemini timed out.",
                safety_code=SafetyCode.TIMEOUT,
                action=Action.RETRY.value,
            )
        except httpx.RequestError as req_err:
            logger.error(f"Network error contacting Gemini: {req_err}", extra=log_extra)
            raise GuardrailedHTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                message=f"Network error communicating with Gemini: {req_err}",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.RETRY.value,
            )
        except httpx.HTTPStatusError as http_err:
            logger.warning(
                f"Gemini API returned error status {http_err.response.status_code}. Forwarding response.",
                extra=log_extra,
            )
            return Response(
                content=http_err.response.content,
                status_code=http_err.response.status_code,
                media_type=http_err.response.headers.get(
                    "content-type", "application/json"
                ),
            )

        llm_output_message = (
            _extract_output_message_from_gemini(gemini_response_data)
            if gemini_response_data
            else None
        )

        if llm_output_message:
            logger.debug("Validating Gemini output message.", extra=log_extra)
            output_validation_status: Optional[Status] = await _validate_messages(
                messages_to_validate=[llm_output_message],
                policies=loaded_policies,
                user_id=user_id,
                request_id=request_id,
                validation_stage="output",
            )

            if output_validation_status:
                if 500 <= output_validation_status.status < 600:
                    logger.error(
                        f"Internal error during output validation: {output_validation_status.message}",
                        extra=log_extra,
                    )
                    return JSONResponse(
                        status_code=output_validation_status.status,
                        content={
                            "safety_code": output_validation_status.safety_code,
                            "message": output_validation_status.message,
                            "action": output_validation_status.action,
                        },
                    )
                else:
                    logger.warning(
                        f"Output blocked by guardrail. Policy msg: {output_validation_status.message}",
                        extra=log_extra,
                    )
                    return create_blocked_response(
                        provider="gemini",
                        block_status=output_validation_status,
                        original_request_data=request_data,
                        original_response_data=gemini_response_data,
                    )
        else:
            if gemini_response_data:
                is_google_blocked = False
                try:
                    if gemini_response_data.get("promptFeedback", {}).get(
                        "blockReason"
                    ):
                        is_google_blocked = True
                    elif any(
                        c.get("finishReason") == "SAFETY"
                        for c in gemini_response_data.get("candidates", [])
                    ):
                        is_google_blocked = True
                except Exception:
                    pass

                if not is_google_blocked:
                    logger.warning(
                        "Could not extract valid text from Gemini response for output validation. Passing through original response.",
                        extra=log_extra,
                    )
                else:
                    logger.debug(
                        "Passing through natively Google-blocked response.",
                        extra=log_extra,
                    )

            else:
                logger.error(
                    "Gemini response data is None despite successful HTTP call.",
                    extra=log_extra,
                )
                raise GuardrailedHTTPException(
                    status_code=500,
                    message="Internal server error processing Gemini response.",
                )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        if backend_response is None or gemini_response_data is None:
            logger.error(
                "Internal logic error: Backend response/data is None before successful return.",
                extra=log_extra,
            )
            raise GuardrailedHTTPException(
                status_code=500, message="Internal server error processing response."
            )

        logger.info(
            f"Gemini Proxy request finished successfully or passed through. Elapsed: {elapsed_ms}ms",
            extra=log_extra,
        )
        return Response(
            content=json.dumps(gemini_response_data),
            status_code=backend_response.status_code,
            media_type="application/json",
        )

    except (ValidationError, AuthenticationError, NotInitializedError) as client_err:
        logger.warning(
            f"Client/Setup error in Gemini proxy: {client_err}", extra=log_extra
        )
        raise client_err
    except GuardrailedHTTPException as http_exc:
        logger.warning(
            f"GuardrailedHTTPException in Gemini proxy: {http_exc.detail}", extra=log_extra
        )
        raise http_exc
    except Exception as e:
        logger.critical(
            f"Unexpected error in Gemini proxy route: {e}",
            extra=log_extra,
            exc_info=True,
        )
        raise GuardrailedHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="An unexpected internal server error occurred.",
            safety_code=SafetyCode.UNEXPECTED,
            action=Action.OVERRIDE.value,
        )
