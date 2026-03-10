import asyncio
import json
import logging
import time
from typing import List

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from src.domain.validators import ContentValidator, ValidationContext
from src.exceptions import (
    NotInitializedError,
    ValidationError,
)
from src.presentation.dependencies import get_loaded_policies
from src.shared import Action, Policy, SafetyCode, Status
from src.utils import get_messages

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/safeguard",
    tags=["Safeguard"],
    response_model=Status,
    summary="Validate messages against local safety policies",
    description="Evaluates messages against configured policies (PII, Toxicity, etc.), handling long inputs via chunking.",
    response_description="Validation status including safety code, message, action, and potentially processed content (if applicable and not chunked).",
)
async def safeguard_messages(
    request: Request,
    loaded_policies: List[Policy] = Depends(get_loaded_policies),
) -> JSONResponse:
    """Validate messages against locally configured safety policies."""
    start_time = time.time()
    request_id = getattr(
        request.state,
        "request_id",
        getattr(asyncio.current_task(), "request_id", "unknown-route"),
    )
    log_extra_base = {"request_id": request_id}
    logger.info("Safeguard request received", extra=log_extra_base)

    try:
        policies = loaded_policies
        if not policies:
            logger.warning("No policies loaded/active.", extra=log_extra_base)

        try:
            data = await request.json()
            messages = get_messages(data)
        except json.JSONDecodeError as json_exc:
            logger.error(
                f"Failed to parse request JSON: {json_exc}", extra=log_extra_base
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "safety_code": SafetyCode.GENERIC_UNSAFE,
                    "message": f"Invalid JSON payload: {json_exc}",
                    "action": Action.OVERRIDE.value,
                },
            )
        if not messages:
            raise ValidationError("Input 'messages' list cannot be empty.")

        context = ValidationContext(
            policies=policies, messages=messages, user_id=data.get("user_id")
        )
        content_validator = ContentValidator(context)
        validation_status: Status = await content_validator.validate_content()

        response_content = {
            "safety_code": validation_status.safety_code,
            "message": validation_status.message,
            "action": validation_status.action,
        }
        if validation_status.processed_content is not None:
            response_content["processed_content"] = validation_status.processed_content

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        action_name = (
            Action(validation_status.action).name
            if validation_status.action is not None
            else "SAFE"
        )
        log_extra_done = {
            **log_extra_base,
            "message_count": len(messages),
            "elapsed_ms": elapsed_ms,
            "safety_code": validation_status.safety_code,
            "action": action_name,
        }
        logger.info("Safeguard request completed", extra=log_extra_done)

        return JSONResponse(
            status_code=validation_status.status,
            content=response_content,
        )

    except ClientDisconnect:
        logger.warning("Client disconnected.", extra=log_extra_base)
        return JSONResponse(
            status_code=499,
            content={
                "safety_code": SafetyCode.GENERIC_UNSAFE,
                "message": "Client disconnected",
                "action": Action.OVERRIDE.value,
            },
        )
    except ValidationError as ve:
        logger.warning(f"Validation Error: {ve}", extra=log_extra_base, exc_info=False)
        raise
    except NotInitializedError as nie:
        logger.error(f"Component not initialized: {nie}", extra=log_extra_base)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "safety_code": SafetyCode.UNEXPECTED,
                "message": f"Service component not ready: {nie}.",
                "action": Action.RETRY.value,
            },
        )
    except Exception as e:
        logger.critical(
            f"Unexpected error in safeguard route: {e}",
            extra=log_extra_base,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "safety_code": SafetyCode.UNEXPECTED,
                "message": "An unexpected internal server error occurred.",
                "action": Action.OVERRIDE.value,
            },
        )
