import json
import logging
import traceback
from typing import Any, Callable, Coroutine, Union

import httpx
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)

from src.exceptions import (
    GuardrailedBaseError,
    GuardrailedHTTPException,
    ValidationError,
)
from src.shared import Action, SafetyCode

logger = logging.getLogger(__name__)

ExceptionHandler = Callable[
    [Request, Exception], Union[Response, Coroutine[Any, Any, Response]]
]


async def guardrailed_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for all GuardrailedBaseError exceptions."""
    guardrailed_exc = exc if isinstance(exc, GuardrailedBaseError) else None
    if not guardrailed_exc:
        return await unhandled_exception_handler(request, exc)

    request_id = getattr(request.state, "request_id", "unknown")

    log_level = (
        logging.ERROR
        if guardrailed_exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR
        else logging.WARNING
    )
    logger.log(
        log_level,
        f"Application exception: {guardrailed_exc.__class__.__name__} - {guardrailed_exc.message}",
        extra={
            "request_id": request_id,
            "exception_type": guardrailed_exc.__class__.__name__,
            "status_code": guardrailed_exc.status_code,
            "path": request.url.path if request else "unknown",
            "method": request.method if request else "unknown",
        },
        exc_info=(guardrailed_exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR),
    )

    if isinstance(guardrailed_exc, ValidationError):
        response_safety_code = SafetyCode.GENERIC_UNSAFE
    elif guardrailed_exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        response_safety_code = SafetyCode.UNEXPECTED
    else:
        response_safety_code = SafetyCode.GENERIC_UNSAFE

    if guardrailed_exc.user_facing:
        response_message = guardrailed_exc.message
    else:
        response_message = "An unexpected error occurred. Please try again later."
        if guardrailed_exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            response_safety_code = SafetyCode.UNEXPECTED

    final_status_code = guardrailed_exc.status_code
    if final_status_code == status.HTTP_200_OK:
        final_status_code = (
            status.HTTP_400_BAD_REQUEST
            if response_safety_code == SafetyCode.GENERIC_UNSAFE
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    return JSONResponse(
        status_code=final_status_code,
        content={
            "safety_code": response_safety_code,
            "message": response_message,
            "action": Action.OVERRIDE.value,
        },
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for GuardrailedHTTPException (usually raised intentionally in routes)."""
    http_exc = exc if isinstance(exc, GuardrailedHTTPException) else None
    if not http_exc:
        return await unhandled_exception_handler(request, exc)

    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"HTTP exception returned: {http_exc.status_code} - {http_exc.message}",
        extra={
            "request_id": request_id,
            "exception_type": http_exc.__class__.__name__,
            "status_code": http_exc.status_code,
            "safety_code": http_exc.safety_code,
            "action": http_exc.action,
            "path": request.url.path if request else "unknown",
            "method": request.method if request else "unknown",
        },
    )

    return JSONResponse(
        status_code=http_exc.status_code,
        content={
            "safety_code": http_exc.safety_code,
            "message": http_exc.message,
            "action": http_exc.action,
        },
    )


async def json_decode_error_handler(
    request: Request, exc: json.JSONDecodeError
) -> JSONResponse:
    """Handles errors during JSON request body parsing."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        f"Invalid JSON received: {exc}",
        extra={
            "request_id": request_id,
            "path": request.url.path if request else "unknown",
            "method": request.method if request else "unknown",
        },
        exc_info=False,
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "safety_code": SafetyCode.GENERIC_UNSAFE,
            "message": f"Invalid JSON payload: {exc}",
            "action": Action.OVERRIDE,
        },
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handles Pydantic validation errors from FastAPI."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        f"Request validation failed: {exc.errors()}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "detail": exc.errors(),
        },
        exc_info=False,
    )
    try:
        first_error = exc.errors()[0]
        field = ".".join(map(str, first_error.get("loc", ["body"])))
        msg = first_error.get("msg", "Validation failed")
        user_message = f"Invalid input for field '{field}': {msg}"
    except Exception:
        user_message = "Invalid request input."

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "safety_code": SafetyCode.GENERIC_UNSAFE,
            "message": user_message,
            "action": Action.OVERRIDE.value,
        },
    )


async def starlette_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handles generic Starlette/FastAPI HTTPExceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    log_level = (
        logging.WARNING
        if exc.status_code < status.HTTP_500_INTERNAL_SERVER_ERROR
        else logging.ERROR
    )
    logger.log(
        log_level,
        f"Caught Starlette HTTPException: {exc.status_code} - {exc.detail}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
        exc_info=(exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR),
    )
    safety_code = (
        SafetyCode.UNEXPECTED
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR
        else SafetyCode.GENERIC_UNSAFE
    )
    message = exc.detail if isinstance(exc.detail, str) else "An HTTP error occurred."
    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        message = "An unexpected error occurred. Please try again later."
        safety_code = SafetyCode.UNEXPECTED

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "safety_code": safety_code,
            "message": message,
            "action": Action.OVERRIDE.value,
        },
        headers=getattr(exc, "headers", None),
    )


async def httpx_http_status_error_handler(
    request: Request, exc: httpx.HTTPStatusError
) -> Response:
    """Handles errors received from the backend service (httpx) by forwarding the original response."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        f"Forwarding backend HTTP error: {exc.response.status_code}",
        extra={
            "request_id": request_id,
            "backend_status": exc.response.status_code,
            "backend_url": str(exc.request.url),
            "backend_content": exc.response.text[:200],
            "path": request.url.path,
            "method": request.method,
        },
        exc_info=False,
    )
    response_headers = dict(exc.response.headers)

    return Response(
        content=exc.response.content,
        status_code=exc.response.status_code,
        headers=response_headers,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for all other unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    path = request.url.path if request else "unknown"
    method = request.method if request else "unknown"

    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.critical(
        f"Unhandled exception: {type(exc).__name__} - {exc!s}",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "traceback": "".join(tb),
            "path": path,
            "method": method,
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "safety_code": SafetyCode.UNEXPECTED,
            "message": "An unexpected internal server error occurred.",
            "action": Action.OVERRIDE,
        },
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure exception handlers for the FastAPI application."""
    app.add_exception_handler(GuardrailedHTTPException, http_exception_handler)
    app.add_exception_handler(json.JSONDecodeError, json_decode_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(
        RequestValidationError, request_validation_exception_handler  # type: ignore[arg-type]
    )
    app.add_exception_handler(httpx.HTTPStatusError, httpx_http_status_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(GuardrailedBaseError, guardrailed_exception_handler)

    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)  # type: ignore[arg-type]

    app.add_exception_handler(Exception, unhandled_exception_handler)

    logger.info(
        "Registered custom exception handlers, including FastAPI/Starlette/HTTPX overrides."
    )
