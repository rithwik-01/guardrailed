import logging
import uuid
from typing import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.core.logging import request_id_var

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id

        token = request_id_var.set(request_id)
        logger.debug(
            f"Request {request_id} processing started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )

        response: Response
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
        except Exception as e:
            logger.error(
                f"Unhandled exception during request {request_id}",
                exc_info=True,
                extra={"request_id": request_id},
            )
            raise e
        finally:
            request_id_var.reset(token)

        logger.debug(
            f"Request {request_id} processing finished",
            extra={"request_id": request_id, "status_code": response.status_code},
        )

        return response
