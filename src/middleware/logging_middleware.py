import logging
import time
from typing import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()

        request_id = getattr(request.state, "request_id", "unknown-in-logging-mw")
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query = request.url.query
        user_agent = request.headers.get("user-agent", "unknown")

        logger.info(
            f"Request received: {method} {path}",
            extra={
                "request_id": request_id,
                "client_ip": client_ip,
                "method": method,
                "path": path,
                "query": query,
                "user_agent": user_agent,
            },
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = f"{process_time:.4f}"

            logger.info(
                f"Response sent: {method} {path} {response.status_code}",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time * 1000, 2),
                    "method": method,
                    "path": path,
                },
            )
            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed during processing: {method} {path}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time_ms": round(process_time * 1000, 2),
                    "method": method,
                    "path": path,
                },
                exc_info=True,
            )
            raise
