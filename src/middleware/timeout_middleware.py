import asyncio
import logging
from typing import Awaitable, Callable, Dict

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from src.core import app_state
from src.exceptions import InitializationError
from src.shared import Action, SafetyCode

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces a timeout for request processing.

    This middleware:
    1. Sets a maximum allowed time for request processing
    2. Returns a timeout error if processing exceeds this time
    3. Can be configured for different timeout values per endpoint
    """

    def __init__(self, app: FastAPI):
        """
        Initialize the timeout middleware.

        Args:
            app: The FastAPI application
            timeout_seconds: Default timeout in seconds
            options: Additional configuration options
        """
        super().__init__(app)
        if not app_state.config:
            raise InitializationError("app_state", "Config is missing.")
        self.default_timeout = app_state.config.middleware.timeout.default_timeout
        self.path_timeouts: Dict[str, int] = (
            app_state.config.middleware.timeout.path_timeouts
        )

    def get_timeout_for_path(self, path: str) -> int:
        """
        Get the timeout for a specific path.

        Args:
            path: The request path

        Returns:
            The timeout in seconds for the given path
        """
        if path in self.path_timeouts:
            return self.path_timeouts[path]

        for prefix, timeout in self.path_timeouts.items():
            if prefix.endswith("*") and path.startswith(prefix[:-1]):
                return timeout

        return self.default_timeout

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process the request with a timeout limit."""
        path = request.url.path
        timeout_seconds = self.get_timeout_for_path(path)

        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timed out: {request.method} {path}",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "timeout_seconds": timeout_seconds,
                },
            )

            return JSONResponse(
                status_code=504,
                content={
                    "safety_code": SafetyCode.TIMEOUT,
                    "message": f"Request processing timed out after {timeout_seconds} seconds",
                    "action": Action.RETRY,
                },
            )
