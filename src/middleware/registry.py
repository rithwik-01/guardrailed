import logging

from fastapi import FastAPI

from src.core import app_state
from src.exceptions import InitializationError

from .logging_middleware import LoggingMiddleware
from .request_id_middleware import RequestIDMiddleware
from .security_middleware import SecurityMiddleware
from .timeout_middleware import TimeoutMiddleware

logger = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> FastAPI:
    if not app_state.config:
        raise InitializationError("app_state", "Config is missing.")
    middleware_config = app_state.config.middleware
    logger.info("Registering API middleware components (OS Core)")

    if middleware_config.security.enabled:
        app.add_middleware(SecurityMiddleware)
        logger.info("Registered security middleware")

    app.add_middleware(RequestIDMiddleware)
    logger.info("Registered request ID middleware")

    app.add_middleware(LoggingMiddleware)
    logger.info("Registered logging middleware")

    app.add_middleware(TimeoutMiddleware)
    logger.info("Registered timeout middleware")
    return app
