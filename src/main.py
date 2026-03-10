import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core import app_state, load_config, setup_logger

try:
    loaded_config = load_config()
    app_state.config = loaded_config
except SystemExit as e:
    logging.critical(f"Configuration failed: {e}. Exiting.")
    exit(1)
except Exception as e:
    logging.critical(f"Unexpected error loading configuration: {e}", exc_info=True)
    exit(1)

from src.core.shutdown import cleanup_system
from src.core.startup import startup_event
from src.exceptions.handlers import setup_exception_handlers
from src.middleware import register_middleware
from src.middleware.metrics_middleware import setup_metrics
from src.presentation.routes.claude_proxy import router as claude_proxy_router
from src.presentation.routes.gemini_proxy import router as gemini_proxy_router
from src.presentation.routes.health import router as health_router
from src.presentation.routes.openai_proxy import router as openai_proxy_router
from src.presentation.routes.safeguard import router as safeguard_router

logger = setup_logger()

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifecycle manager for the application.
    - Sets up resources on startup
    - Cleans up resources on shutdown
    """
    from src.presentation.proxy_utils import http_client as shared_proxy_http_client

    try:
        logger.info("Starting application initialization...")
        await startup_event()
        logger.info("Application successfully initialized")
        yield
    finally:
        logger.info("Starting application shutdown...")
        try:
            await shared_proxy_http_client.aclose()
            logger.info("Shared HTTPX client closed.")
        except Exception as e:
            logger.error(f"Error closing shared httpx client: {e}", exc_info=True)

        await cleanup_system()
        logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Guardrailed Data Plane API - Guardrails & Proxy",
        description="Content validation service and LLM proxy for safeguarding interactions.",
        version="1.2.0-OS",
        lifespan=lifespan,
    )

    register_middleware(app)

    setup_metrics(app)

    setup_exception_handlers(app)

    if not app_state.config:
        raise RuntimeError("App state is not initialized.")
    cors_origins = list(app_state.config.allowed_origins)
    if not cors_origins or cors_origins == ["*"]:
        logger.warning("CORS configured to allow all origins ('*').")
        cors_origins = ["*"]
    else:
        logger.info(f"CORS configured for specific origins: {cors_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*", "Authorization", "X-Goog-Api-Key", "X-Request-ID"],
    )

    app.include_router(health_router, tags=["Health"])
    app.include_router(safeguard_router, tags=["Safeguard"])

    app.include_router(openai_proxy_router, prefix="/v1", tags=["OpenAI Proxy"])
    app.include_router(
        gemini_proxy_router,
        prefix=f"/{app_state.config.gemini_api_version}",
        tags=["Gemini Proxy"],
    )
    app.include_router(
        claude_proxy_router, prefix="/anthropic/v1", tags=["Claude Proxy"]
    )
    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn

    log_level = (
        "info"
        if app_state.config.environment == app_state.config.environment.PRODUCTION
        else "debug"
    )
    print(
        f"Starting Uvicorn server on {app_state.config.host}:{app_state.config.port} with log level '{log_level}'..."
    )

    uvicorn.run(
        "src.main:app",
        host=app_state.config.host,
        port=app_state.config.port,
        reload=False,
        log_level=log_level,
    )
