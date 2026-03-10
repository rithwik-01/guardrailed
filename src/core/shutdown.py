import asyncio
import logging
from typing import Any, Coroutine

from .state import app_state

logger = logging.getLogger(__name__)


class ShutdownManager:
    """Simplified ShutdownManager, primarily for potential future resource management."""

    def __init__(self) -> None:
        pass

    async def cleanup(self) -> None:
        """Gracefully shut down all managed resources."""
        logger.info("Starting graceful shutdown sequence")
        logger.debug(
            "No specific resources managed by ShutdownManager needed cleanup (OS Core)."
        )
        logger.info("Completed ShutdownManager cleanup sequence (OS Core)")

    async def _safe_cleanup(
        self,
        cleanup_coro: Coroutine[Any, Any, None],
        component_name: str,
        timeout: float = 5.0,
    ) -> None:
        """Execute cleanup with timeout protection."""
        try:
            await asyncio.wait_for(cleanup_coro, timeout=timeout)
            logger.info(f"Successfully closed {component_name}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout while closing {component_name}")
        except Exception as e:
            logger.error(f"Error closing {component_name}: {e}", exc_info=True)


async def cleanup_system() -> None:
    """Cleanup all system components (models, engines)."""
    logger.info("Starting system cleanup")

    if app_state.profanity_model:
        logger.debug("Cleaning up profanity model.")
        try:
            await app_state.profanity_model.close()
            app_state.profanity_model = None
            logger.info("Profanity model cleaned up.")
        except Exception as e:
            logger.error(f"Error cleaning up profanity model: {e}", exc_info=True)

    if app_state.ner_model:
        logger.debug("Cleaning up NER model.")
        try:
            await app_state.ner_model.close()
            app_state.ner_model = None
            logger.info("NER model cleaned up.")
        except Exception as e:
            logger.error(f"Error cleaning up NER model: {e}", exc_info=True)

    if app_state.presidio_analyzer_engine:
        logger.debug("Cleaning up AnalyzerEngine reference.")
        try:
            del app_state.presidio_analyzer_engine
            app_state.presidio_analyzer_engine = None
            logger.info("AnalyzerEngine reference cleared.")
        except Exception as e:
            logger.error(
                f"Error cleaning up AnalyzerEngine reference: {e}",
                exc_info=True,
            )

    if app_state.presidio_anonymizer_engine:
        logger.debug("Cleaning up AnonymizerEngine reference.")
        try:
            del app_state.presidio_anonymizer_engine
            app_state.presidio_anonymizer_engine = None
            logger.info("AnonymizerEngine reference cleared.")
        except Exception as e:
            logger.error(
                f"Error cleaning up AnonymizerEngine reference: {e}",
                exc_info=True,
            )

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
    except ImportError:
        logger.debug("PyTorch not installed, skipping CUDA cache clear.")
    except Exception as e:
        logger.error(f"Error clearing CUDA cache: {e}", exc_info=True)

    logger.info("System cleanup finished.")
