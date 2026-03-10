import importlib.util
import logging
import os

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from src.core.state import app_state
from src.domain.transformers import ClassificationModel, NERModel
from src.exceptions import InitializationError

from .shutdown import cleanup_system

logger = logging.getLogger(__name__)


async def init_presidio_engines() -> None:
    """Initializes Presidio Analyzer and Anonymizer engines using Transformers backend."""
    if app_state.presidio_analyzer_engine and app_state.presidio_anonymizer_engine:
        logger.info("Engines already initialized.")
        return

    logger.info("Initializing engines with Transformers backend...")
    try:
        spacy_model_name = os.getenv("SPACY_MODEL_FOR_PRESIDIO", "en_core_web_sm")
        transformers_model_name = os.getenv(
            "PRESIDIO_TRANSFORMERS_MODEL", "dslim/bert-base-NER"
        )

        try:
            importlib.util.find_spec(spacy_model_name)
            logger.info(f"Found model '{spacy_model_name}' for tokenization.")
        except ModuleNotFoundError:
            logger.error(
                f"Required model '{spacy_model_name}' for not found. "
                f"Please install it via: python -m spacy download {spacy_model_name}"
            )
            raise InitializationError(
                "Presidio", f"Missing spaCy model '{spacy_model_name}'"
            )

        presidio_nlp_configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": {
                        "spacy": spacy_model_name,
                        "transformers": transformers_model_name,
                    },
                }
            ],
            "ner_model_configuration": {
                "labels_to_ignore": ["O", "MISC"],
                "aggregation_strategy": "simple",
            },
        }

        provider = NlpEngineProvider(nlp_configuration=presidio_nlp_configuration)
        nlp_engine = provider.create_engine()
        logger.info(
            f"Presidio NLP engine (Transformers backend with {transformers_model_name} NER) created."
        )

        app_state.presidio_analyzer_engine = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=["en"]
        )
        logger.info("Presidio AnalyzerEngine initialized.")

        app_state.presidio_anonymizer_engine = AnonymizerEngine()
        logger.info("Presidio AnonymizerEngine initialized.")

    except InitializationError:
        raise
    except ImportError as e:
        logger.error(
            f"Import error during Presidio init. Check dependencies: {e}", exc_info=True
        )
        raise InitializationError("Presidio", f"Missing dependency: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize Presidio engines: {e}", exc_info=True)
        raise InitializationError("Presidio", f"Failed to initialize: {e}")


async def init_transformer_models() -> None:
    """Initializes non-Presidio Transformer models (Toxicity, standalone NER)."""
    if app_state.profanity_model or app_state.ner_model:
        logger.info("Standard Transformer models already initialized or skipped.")
        return

    logger.info("Initializing standard Transformer models (Toxicity/NER)...")
    try:
        if not app_state.config:
            raise InitializationError("app_state", "Missing config.")
        if app_state.config.toxicity_model_url:
            logger.info(
                f"Initializing ClassificationModel from: {app_state.config.toxicity_model_url}"
            )
            app_state.profanity_model = ClassificationModel(
                app_state.config.toxicity_model_url
            )
            await app_state.profanity_model.initialize()
            logger.info("ClassificationModel (Toxicity) initialized.")
        else:
            logger.warning(
                "TOXICITY_MODEL_URL not configured. Skipping initialization."
            )

        if app_state.config.ner_model_url:
            logger.info(
                f"Initializing standalone NERModel from: {app_state.config.ner_model_url}"
            )
            app_state.ner_model = NERModel(app_state.config.ner_model_url)
            await app_state.ner_model.initialize()
            logger.info("Standalone NERModel initialized.")
        else:
            logger.warning(
                "NER_MODEL_URL not configured. Skipping standalone NER initialization."
            )

    except Exception as e:
        logger.error(
            f"Failed to initialize standard Transformer models: {e}", exc_info=True
        )
        raise InitializationError("Transformer Models", f"Initialization failed: {e}")


async def init_system() -> None:
    """Initializes all system components including models and engines."""
    logger.info("Initializing system components")
    try:
        await init_presidio_engines()
        await init_transformer_models()

    except InitializationError as e:
        logger.error(f"System initialization failed: {e.message}", exc_info=False)
        await cleanup_system()
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during system initialization: {e}", exc_info=True
        )
        await cleanup_system()
        raise InitializationError("System", f"Unexpected error: {e}")


async def startup_event() -> None:
    logger.info("Starting application startup process")
    try:
        await init_system()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.critical(f"Fatal error during application startup: {e}", exc_info=True)
        raise
