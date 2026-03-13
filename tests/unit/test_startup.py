import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.core.startup import (
    init_system,
    init_transformer_models,
    startup_event,
)
from src.domain.transformers import ClassificationModel, NERModel
from src.exceptions import InitializationError

APP_STATE_TARGET = "src.core.startup.app_state"
GETENV_TARGET = "src.core.startup.os.getenv"
FIND_SPEC_TARGET = "src.core.startup.importlib.util.find_spec"
PROVIDER_TARGET = "src.core.startup.NlpEngineProvider"
ANALYZER_TARGET = "src.core.startup.AnalyzerEngine"
ANONYMIZER_TARGET = "src.core.startup.AnonymizerEngine"
CLASS_MODEL_TARGET = "src.core.startup.ClassificationModel"
NER_MODEL_TARGET = "src.core.startup.NERModel"
CLEANUP_SYSTEM_TARGET = "src.core.startup.cleanup_system"


@pytest.fixture
def mock_app_state():
    """Provides a mock app_state with a mock config."""
    state = MagicMock(name="mock_app_state")
    state.config = MagicMock(name="mock_config")
    state.config.toxicity_model_url = "mock/toxicity/model"
    state.config.ner_model_url = "mock/ner/model"
    state.presidio_analyzer_engine = None
    state.presidio_anonymizer_engine = None
    state.profanity_model = None
    state.ner_model = None
    return state


@pytest.fixture
def mock_classification_model_instance():
    """Provides a mock ClassificationModel INSTANCE."""
    mock_instance = AsyncMock(spec=ClassificationModel)
    mock_instance.initialize = AsyncMock()
    return mock_instance


@pytest.fixture
def mock_ner_model_instance():
    """Provides a mock NERModel INSTANCE."""
    mock_instance = AsyncMock(spec=NERModel)
    mock_instance.initialize = AsyncMock()
    return mock_instance


@pytest.fixture
def mock_classification_model_cls(mock_classification_model_instance):
    """Provides a mock ClassificationModel CLASS."""
    mock_cls = MagicMock(spec=ClassificationModel)
    mock_cls.return_value = mock_classification_model_instance
    return mock_cls


@pytest.fixture
def mock_ner_model_cls(mock_ner_model_instance):
    """Provides a mock NERModel CLASS."""
    mock_cls = MagicMock(spec=NERModel)
    mock_cls.return_value = mock_ner_model_instance
    return mock_cls


@pytest.fixture
def mock_provider_cls():
    """Provides a mock NlpEngineProvider class."""
    mock_cls = MagicMock(spec=True)
    mock_instance = MagicMock()
    mock_engine = MagicMock(name="mock_nlp_engine")
    mock_instance.create_engine.return_value = mock_engine
    mock_cls.return_value = mock_instance
    return mock_cls


@pytest.fixture
def mock_analyzer_cls():
    """Provides a mock AnalyzerEngine class."""
    mock_cls = MagicMock(spec=AnalyzerEngine)
    mock_instance = MagicMock(spec=AnalyzerEngine)
    mock_cls.return_value = mock_instance
    return mock_cls


@pytest.fixture
def mock_anonymizer_cls():
    """Provides a mock AnonymizerEngine class."""
    mock_cls = MagicMock(spec=AnonymizerEngine)
    mock_instance = MagicMock(spec=AnonymizerEngine)
    mock_cls.return_value = mock_instance
    return mock_cls


@pytest.mark.asyncio
@patch(CLASS_MODEL_TARGET)
@patch(NER_MODEL_TARGET)
@patch(APP_STATE_TARGET)
async def test_init_transformers_success_both(
    mock_app_state_obj,
    mock_ner_cls_patch,
    mock_class_cls_patch,
    mock_classification_model_instance,
    mock_ner_model_instance,
    caplog,
):
    """Test successful initialization of both standard transformer models."""
    mock_class_cls_patch.return_value = mock_classification_model_instance
    mock_ner_cls_patch.return_value = mock_ner_model_instance

    mock_app_state_obj.config.toxicity_model_url = "path/to/tox"
    mock_app_state_obj.config.ner_model_url = "path/to/ner"
    mock_app_state_obj.profanity_model = None
    mock_app_state_obj.ner_model = None

    caplog.set_level(logging.INFO)

    await init_transformer_models()

    mock_class_cls_patch.assert_called_once_with("path/to/tox")
    mock_classification_model_instance.initialize.assert_awaited_once()
    assert mock_app_state_obj.profanity_model is mock_classification_model_instance

    mock_ner_cls_patch.assert_called_once_with("path/to/ner")
    mock_ner_model_instance.initialize.assert_awaited_once()
    assert mock_app_state_obj.ner_model is mock_ner_model_instance

    assert "Initializing ClassificationModel from: path/to/tox" in caplog.text
    assert "ClassificationModel (Toxicity) initialized." in caplog.text
    assert "Initializing standalone NERModel from: path/to/ner" in caplog.text
    assert "Standalone NERModel initialized." in caplog.text


@pytest.mark.asyncio
@patch(CLASS_MODEL_TARGET)
@patch(NER_MODEL_TARGET)
@patch(APP_STATE_TARGET)
async def test_init_transformers_success_both_missing(
    mock_app_state_obj,
    mock_ner_cls_patch,
    mock_class_cls_patch,
    mock_classification_model_instance,
    mock_ner_model_instance,
    caplog,
):
    """Test initialization when both model URLs are missing."""
    mock_app_state_obj.config.toxicity_model_url = None
    mock_app_state_obj.config.ner_model_url = None
    mock_app_state_obj.profanity_model = None
    mock_app_state_obj.ner_model = None
    caplog.set_level(logging.INFO)

    await init_transformer_models()

    mock_class_cls_patch.assert_not_called()
    mock_ner_cls_patch.assert_not_called()
    mock_classification_model_instance.initialize.assert_not_awaited()
    mock_ner_model_instance.initialize.assert_not_awaited()
    assert mock_app_state_obj.profanity_model is None
    assert mock_app_state_obj.ner_model is None

    assert "TOXICITY_MODEL_URL not configured. Skipping initialization." in caplog.text
    assert (
        "NER_MODEL_URL not configured. Skipping standalone NER initialization."
        in caplog.text
    )


@pytest.mark.asyncio
@patch(APP_STATE_TARGET)
async def test_init_transformers_already_initialized(mock_app_state_obj, caplog):
    """Test skipping init if standard models already exist."""
    mock_app_state_obj.profanity_model = MagicMock()
    mock_app_state_obj.ner_model = MagicMock()
    caplog.set_level(logging.INFO)

    await init_transformer_models()

    assert "Standard Transformer models already initialized or skipped." in caplog.text
    assert "Initializing standard Transformer models" not in caplog.text


@pytest.mark.asyncio
@patch("src.core.startup.init_presidio_engines", new_callable=AsyncMock)
@patch("src.core.startup.init_transformer_models", new_callable=AsyncMock)
async def test_init_system_success(mock_init_transformers, mock_init_presidio):
    """Test successful system initialization."""
    await init_system()
    mock_init_presidio.assert_awaited_once()
    mock_init_transformers.assert_awaited_once()


@pytest.mark.asyncio
@patch(
    "src.core.startup.init_presidio_engines",
    new_callable=AsyncMock,
    side_effect=InitializationError("Presidio", "Presidio failed"),
)
@patch("src.core.startup.init_transformer_models", new_callable=AsyncMock)
@patch(CLEANUP_SYSTEM_TARGET, new_callable=AsyncMock)
async def test_init_system_presidio_fails(
    mock_cleanup, mock_init_transformers, mock_init_presidio, caplog
):
    """Test init_system handles InitializationError from Presidio and cleans up."""
    caplog.set_level(logging.ERROR)
    presidio_error_message = "Presidio failed"
    expected_log_message = f"System initialization failed: Failed to initialize Presidio: {presidio_error_message}"

    with pytest.raises(InitializationError, match=presidio_error_message):
        await init_system()

    mock_init_presidio.assert_awaited_once()
    mock_init_transformers.assert_not_awaited()
    mock_cleanup.assert_awaited_once()
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch("src.core.startup.init_presidio_engines", new_callable=AsyncMock)
@patch(
    "src.core.startup.init_transformer_models",
    new_callable=AsyncMock,
    side_effect=InitializationError("Transformers", "HF failed"),
)
@patch(CLEANUP_SYSTEM_TARGET, new_callable=AsyncMock)
async def test_init_system_transformers_fail(
    mock_cleanup, mock_init_transformers, mock_init_presidio, caplog
):
    """Test init_system handles InitializationError from Transformers and cleans up."""
    caplog.set_level(logging.ERROR)
    transformers_error_message = "HF failed"
    expected_log_message = f"System initialization failed: Failed to initialize Transformers: {transformers_error_message}"

    with pytest.raises(InitializationError, match=transformers_error_message):
        await init_system()

    mock_init_presidio.assert_awaited_once()
    mock_init_transformers.assert_awaited_once()
    mock_cleanup.assert_awaited_once()
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch(
    "src.core.startup.init_presidio_engines",
    new_callable=AsyncMock,
    side_effect=ValueError("Unexpected crash"),
)
@patch("src.core.startup.init_transformer_models", new_callable=AsyncMock)
@patch(CLEANUP_SYSTEM_TARGET, new_callable=AsyncMock)
async def test_init_system_unexpected_error(
    mock_cleanup, mock_init_transformers, mock_init_presidio, caplog
):
    """Test init_system handles unexpected errors, cleans up, and raises InitializationError."""
    caplog.set_level(logging.ERROR)
    unexpected_error_message = "Unexpected crash"
    expected_log_message = (
        f"Unexpected error during system initialization: {unexpected_error_message}"
    )

    with pytest.raises(
        InitializationError, match=f"Unexpected error: {unexpected_error_message}"
    ):
        await init_system()

    mock_init_presidio.assert_awaited_once()
    mock_init_transformers.assert_not_awaited()
    mock_cleanup.assert_awaited_once()
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch("src.core.startup.init_system", new_callable=AsyncMock)
async def test_startup_event_success(mock_init_system, caplog):
    """Test successful startup event."""
    caplog.set_level(logging.INFO)
    await startup_event()
    mock_init_system.assert_awaited_once()
    assert "Starting application startup process" in caplog.text
    assert "Application startup completed successfully" in caplog.text


@pytest.mark.asyncio
@patch(
    "src.core.startup.init_system",
    new_callable=AsyncMock,
    side_effect=Exception("Fatal error"),
)
async def test_startup_event_failure(mock_init_system, caplog):
    """Test startup event failure raises exception."""
    caplog.set_level(logging.CRITICAL)
    with pytest.raises(Exception, match="Fatal error"):
        await startup_event()
    mock_init_system.assert_awaited_once()
    assert "Fatal error during application startup: Fatal error" in caplog.text
