import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.shutdown import cleanup_system

mock_app_state_patch_target = "src.core.shutdown.app_state"


@pytest.fixture
def mock_profanity_model():
    """Fixture for a mock profanity model with an async close."""
    model = AsyncMock(spec=True)
    model.close = AsyncMock(return_value=None)
    model.name = "MockProfanityModel"
    return model


@pytest.fixture
def mock_ner_model():
    """Fixture for a mock NER model with an async close."""
    model = AsyncMock(spec=True)
    model.close = AsyncMock(return_value=None)
    model.name = "MockNERModel"
    return model


@pytest.fixture
def mock_presidio_analyzer():
    """Fixture for a mock Presidio AnalyzerEngine."""
    engine = MagicMock(spec=True)
    engine.name = "MockPresidioAnalyzer"
    return engine


@pytest.fixture
def mock_presidio_anonymizer():
    """Fixture for a mock Presidio AnonymizerEngine."""
    engine = MagicMock(spec=True)
    engine.name = "MockPresidioAnonymizer"
    return engine


@pytest.fixture
def mock_torch_module():
    """Creates a mock torch module structure."""
    mock_torch = MagicMock(name="mock_torch")
    mock_torch.cuda = MagicMock(name="mock_torch.cuda")
    mock_torch.cuda.is_available = MagicMock(return_value=True)
    mock_torch.cuda.empty_cache = MagicMock()
    return mock_torch


@pytest.fixture
def patch_torch_in_sys_modules(mock_torch_module):
    """Fixture to patch torch IN sys.modules for tests simulating torch presence."""
    with patch.dict(sys.modules, {"torch": mock_torch_module}):
        yield mock_torch_module


@pytest.mark.asyncio
async def test_cleanup_system_all_present_cuda_available(
    mock_profanity_model,
    mock_ner_model,
    mock_presidio_analyzer,
    mock_presidio_anonymizer,
    patch_torch_in_sys_modules,
    caplog,
):
    """Test successful cleanup with all components present and CUDA available."""
    mock_state = MagicMock(name="MockAppState_AllPresent")
    mock_state.profanity_model = mock_profanity_model
    mock_state.ner_model = mock_ner_model
    mock_state.presidio_analyzer_engine = mock_presidio_analyzer
    mock_state.presidio_anonymizer_engine = mock_presidio_anonymizer

    mock_torch = patch_torch_in_sys_modules
    mock_torch.cuda.is_available.return_value = True

    caplog.set_level(logging.DEBUG)
    with patch(mock_app_state_patch_target, mock_state):
        await cleanup_system()

    mock_profanity_model.close.assert_awaited_once()
    assert mock_state.profanity_model is None
    assert "Cleaning up profanity model." in caplog.text
    assert "Profanity model cleaned up." in caplog.text

    mock_ner_model.close.assert_awaited_once()
    assert mock_state.ner_model is None

    assert mock_state.presidio_analyzer_engine is None

    assert mock_state.presidio_anonymizer_engine is None

    mock_torch.cuda.is_available.assert_called_once()
    mock_torch.cuda.empty_cache.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_system_some_missing(
    mock_ner_model, mock_presidio_analyzer, patch_torch_in_sys_modules, caplog
):
    """Test cleanup when some components are None initially."""
    mock_state = MagicMock(name="MockAppState_SomeMissing")
    mock_state.profanity_model = None
    mock_state.ner_model = mock_ner_model
    mock_state.presidio_analyzer_engine = mock_presidio_analyzer
    mock_state.presidio_anonymizer_engine = None

    mock_torch = patch_torch_in_sys_modules
    mock_torch.cuda.is_available.return_value = False

    caplog.set_level(logging.DEBUG)

    with patch(mock_app_state_patch_target, mock_state):
        await cleanup_system()

    mock_ner_model.close.assert_awaited_once()
    assert mock_state.ner_model is None

    assert mock_state.presidio_analyzer_engine is None

    mock_torch.cuda.is_available.assert_called_once()
    mock_torch.cuda.empty_cache.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_system_pytorch_not_installed(mock_profanity_model, caplog):
    """Test cleanup skips CUDA clear if torch import fails."""
    mock_state_instance = MagicMock(name="MockAppState_NoTorch")
    mock_state_instance.profanity_model = mock_profanity_model
    mock_state_instance.ner_model = None
    mock_state_instance.presidio_analyzer_engine = None
    mock_state_instance.presidio_anonymizer_engine = None

    caplog.set_level(logging.DEBUG)

    original_import = __import__

    def import_mock(name, *args, **kwargs):
        if name == "torch":
            raise ImportError(f"Mock ImportError: No module named {name}")
        return original_import(name, *args, **kwargs)

    with patch.dict(sys.modules):
        if "torch" in sys.modules:
            del sys.modules["torch"]

        with patch("builtins.__import__", side_effect=import_mock):
            with patch(mock_app_state_patch_target, mock_state_instance):
                await cleanup_system()
    mock_profanity_model.close.assert_awaited_once()
    assert mock_state_instance.profanity_model is None

    assert "PyTorch not installed, skipping CUDA cache clear." in caplog.text
    assert "Clearing CUDA cache..." not in caplog.text
    assert "CUDA cache cleared." not in caplog.text
    assert "Error clearing CUDA cache:" not in caplog.text

    assert "Starting system cleanup" in caplog.text
    assert "System cleanup finished." in caplog.text


@pytest.mark.asyncio
async def test_cleanup_system_cuda_error(
    mock_profanity_model, patch_torch_in_sys_modules, caplog
):
    """Test cleanup handles errors during CUDA cache clearing."""
    mock_state = MagicMock(name="MockAppState_CudaError")
    mock_state.profanity_model = mock_profanity_model
    mock_state.ner_model = None
    mock_state.presidio_analyzer_engine = None
    mock_state.presidio_anonymizer_engine = None
    mock_torch = patch_torch_in_sys_modules
    mock_torch.cuda.is_available.return_value = True
    error_msg = "CUDA driver error"
    mock_torch.cuda.empty_cache.side_effect = Exception(error_msg)

    caplog.set_level(logging.DEBUG)

    with patch(mock_app_state_patch_target, mock_state):
        await cleanup_system()

    mock_profanity_model.close.assert_awaited_once()
    assert mock_state.profanity_model is None

    mock_torch.cuda.is_available.assert_called_once()
    mock_torch.cuda.empty_cache.assert_called_once()
    assert "Clearing CUDA cache..." in caplog.text
    assert f"Error clearing CUDA cache: {error_msg}" in caplog.text
    assert "CUDA cache cleared." not in caplog.text

    assert "Starting system cleanup" in caplog.text
    assert "System cleanup finished." in caplog.text
