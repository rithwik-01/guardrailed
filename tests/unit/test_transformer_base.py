from unittest.mock import MagicMock, patch

import pytest
import torch

from src.domain.transformers.base import BaseTransformerModel


class ConcreteModel(BaseTransformerModel):
    """Dummy subclass to instantiate BaseTransformerModel for testing."""

    async def initialize(self) -> None:
        pass


@pytest.fixture
def cpu_device():
    """Fixture for CPU device object."""
    return torch.device("cpu")


@pytest.fixture
def gpu_device():
    """Fixture for mocked GPU device object."""
    return torch.device("cuda")


@pytest.fixture
def mock_tokenizer_output():
    """Fixture providing a mock tokenizer output object (like BatchEncoding)."""
    mock_output = MagicMock()
    mock_output.input_ids = torch.tensor([[101, 1000, 2000, 102]])
    mock_output.attention_mask = torch.tensor([[1, 1, 1, 1]])
    mock_output.to = MagicMock(return_value=mock_output)
    return mock_output


@pytest.fixture
def base_model(cpu_device, mock_tokenizer_output):
    """Fixture for BaseTransformerModel initialized for CPU."""
    with patch("torch.cuda.is_available", return_value=False):
        model = ConcreteModel("test-base-cpu-model")
        model.tokenizer = MagicMock(return_value=mock_tokenizer_output)
        model.device = cpu_device
        return model


@pytest.fixture
def base_model_gpu(gpu_device, mock_tokenizer_output):
    """Fixture for BaseTransformerModel initialized for GPU."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.device") as mock_torch_device:
            mock_torch_device.return_value = gpu_device
            model = ConcreteModel("test-base-gpu-model")
            model.tokenizer = MagicMock(return_value=mock_tokenizer_output)
            model.device = gpu_device
            return model


def test_base_model_initialization_cpu(base_model, cpu_device):
    """Test basic initialization attributes on CPU."""
    assert base_model.model_name == "test-base-cpu-model"
    assert base_model.device == cpu_device
    assert base_model.model is None
    assert base_model.tokenizer is not None
    assert base_model.pipe is None
    assert base_model.use_float16 is False


def test_base_model_initialization_gpu(base_model_gpu, gpu_device):
    """Test basic initialization attributes on GPU."""
    assert base_model_gpu.model_name == "test-base-gpu-model"
    assert base_model_gpu.device == gpu_device
    assert base_model_gpu.model is None
    assert base_model_gpu.tokenizer is not None
    assert base_model_gpu.pipe is None
    assert base_model_gpu.use_float16 is False


@patch("torch.cuda.is_available", return_value=False)
def test_setup_model_precision_cpu(mock_cuda_check, base_model):
    """Test precision setup defaults to float on CPU."""
    mock_nn_model = MagicMock(spec=torch.nn.Module)
    mock_nn_model.float.return_value.to.return_value = mock_nn_model

    processed_model, use_float16 = base_model._setup_model_precision(mock_nn_model)

    assert use_float16 is False
    mock_nn_model.float.assert_called_once()
    mock_nn_model.float.return_value.to.assert_called_once_with(base_model.device)
    mock_nn_model.half.assert_not_called()
    assert processed_model is mock_nn_model


@patch("torch.cuda.is_available", return_value=True)
def test_setup_model_precision_gpu(mock_cuda_check, base_model_gpu):
    """Test precision setup uses half precision on GPU."""
    mock_nn_model = MagicMock(spec=torch.nn.Module)
    mock_nn_model.half.return_value.to.return_value = mock_nn_model
    processed_model, use_float16 = base_model_gpu._setup_model_precision(mock_nn_model)
    assert use_float16 is True
    mock_nn_model.half.assert_called_once()
    mock_nn_model.half.return_value.to.assert_called_once_with(base_model_gpu.device)
    mock_nn_model.float.assert_not_called()
    assert processed_model is mock_nn_model


@patch("torch.cuda.is_available", return_value=True)
@patch("src.domain.transformers.base.logger")
def test_setup_model_precision_gpu_fallback(
    mock_logger, mock_cuda_check, base_model_gpu
):
    """Test precision setup falls back to float on GPU if half() fails."""
    mock_nn_model = MagicMock(spec=torch.nn.Module)
    mock_nn_model.half.side_effect = RuntimeError("Half precision not supported")
    mock_nn_model.float.return_value.to.return_value = mock_nn_model

    processed_model, use_float16 = base_model_gpu._setup_model_precision(mock_nn_model)
    assert use_float16 is False
    mock_nn_model.half.assert_called_once()
    mock_nn_model.float.assert_called_once()
    mock_nn_model.float.return_value.to.assert_called_once_with(base_model_gpu.device)
    assert processed_model is mock_nn_model
    mock_logger.warning.assert_called_once()
    assert "Failed to use half precision" in mock_logger.warning.call_args[0][0]


def test_tokenize(base_model, mock_tokenizer_output):
    """Test the internal _tokenize method."""
    text = "sample text"
    device_mock = base_model.device

    inputs, token_count = base_model._tokenize(text)

    base_model.tokenizer.assert_called_once_with(
        text, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    mock_tokenizer_output.to.assert_called_once_with(device_mock)

    assert inputs is mock_tokenizer_output

    assert hasattr(inputs, "input_ids")
    assert torch.equal(inputs.input_ids, torch.tensor([[101, 1000, 2000, 102]]))
    assert token_count == 4


def test_tokenize_list_input(base_model, mock_tokenizer_output):
    """Test _tokenize with a list of strings."""
    texts = ["sample text one", "sample text two"]
    device_mock = base_model.device
    mock_tokenizer_output.input_ids = torch.tensor([[101, 1, 2, 102], [101, 3, 4, 102]])
    inputs, token_count = base_model._tokenize(texts)
    base_model.tokenizer.assert_called_once_with(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    mock_tokenizer_output.to.assert_called_once_with(device_mock)
    assert inputs is mock_tokenizer_output
    assert token_count == 4


def test_tokenize_no_tokenizer(base_model):
    """Test _tokenize raises error if tokenizer is not initialized."""
    base_model.tokenizer = None
    with pytest.raises(RuntimeError, match="Tokenizer not initialized."):
        base_model._tokenize("some text")


@pytest.mark.asyncio
@patch("torch.cuda.empty_cache")
@patch("src.domain.transformers.base.logger")
async def test_close(mock_logger, mock_empty_cache, base_model):
    """Test the close method clears resources."""
    mock_model = MagicMock()
    mock_model.cpu = MagicMock()
    base_model.model = mock_model
    base_model.tokenizer = MagicMock()
    base_model.pipe = MagicMock()
    await base_model.close()
    assert base_model.model is None
    assert base_model.tokenizer is None
    assert base_model.pipe is None
    mock_empty_cache.assert_not_called()
    mock_logger.info.assert_any_call(
        f"Initiating cleanup for model '{base_model.model_name}'..."
    )
    mock_logger.info.assert_any_call(
        f"Model '{base_model.model_name}' closed and cleaned up successfully."
    )


@pytest.mark.asyncio
@patch("torch.cuda.empty_cache")
@patch("src.domain.transformers.base.logger")
async def test_close_gpu(mock_logger, mock_empty_cache, base_model_gpu):
    """Test the close method clears resources and CUDA cache for GPU."""
    mock_model = MagicMock()
    mock_model.cpu = MagicMock()
    base_model_gpu.model = mock_model
    base_model_gpu.tokenizer = MagicMock()
    base_model_gpu.pipe = MagicMock()

    await base_model_gpu.close()

    assert base_model_gpu.model is None
    assert base_model_gpu.tokenizer is None
    assert base_model_gpu.pipe is None
    mock_empty_cache.assert_called_once()
    mock_logger.info.assert_any_call(
        f"Initiating cleanup for model '{base_model_gpu.model_name}'..."
    )
    mock_logger.info.assert_any_call(
        f"Model '{base_model_gpu.model_name}' closed and cleaned up successfully."
    )


@pytest.mark.asyncio
@patch("src.domain.transformers.base.logger")
async def test_close_error(mock_logger, base_model):
    """Test error handling during the blocking close function."""
    error_message = "Simulated CPU move error"
    mock_model = MagicMock()
    mock_model.cpu.side_effect = RuntimeError(error_message)

    base_model.model = mock_model
    mock_tokenizer_instance = MagicMock()
    base_model.tokenizer = mock_tokenizer_instance
    await base_model.close()
    mock_logger.error.assert_called_once()
    log_args, log_kwargs = mock_logger.error.call_args
    assert "Error during model cleanup" in log_args[0]
    assert error_message in log_args[0]
    assert log_kwargs.get("exc_info") is True

    assert base_model.model is mock_model
    assert base_model.tokenizer is mock_tokenizer_instance
    assert base_model.tokenizer is not None
