from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@pytest.fixture
def mock_hf_model_instance():
    instance = MagicMock(spec=AutoModelForSequenceClassification)
    instance.eval = MagicMock()
    instance.return_value = MagicMock(logits=torch.tensor([[0.1, 0.9]]))
    return instance


@pytest.fixture
def mock_hf_tokenizer_instance():
    instance = MagicMock(spec=AutoTokenizer)
    return instance


@pytest.fixture(autouse=True)
def patch_transformers_classes(mock_hf_model_instance, mock_hf_tokenizer_instance):
    with (
        patch(
            "src.domain.transformers.classification.AutoModelForSequenceClassification"
        ) as mock_model_cls,
        patch(
            "src.domain.transformers.classification.AutoTokenizer"
        ) as mock_tokenizer_cls,
    ):
        mock_model_cls.from_pretrained.return_value = mock_hf_model_instance
        mock_tokenizer_cls.from_pretrained.return_value = mock_hf_tokenizer_instance
        yield mock_model_cls, mock_tokenizer_cls


@pytest.fixture
def mock_base_methods():
    with (
        patch(
            "src.domain.transformers.base.BaseTransformerModel._setup_model_precision"
        ) as mock_setup,
        patch(
            "src.domain.transformers.base.BaseTransformerModel._tokenize"
        ) as mock_tokenize,
    ):

        def setup_side_effect(model_instance):
            model_instance.eval = MagicMock()
            return model_instance, False

        mock_setup.side_effect = setup_side_effect

        mock_tokenized_output_dict = {
            "input_ids": torch.tensor([[101, 1000, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenize.return_value = (mock_tokenized_output_dict, 3)

        yield mock_setup, mock_tokenize


@pytest.fixture
def classification_model(mock_base_methods):
    with patch("torch.cuda.is_available", return_value=False):
        from src.domain.transformers.classification import ClassificationModel

        model = ClassificationModel("test-class-model")
        model.device = torch.device("cpu")
        return model


@pytest.mark.asyncio
async def test_initialize_success(
    classification_model,
    mock_base_methods,
    patch_transformers_classes,
    mock_hf_model_instance,
    mock_hf_tokenizer_instance,
):
    """Test successful initialization."""
    mock_setup, _ = mock_base_methods
    mock_model_cls, mock_tokenizer_cls = patch_transformers_classes
    mock_setup.side_effect = lambda m: (
        m,
        False,
    )

    await classification_model.initialize()

    mock_model_cls.from_pretrained.assert_called_once_with("test-class-model")
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "test-class-model", clean_up_tokenization_spaces=True
    )
    mock_setup.assert_called_once_with(mock_hf_model_instance)
    assert mock_hf_model_instance.eval.call_count == 1
    assert classification_model.model is mock_hf_model_instance
    assert classification_model.tokenizer is mock_hf_tokenizer_instance
    assert classification_model.use_float16 is False


@pytest.mark.asyncio
async def test_initialize_failure(classification_model, patch_transformers_classes):
    """Test initialization failure when from_pretrained fails."""
    mock_model_cls, _ = patch_transformers_classes
    error_message = "HF model not found"
    mock_model_cls.from_pretrained.side_effect = ValueError(error_message)
    with pytest.raises(ValueError, match=error_message):
        await classification_model.initialize()
    mock_model_cls.from_pretrained.assert_called_once()


@pytest.mark.asyncio
async def test_predict_not_initialized(classification_model):
    """Test predict raises error if model not initialized."""
    assert classification_model.model is None
    with pytest.raises(RuntimeError, match="Model not initialized"):
        await classification_model.predict("test text")


@pytest.mark.asyncio
async def test_predict_success_cpu(
    classification_model,
    mock_base_methods,
    mock_hf_model_instance,
    mock_hf_tokenizer_instance,
):
    """Test successful prediction on CPU."""
    mock_setup, mock_tokenize = mock_base_methods

    classification_model.model = mock_hf_model_instance
    classification_model.tokenizer = mock_hf_tokenizer_instance
    classification_model.use_float16 = False
    classification_model.device = torch.device("cpu")

    mock_output = MagicMock(logits=torch.tensor([[0.1, 0.9]]))
    mock_hf_model_instance.return_value = mock_output

    mock_tokenized_dict, expected_token_count = mock_tokenize.return_value

    text = "This is positive sentiment."
    result, token_count = await classification_model.predict(text)

    mock_tokenize.assert_called_once_with(text)

    mock_hf_model_instance.assert_called_once_with(**mock_tokenized_dict)

    assert isinstance(result, tuple) and len(result) == 2
    assert result[0] + result[1] == pytest.approx(1.0)
    assert result[1] > result[0]
    assert token_count == expected_token_count


@pytest.mark.asyncio
@patch("torch.amp.autocast")
async def test_predict_success_gpu(
    mock_autocast,
    classification_model,
    mock_base_methods,
    mock_hf_model_instance,
    mock_hf_tokenizer_instance,
):
    """Test successful prediction on GPU with float16."""
    mock_setup, mock_tokenize = mock_base_methods

    classification_model.model = mock_hf_model_instance
    classification_model.tokenizer = mock_hf_tokenizer_instance
    classification_model.use_float16 = True
    classification_model.device = torch.device("cuda")

    mock_output = MagicMock(logits=torch.tensor([[0.8, 0.2]]))
    mock_hf_model_instance.return_value = mock_output

    mock_tokenized_dict, expected_token_count = mock_tokenize.return_value

    text = "This is negative sentiment."
    result, token_count = await classification_model.predict(text)

    mock_tokenize.assert_called_once_with(text)
    mock_autocast.assert_called_once_with("cuda")
    assert mock_hf_model_instance.call_count == 1

    args, kwargs = mock_hf_model_instance.call_args
    assert "input_ids" in kwargs
    assert "attention_mask" in kwargs
    assert torch.equal(kwargs["input_ids"], mock_tokenized_dict["input_ids"])
    assert torch.equal(kwargs["attention_mask"], mock_tokenized_dict["attention_mask"])

    assert isinstance(result, tuple) and len(result) == 2
    assert result[0] > result[1]
    assert result[0] + result[1] == pytest.approx(1.0)
    assert token_count == expected_token_count


@pytest.mark.asyncio
async def test_predict_failure(
    classification_model,
    mock_base_methods,
    mock_hf_model_instance,
    mock_hf_tokenizer_instance,
):
    """Test prediction failure during model inference."""
    mock_setup, mock_tokenize = mock_base_methods

    classification_model.model = mock_hf_model_instance
    classification_model.tokenizer = mock_hf_tokenizer_instance
    classification_model.use_float16 = False
    classification_model.device = torch.device("cpu")

    error_message = "Inference crashed"
    mock_hf_model_instance.side_effect = RuntimeError(error_message)

    mock_tokenized_dict, _ = mock_tokenize.return_value

    with pytest.raises(RuntimeError, match=error_message):
        await classification_model.predict("some text")

    mock_tokenize.assert_called_once_with("some text")

    mock_hf_model_instance.assert_called_once_with(**mock_tokenized_dict)
