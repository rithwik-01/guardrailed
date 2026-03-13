from unittest.mock import ANY, MagicMock, patch

import pytest
import torch

mock_auto_tokenizer_ner = MagicMock()
mock_auto_model_ner = MagicMock()
mock_pipeline_ner = MagicMock()


@pytest.fixture(autouse=True)
def patch_transformers_ner():
    with (
        patch("src.domain.transformers.ner.AutoTokenizer", mock_auto_tokenizer_ner),
        patch(
            "src.domain.transformers.ner.AutoModelForTokenClassification",
            mock_auto_model_ner,
        ),
        patch("src.domain.transformers.ner.pipeline", mock_pipeline_ner),
    ):
        yield


@pytest.fixture
def mock_base_methods_ner():
    with (
        patch(
            "src.domain.transformers.base.BaseTransformerModel._setup_model_precision"
        ) as mock_setup,
        patch(
            "src.domain.transformers.base.BaseTransformerModel._tokenize"
        ) as mock_tokenize,
    ):
        mock_setup.return_value = (
            MagicMock(eval=MagicMock()),
            False,
        )
        mock_tokenize.return_value = (
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            },
            2,
        )
        yield mock_setup, mock_tokenize


@pytest.fixture
def ner_model(mock_base_methods_ner):
    with patch("torch.cuda.is_available", return_value=False):
        from src.domain.transformers.ner import NERModel

        model = NERModel("test-ner-model")
        model.device = torch.device("cpu")
        return model


@pytest.mark.asyncio
async def test_initialize_success_ner(ner_model, mock_base_methods_ner):
    """Test successful initialization."""
    mock_setup, _ = mock_base_methods_ner
    mock_hf_model_instance = MagicMock()
    mock_hf_tokenizer_instance = MagicMock()
    mock_pipeline_instance = MagicMock()

    mock_auto_model_ner.from_pretrained.return_value = mock_hf_model_instance
    mock_auto_tokenizer_ner.from_pretrained.return_value = mock_hf_tokenizer_instance
    mock_setup.return_value = (
        mock_hf_model_instance,
        False,
    )
    mock_pipeline_ner.return_value = mock_pipeline_instance

    await ner_model.initialize()

    mock_auto_model_ner.from_pretrained.assert_called_once_with("test-ner-model")
    mock_auto_tokenizer_ner.from_pretrained.assert_called_once_with("test-ner-model")
    mock_setup.assert_called_once_with(mock_hf_model_instance)
    mock_hf_model_instance.eval.assert_called_once()

    mock_pipeline_ner.assert_called_once_with(
        "ner",
        model=mock_hf_model_instance,
        tokenizer=mock_hf_tokenizer_instance,
        device=ANY,
        framework="pt",
        aggregation_strategy="simple",
    )

    assert ner_model.model is mock_hf_model_instance
    assert ner_model.tokenizer is mock_hf_tokenizer_instance
    assert ner_model.pipe is mock_pipeline_instance
    assert ner_model.use_float16 is False


@pytest.mark.asyncio
async def test_initialize_failure_ner(ner_model):
    """Test initialization failure."""
    error_message = "HF NER model not found"
    mock_auto_model_ner.from_pretrained.side_effect = ValueError(error_message)

    with pytest.raises(ValueError, match=error_message):
        await ner_model.initialize()


@pytest.mark.asyncio
async def test_predict_not_initialized_ner(ner_model):
    """Test predict raises error if model not initialized."""
    with pytest.raises(RuntimeError, match="Model not initialized"):
        await ner_model.predict("test text")


@pytest.mark.asyncio
async def test_predict_success_cpu_ner(ner_model, mock_base_methods_ner):
    """Test successful prediction on CPU."""
    mock_setup, mock_tokenize = mock_base_methods_ner
    mock_pipeline_instance = MagicMock()
    expected_results = [
        {"entity_group": "LOC", "score": 0.99, "word": "London", "start": 5, "end": 11}
    ]
    mock_pipeline_instance.return_value = expected_results

    ner_model.model = MagicMock()
    ner_model.tokenizer = MagicMock()
    ner_model.pipe = mock_pipeline_instance
    ner_model.use_float16 = False
    ner_model.device = torch.device("cpu")

    text = "Visit London today."
    results, token_count = await ner_model.predict(text)

    mock_tokenize.assert_called_once_with(text)
    mock_pipeline_instance.assert_called_once_with(text)
    assert results == expected_results
    assert token_count == 2


@pytest.mark.asyncio
@patch("torch.amp.autocast")
async def test_predict_success_gpu_ner(mock_autocast, ner_model, mock_base_methods_ner):
    """Test successful prediction on GPU."""
    mock_setup, mock_tokenize = mock_base_methods_ner
    mock_pipeline_instance = MagicMock()
    expected_results = [
        {"entity_group": "PER", "score": 0.98, "word": "Alice", "start": 0, "end": 5}
    ]
    mock_pipeline_instance.return_value = expected_results

    ner_model.model = MagicMock()
    ner_model.tokenizer = MagicMock()
    ner_model.pipe = mock_pipeline_instance
    ner_model.use_float16 = True  # GPU case
    ner_model.device = torch.device("cuda")

    text = "Alice went."
    results, token_count = await ner_model.predict(text)

    mock_tokenize.assert_called_once_with(text)
    mock_autocast.assert_called_once_with("cuda")
    mock_pipeline_instance.assert_called_once_with(text)
    assert results == expected_results
    assert token_count == 2


@pytest.mark.asyncio
async def test_predict_failure_ner(ner_model, mock_base_methods_ner):
    """Test prediction failure during pipeline call."""
    mock_setup, mock_tokenize = mock_base_methods_ner
    mock_pipeline_instance = MagicMock()
    error_message = "Pipeline error"
    mock_pipeline_instance.side_effect = RuntimeError(error_message)

    ner_model.model = MagicMock()
    ner_model.tokenizer = MagicMock()
    ner_model.pipe = mock_pipeline_instance
    ner_model.use_float16 = False
    ner_model.device = torch.device("cpu")

    with pytest.raises(RuntimeError, match=error_message):
        await ner_model.predict("some text")

    mock_tokenize.assert_called_once()
    mock_pipeline_instance.assert_called_once()
