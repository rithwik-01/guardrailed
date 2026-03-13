import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

from src.domain.validators.toxicity.main import check_toxicity
from src.exceptions import NotInitializedError
from src.shared import Action, Policy, PolicyType, Result, SafetyCode

APP_STATE_TARGET = "src.domain.validators.toxicity.main.app_state"
LOGGER_TARGET = "src.domain.validators.toxicity.main.logger"


@pytest.fixture
def mock_profanity_model():
    """Fixture for a mocked profanity model instance."""
    model = AsyncMock(spec=True)
    model.predict = AsyncMock(return_value=((0.8, 0.2), 10))
    return model


@pytest.fixture(autouse=True)
def patch_dependencies(mock_profanity_model):
    """Auto-patches app_state to use the mock model for all tests."""
    mock_state = MagicMock()
    mock_state.profanity_model = mock_profanity_model
    mock_state.ner_model = None
    mock_state.presidio_analyzer_engine = None
    mock_state.presidio_anonymizer_engine = None

    with patch(APP_STATE_TARGET, mock_state):
        yield mock_state


@pytest.fixture
def toxicity_policy():
    """Provides a standard toxicity policy."""
    return Policy(
        id=PolicyType.PROFANITY.value,
        name="Standard Toxicity Block",
        state=True,
        action=Action.OVERRIDE.value,
        threshold=0.5,
        message="Toxic content detected.",
        is_user_policy=True,
        is_llm_policy=True,
    )


@pytest.fixture
def toxicity_policy_no_threshold(toxicity_policy: Policy):
    """Provides a toxicity policy with threshold=None to test default."""
    policy = toxicity_policy.model_copy()
    policy.threshold = None
    return policy


@pytest.fixture
def toxicity_policy_observe(toxicity_policy: Policy):
    """Provides a toxicity policy with action=OBSERVE."""
    policy = toxicity_policy.model_copy()
    policy.action = Action.OBSERVE.value
    policy.message = "Toxic content observed."
    return policy


@pytest.mark.asyncio
async def test_check_toxicity_safe_result(
    mock_profanity_model: AsyncMock, toxicity_policy: Policy, caplog
):
    """Test check_toxicity returns SAFE when score is below threshold."""
    message = "This is a perfectly normal message."
    expected_tokens = 12
    mock_profanity_model.predict.return_value = ((0.7, 0.3), expected_tokens)
    caplog.set_level(logging.WARNING)

    status_result, token_count = await check_toxicity(message, toxicity_policy)

    mock_profanity_model.predict.assert_awaited_once_with(message.lower())
    assert status_result.safety_code == SafetyCode.SAFE
    assert status_result.status == status.HTTP_200_OK
    assert status_result.action is None
    assert status_result.message == Result.safe_result().message
    assert token_count == expected_tokens
    assert "Profanity detected" not in caplog.text


@pytest.mark.asyncio
async def test_check_toxicity_above_threshold_override(
    mock_profanity_model: AsyncMock, toxicity_policy: Policy, caplog
):
    """Test check_toxicity returns PROFANE + OVERRIDE when score is above threshold."""
    message = "This message contains darn profanity!"
    expected_tokens = 15
    profane_score = 0.9
    mock_profanity_model.predict.return_value = ((0.1, profane_score), expected_tokens)
    caplog.set_level(logging.WARNING)

    status_result, token_count = await check_toxicity(message, toxicity_policy)

    mock_profanity_model.predict.assert_awaited_once_with(message.lower())
    assert status_result.safety_code == SafetyCode.PROFANE
    assert status_result.status == status.HTTP_200_OK
    assert status_result.action == toxicity_policy.action
    assert status_result.message == toxicity_policy.message
    assert token_count == expected_tokens

    assert "Profanity detected" in caplog.text
    assert f"above threshold {toxicity_policy.threshold}" in caplog.text
    assert f"Score: {profane_score:.4f}" in caplog.text
    assert f"Action: {Action(toxicity_policy.action).name}" in caplog.text


@pytest.mark.asyncio
async def test_check_toxicity_above_threshold_observe(
    mock_profanity_model: AsyncMock, toxicity_policy_observe: Policy, caplog
):
    """Test check_toxicity returns SAFE + None action for OBSERVE policy."""
    message = "You are really stupid."

    status_result, token_count = await check_toxicity(message, toxicity_policy_observe)

    mock_profanity_model.predict.assert_awaited_once_with(message.lower())
    assert status_result.safety_code == SafetyCode.SAFE
    assert status_result.status == status.HTTP_200_OK
    assert status_result.action is None
    assert status_result.message == Result.safe_result().message


@pytest.mark.asyncio
async def test_check_toxicity_uses_default_threshold(
    mock_profanity_model: AsyncMock, toxicity_policy_no_threshold: Policy, caplog
):
    """Test check_toxicity uses default threshold 0.5 if policy.threshold is None."""
    message = "This message is borderline."
    expected_tokens = 8
    default_threshold = 0.5
    mock_profanity_model.predict.return_value = ((0.4, 0.6), expected_tokens)
    caplog.set_level(logging.WARNING)

    status_result, token_count = await check_toxicity(
        message, toxicity_policy_no_threshold
    )

    mock_profanity_model.predict.assert_awaited_once_with(message.lower())
    assert status_result.safety_code == SafetyCode.PROFANE
    assert status_result.action == toxicity_policy_no_threshold.action

    assert "Profanity detected" in caplog.text
    assert f"above threshold {default_threshold}" in caplog.text

    mock_profanity_model.predict.reset_mock()
    mock_profanity_model.predict.return_value = ((0.51, 0.49), expected_tokens)
    caplog.clear()

    status_result_safe, _ = await check_toxicity(message, toxicity_policy_no_threshold)
    assert status_result_safe.safety_code == SafetyCode.SAFE
    assert "Profanity detected" not in caplog.text


@pytest.mark.asyncio
async def test_check_toxicity_model_not_initialized(
    patch_dependencies,
    toxicity_policy: Policy,
    caplog,
):
    """Test check_toxicity raises NotInitializedError if model is None."""
    patch_dependencies.profanity_model = None
    message = "Any message."
    caplog.set_level(logging.ERROR)

    with pytest.raises(NotInitializedError, match="Profanity model"):
        await check_toxicity(message, toxicity_policy)

    assert "Profanity model not initialized during check" in caplog.text
    assert "Model not initialized" in caplog.text


@pytest.mark.asyncio
@patch(LOGGER_TARGET)
async def test_check_toxicity_predict_exception(
    mock_logger: MagicMock,
    mock_profanity_model: AsyncMock,
    toxicity_policy: Policy,
):
    """Test check_toxicity handles general exceptions during model.predict."""
    message = "This will cause an error."
    error_message = "Prediction failed!"
    mock_profanity_model.predict.side_effect = Exception(error_message)

    status_result, token_count = await check_toxicity(message, toxicity_policy)

    mock_profanity_model.predict.assert_awaited_once_with(message.lower())
    assert status_result.safety_code == SafetyCode.UNEXPECTED
    assert status_result.status == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert status_result.action == Action.OVERRIDE.value
    assert "Internal error during profanity check." in status_result.message
    assert token_count == 0

    mock_logger.error.assert_called_once()
    log_args, log_kwargs = mock_logger.error.call_args
    assert (
        f"Error during profanity check for policy {toxicity_policy.id}" in log_args[0]
    )
    assert error_message in log_args[0]
    assert log_kwargs.get("exc_info") is True
