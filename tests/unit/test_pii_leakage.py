from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from presidio_analyzer import RecognizerResult

from src.domain.validators.pii_leakage.main import _get_entities_to_scan, check_pii
from src.exceptions import NotInitializedError
from src.shared import (
    Action,
    Policy,
    PolicyType,
    SafetyCode,
)


@pytest.fixture
def mock_analyzer_engine():
    """Fixture for a mocked Presidio AnalyzerEngine instance."""
    engine = MagicMock(spec=True)
    engine.analyze = MagicMock()
    return engine


@pytest.fixture(autouse=True)
def patch_app_state_for_pii(mock_analyzer_engine):
    """Automatically patch app_state for PII tests."""
    mock_state = MagicMock()
    mock_state.presidio_analyzer_engine = mock_analyzer_engine
    mock_state.presidio_anonymizer_engine = None
    mock_state.profanity_model = None
    mock_state.ner_model = None
    with patch("src.domain.validators.pii_leakage.main.app_state", mock_state):
        yield mock_state


@pytest.fixture
def pii_policy_entities():
    """PII policy specifying entities."""
    return Policy(
        id=PolicyType.PII_LEAKAGE.value,
        name="PII Entities (Block)",
        state=True,
        action=Action.OVERRIDE.value,
        message="PII Entity Detected",
        is_user_policy=True,
        is_llm_policy=True,
        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
        pii_threshold=0.6,
        pii_categories=None,
        prompt_leakage_threshold=0.85,
        metadata={},
    )


@pytest.fixture
def pii_policy_categories():
    """PII policy specifying categories."""
    return Policy(
        id=PolicyType.PII_LEAKAGE.value,
        name="PII Categories (Observe)",
        state=True,
        action=Action.OBSERVE.value,
        message="PII Category Detected",
        is_user_policy=True,
        is_llm_policy=True,
        pii_categories=["US_SPECIFIC", "FINANCIAL"],
        pii_threshold=0.7,
        pii_entities=None,
        prompt_leakage_threshold=0.85,
        metadata={},
    )


@pytest.fixture
def pii_policy_default():
    """PII policy without specific entities or categories (use Presidio defaults)."""
    return Policy(
        id=PolicyType.PII_LEAKAGE.value,
        name="PII Default (Block)",
        state=True,
        action=Action.OVERRIDE.value,
        message="Default PII Detected",
        is_user_policy=True,
        is_llm_policy=True,
        pii_categories=None,
        pii_entities=None,
        pii_threshold=0.5,
        prompt_leakage_threshold=0.85,
        metadata={},
    )


@pytest.fixture
def mock_presidio_email_result():
    """Mock result for an email address."""
    return RecognizerResult(entity_type="EMAIL_ADDRESS", start=11, end=28, score=0.85)


@pytest.fixture
def mock_presidio_phone_result():
    """Mock result for a phone number."""
    return RecognizerResult(entity_type="PHONE_NUMBER", start=8, end=20, score=0.7)


def test_get_entities_to_scan_uses_entities(pii_policy_entities):
    assert _get_entities_to_scan(pii_policy_entities) == [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
    ]


def test_get_entities_to_scan_uses_categories(pii_policy_categories):
    assert _get_entities_to_scan(pii_policy_categories) is None


def test_get_entities_to_scan_uses_default(pii_policy_default):
    assert _get_entities_to_scan(pii_policy_default) is None


@pytest.mark.asyncio
async def test_check_pii_engine_not_initialized(
    patch_app_state_for_pii, pii_policy_default
):
    patch_app_state_for_pii.presidio_analyzer_engine = None
    prompt = "Some text"
    with pytest.raises(NotInitializedError, match="Presidio Analyzer Engine"):
        await check_pii(prompt, pii_policy_default)


@pytest.mark.asyncio
async def test_check_pii_no_pii_found(mock_analyzer_engine, pii_policy_entities):
    mock_analyzer_engine.analyze.return_value = []
    prompt = "This is a perfectly safe message."
    result = await check_pii(prompt, pii_policy_entities)
    assert result.safety_code == SafetyCode.SAFE
    assert result.status == status.HTTP_200_OK
    mock_analyzer_engine.analyze.assert_called_once()
    call_kwargs = mock_analyzer_engine.analyze.call_args.kwargs
    assert call_kwargs["entities"] == pii_policy_entities.pii_entities
    assert call_kwargs["score_threshold"] == pii_policy_entities.pii_threshold


@pytest.mark.asyncio
async def test_check_pii_found_above_threshold_block(
    mock_analyzer_engine, pii_policy_entities, mock_presidio_email_result
):
    policy = pii_policy_entities
    policy.pii_threshold = 0.6
    mock_analyzer_engine.analyze.return_value = [mock_presidio_email_result]
    prompt = "My email: test@test.com"
    result = await check_pii(prompt, policy)
    assert result.safety_code == SafetyCode.PII_DETECTED
    assert result.status == status.HTTP_400_BAD_REQUEST
    assert result.action == Action.OVERRIDE.value


@pytest.mark.asyncio
async def test_check_pii_found_above_threshold_observe(
    mock_analyzer_engine, pii_policy_categories, mock_presidio_phone_result
):
    policy = pii_policy_categories
    policy.pii_threshold = 0.6
    mock_analyzer_engine.analyze.return_value = [mock_presidio_phone_result]
    prompt = "Call me 555-1234"
    result = await check_pii(prompt, policy)
    assert result.safety_code == SafetyCode.SAFE
    assert result.status == status.HTTP_200_OK
    assert result.action is None
    mock_analyzer_engine.analyze.assert_called_once()
    assert mock_analyzer_engine.analyze.call_args.kwargs["entities"] is None


@pytest.mark.asyncio
async def test_check_pii_uses_default_entities(
    mock_analyzer_engine, pii_policy_default, mock_presidio_email_result
):
    policy = pii_policy_default
    mock_analyzer_engine.analyze.return_value = [mock_presidio_email_result]
    prompt = "Send to default@test.net"
    result = await check_pii(prompt, policy)
    assert result.safety_code == SafetyCode.PII_DETECTED
    assert result.status == status.HTTP_400_BAD_REQUEST
    mock_analyzer_engine.analyze.assert_called_once()
    assert mock_analyzer_engine.analyze.call_args.kwargs["entities"] is None


@pytest.mark.asyncio
async def test_check_pii_analyzer_exception(mock_analyzer_engine, pii_policy_default):
    mock_analyzer_engine.analyze.side_effect = Exception("Presidio crashed")
    prompt = "Some text"
    result = await check_pii(prompt, pii_policy_default)
    assert result.safety_code == SafetyCode.UNEXPECTED
    assert result.status == status.HTTP_500_INTERNAL_SERVER_ERROR
