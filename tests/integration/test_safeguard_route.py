from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from src.exceptions import NotInitializedError, ValidationError
from src.main import app
from src.presentation.dependencies import get_loaded_policies
from src.shared import Action, Policy, PolicyType, Result, SafetyCode, Status


@pytest.fixture(scope="module")
def client():
    """Return a TestClient for the application. Module scope is fine."""
    app.dependency_overrides = {}
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides = {}


@pytest.fixture
def sample_policy_list():
    """Provides a list of sample policies for dependency override."""
    return [
        Policy(
            id=PolicyType.PII_LEAKAGE.value,
            name="PII Block",
            state=True,
            action=Action.OVERRIDE.value,
            message="PII violation - Blocked",
            is_user_policy=True,
            is_llm_policy=True,
            pii_entities=None,
            pii_threshold=0.5,
            pii_categories=None,
            threshold=None,
            protected_prompts=None,
            prompt_leakage_threshold=0.85,
            locations=None,
            persons=None,
            competitors=None,
            metadata={},
        ),
        Policy(
            id=PolicyType.PROFANITY.value,
            name="Toxicity Observe",
            state=True,
            action=Action.OBSERVE.value,
            threshold=0.8,
            message="Toxic content - Observed",
            is_user_policy=True,
            is_llm_policy=True,
            pii_entities=None,
            pii_threshold=0.5,
            pii_categories=None,
            protected_prompts=None,
            prompt_leakage_threshold=0.85,
            locations=None,
            persons=None,
            competitors=None,
            metadata={},
        ),
    ]


@pytest.mark.parametrize(
    "test_id, mock_validator_return, request_payload, expected_status, expected_response_json",
    [
        (
            "safe_basic",
            Result.safe_result(),
            {"messages": [{"role": "user", "content": "This is fine."}]},
            status.HTTP_200_OK,
            {
                "safety_code": SafetyCode.SAFE,
                "message": "Message validated successfully",
                "action": None,
            },
        ),
        (
            "unsafe_pii_override",
            Status(
                status=status.HTTP_400_BAD_REQUEST,
                message="PII Violation - Blocked",
                safety_code=SafetyCode.PII_DETECTED,
                action=Action.OVERRIDE.value,
            ),
            {"messages": [{"role": "user", "content": "email@example.com"}]},
            status.HTTP_400_BAD_REQUEST,
            {
                "safety_code": SafetyCode.PII_DETECTED,
                "message": "PII Violation - Blocked",
                "action": Action.OVERRIDE.value,
            },
        ),
        (
            "unsafe_profane_observe",
            Status(
                status=status.HTTP_200_OK,
                message="Toxic but observed",
                safety_code=SafetyCode.PROFANE,
                action=Action.OBSERVE.value,
            ),
            {"messages": [{"role": "user", "content": "darn it"}]},
            status.HTTP_200_OK,
            {
                "safety_code": SafetyCode.PROFANE,
                "message": "Toxic but observed",
                "action": Action.OBSERVE.value,
            },
        ),
        (
            "validator_internal_error",
            Status(
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Internal validator failure",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.OVERRIDE.value,
            ),
            {"messages": [{"role": "user", "content": "trigger internal path"}]},
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            {
                "safety_code": SafetyCode.UNEXPECTED,
                "message": "Internal validator failure",
                "action": Action.OVERRIDE.value,
            },
        ),
    ],
)
@patch("src.presentation.routes.safeguard.ContentValidator", new_callable=MagicMock)
def test_safeguard_scenarios(
    mock_validator_cls: MagicMock,
    client: TestClient,
    sample_policy_list: list[Policy],
    test_id: str,
    mock_validator_return: Status,
    request_payload: dict,
    expected_status: int,
    expected_response_json: dict,
):
    """Test various safeguard scenarios using the OS core structure."""
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_content = AsyncMock(
        return_value=mock_validator_return
    )
    mock_validator_cls.return_value = mock_validator_instance

    async def override_policies():
        return sample_policy_list

    original_overrides = client.app.dependency_overrides.copy()
    client.app.dependency_overrides[get_loaded_policies] = override_policies

    try:
        response = client.post("/safeguard", json=request_payload)

        print(f"Response for {test_id}: {response.status_code} - {response.text}")
        assert (
            response.status_code == expected_status
        ), f"Test ID '{test_id}': Status mismatch"
        assert (
            response.json() == expected_response_json
        ), f"Test ID '{test_id}': JSON body mismatch"

        if expected_status != status.HTTP_400_BAD_REQUEST:
            mock_validator_cls.assert_called_once()
            call_args, _ = mock_validator_cls.call_args
            validation_context_arg = call_args[0]
            assert validation_context_arg.policies == sample_policy_list
            assert validation_context_arg.messages == request_payload["messages"]
            mock_validator_instance.validate_content.assert_awaited_once()

    finally:
        client.app.dependency_overrides = original_overrides


@patch("src.presentation.routes.safeguard.get_messages")
def test_safeguard_invalid_messages_payload(
    mock_get_messages: MagicMock, client: TestClient, sample_policy_list: list[Policy]
):
    """Test safeguard endpoint handles ValidationError from get_messages correctly."""
    error_message = "Invalid message structure provided by client."
    mock_get_messages.side_effect = ValidationError(error_message)

    async def override_policies():
        return sample_policy_list

    client.app.dependency_overrides[get_loaded_policies] = override_policies

    try:
        response = client.post("/safeguard", json={"messages": "not-a-list"})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_json = response.json()
        assert error_message in response_json["message"]
        assert response_json["safety_code"] == SafetyCode.GENERIC_UNSAFE
        assert response_json["action"] == Action.OVERRIDE.value

        mock_get_messages.assert_called_once()

    finally:
        del client.app.dependency_overrides[get_loaded_policies]


def test_safeguard_empty_messages_list(
    client: TestClient, sample_policy_list: list[Policy]
):
    """Test safeguard endpoint returns 400 if the messages list is empty (route check)."""

    async def override_policies():
        return sample_policy_list

    client.app.dependency_overrides[get_loaded_policies] = override_policies

    try:
        response = client.post("/safeguard", json={"messages": []})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_json = response.json()
        assert "Input 'messages' list cannot be empty" in response_json["message"]
        assert response_json["safety_code"] == SafetyCode.GENERIC_UNSAFE
        assert response_json["action"] == Action.OVERRIDE.value

    finally:
        del client.app.dependency_overrides[get_loaded_policies]


@patch("src.presentation.routes.safeguard.ContentValidator", new_callable=MagicMock)
def test_safeguard_client_disconnect_during_validation(
    mock_validator_cls: MagicMock, client: TestClient, sample_policy_list: list[Policy]
):
    """Test safeguard endpoint simulating client disconnect during validation."""
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_content = AsyncMock(side_effect=ClientDisconnect)
    mock_validator_cls.return_value = mock_validator_instance

    async def override_policies():
        return sample_policy_list

    client.app.dependency_overrides[get_loaded_policies] = override_policies

    try:
        response = client.post(
            "/safeguard",
            json={"messages": [{"role": "user", "content": "Disconnect test"}]},
        )

        assert response.status_code == 499
        response_json = response.json()
        assert "Client disconnected" in response_json["message"]
        assert response_json["safety_code"] == SafetyCode.GENERIC_UNSAFE
        assert response_json["action"] == Action.OVERRIDE.value

    finally:
        del client.app.dependency_overrides[get_loaded_policies]


@patch("src.presentation.routes.safeguard.ContentValidator", new_callable=MagicMock)
def test_safeguard_not_initialized_error(
    mock_validator_cls: MagicMock, client: TestClient, sample_policy_list: list[Policy]
):
    """Test safeguard handles NotInitializedError from validator correctly."""
    error_message = "Required component 'XYZ' is not ready."
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_content = AsyncMock(
        side_effect=NotInitializedError(error_message)
    )
    mock_validator_cls.return_value = mock_validator_instance

    async def override_policies():
        return sample_policy_list

    client.app.dependency_overrides[get_loaded_policies] = override_policies

    try:
        response = client.post(
            "/safeguard",
            json={"messages": [{"role": "user", "content": "Test init error"}]},
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        response_json = response.json()
        assert (
            f"Service component not ready: {error_message}" in response_json["message"]
        )
        assert response_json["safety_code"] == SafetyCode.UNEXPECTED
        assert response_json["action"] == Action.RETRY.value

    finally:
        del client.app.dependency_overrides[get_loaded_policies]
