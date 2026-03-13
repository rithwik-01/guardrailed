from fastapi import HTTPException, status

from src.exceptions import (
    AuthenticationError,
    InitializationError,
    NotInitializedError,
    GuardrailedBaseError,
    GuardrailedHTTPException,
    ValidationError,
)
from src.shared import Action, SafetyCode


def test_guardrailed_base_error():
    """Test that GuardrailedBaseError initializes correctly."""
    error = GuardrailedBaseError("Base error", status.HTTP_500_INTERNAL_SERVER_ERROR)
    assert error.message == "Base error"
    assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert error.user_facing is False
    assert str(error) == "Base error"

    error_user = GuardrailedBaseError(
        "User-facing base error", status.HTTP_400_BAD_REQUEST, user_facing=True
    )
    assert error_user.message == "User-facing base error"
    assert error_user.status_code == status.HTTP_400_BAD_REQUEST
    assert error_user.user_facing is True


def test_authentication_error():
    """Test AuthenticationError initialization."""
    error_default = AuthenticationError()
    assert error_default.message == "Authentication failed."
    assert error_default.status_code == status.HTTP_401_UNAUTHORIZED
    assert error_default.user_facing is True

    error_custom = AuthenticationError("Custom auth message")
    assert error_custom.message == "Custom auth message"
    assert error_custom.status_code == status.HTTP_401_UNAUTHORIZED


def test_validation_error():
    """Test ValidationError initialization."""
    error_default = ValidationError()
    assert error_default.message == "Invalid input provided."
    assert error_default.status_code == status.HTTP_400_BAD_REQUEST
    assert error_default.user_facing is True

    error_custom = ValidationError("Specific field validation failed")
    assert error_custom.message == "Specific field validation failed"


def test_initialization_error():
    """Test InitializationError initialization."""
    error = InitializationError("Cache", "Connection refused")
    assert error.message == "Failed to initialize Cache: Connection refused"
    assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert error.user_facing is False


def test_not_initialized_error():
    """Test NotInitializedError initialization."""
    error = NotInitializedError("LLM Model")
    assert error.message == "LLM Model not initialized."
    assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert error.user_facing is False


def test_guardrailed_http_exception():
    """Test GuardrailedHTTPException initialization."""
    error = GuardrailedHTTPException(
        status_code=418,
        message="Custom HTTP message",
        safety_code=SafetyCode.GENERIC_UNSAFE,
        action=Action.OBSERVE.value,
    )
    assert error.status_code == 418
    assert error.detail == "Custom HTTP message"
    assert error.message == "Custom HTTP message"
    assert error.safety_code == SafetyCode.GENERIC_UNSAFE
    assert error.action == Action.OBSERVE.value
    assert not isinstance(error, GuardrailedBaseError)
    assert isinstance(error, HTTPException)
