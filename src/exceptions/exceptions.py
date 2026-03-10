from fastapi import status

from .base import GuardrailedBaseError


class AuthenticationError(GuardrailedBaseError):
    def __init__(self, message: str = "Authentication failed."):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED, user_facing=True)


class ValidationError(GuardrailedBaseError):
    def __init__(self, message: str = "Invalid input provided."):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, user_facing=True)


class InitializationError(GuardrailedBaseError):
    def __init__(self, component: str, detail: str):
        super().__init__(
            f"Failed to initialize {component}: {detail}",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class NotInitializedError(GuardrailedBaseError):
    def __init__(self, component: str):
        super().__init__(
            f"{component} not initialized.", status.HTTP_500_INTERNAL_SERVER_ERROR
        )
