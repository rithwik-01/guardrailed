import json

import pytest
from fastapi import FastAPI, Request, status
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)

from src.exceptions import (
    AuthenticationError,
    InitializationError,
    NotInitializedError,
    GuardrailedBaseError,
    GuardrailedHTTPException,
    ValidationError,
)
from src.exceptions.handlers import setup_exception_handlers
from src.shared import Action, SafetyCode


@pytest.fixture
def app_with_handlers():
    """Create a FastAPI app with exception handlers set up."""
    app = FastAPI()

    @app.get("/guardrailed-base-500")
    async def raise_guardrailed_base_500():
        raise GuardrailedBaseError(
            "Base 500 (Internal)", status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    @app.get("/guardrailed-base-400-user")
    async def raise_guardrailed_base_400_user():
        raise GuardrailedBaseError(
            "Base 400 (User)", status.HTTP_400_BAD_REQUEST, user_facing=True
        )

    @app.get("/guardrailed-base-400-internal")
    async def raise_guardrailed_base_400_internal():
        raise GuardrailedBaseError(
            "Base 400 (Internal)", status.HTTP_400_BAD_REQUEST, user_facing=False
        )

    @app.get("/auth-error")
    async def raise_auth():
        raise AuthenticationError("Invalid API Key")

    @app.get("/validation-error")
    async def raise_validation():
        raise ValidationError("Input field 'email' is invalid")

    @app.get("/init-error")
    async def raise_init_error():
        raise InitializationError("ComponentX", "Config file missing")

    @app.get("/not-init-error")
    async def raise_not_init_error():
        raise NotInitializedError("Database Connection")

    @app.get("/guardrailed-http-exception")
    async def raise_guardrailed_http():
        raise GuardrailedHTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message="Rate limit exceeded",
            safety_code=SafetyCode.GENERIC_UNSAFE,
            action=Action.RETRY.value,
        )

    @app.get("/unhandled-value-error")
    async def raise_unhandled_value():
        raise ValueError("Unexpected value encountered")

    @app.get("/unhandled-generic-exception")
    async def raise_unhandled_generic():
        raise Exception("A generic unexpected error")

    class Item(BaseModel):
        name: str
        price: float

    @app.post("/pydantic-validation-error")
    async def pydantic_validation_error(item: Item):
        return item

    @app.get("/starlette-http-exception")
    async def starlette_http_exception():
        raise StarletteHTTPException(status_code=404, detail="Native Not Found")

    @app.post("/json-decode-error")
    async def json_decode_error_endpoint(request: Request):
        try:
            await request.json()
            return {"status": "ok"}
        except json.JSONDecodeError as e:
            raise e

    setup_exception_handlers(app)
    return app


@pytest.fixture
def client(app_with_handlers: FastAPI):
    """Test client for the app with handlers."""
    return TestClient(app_with_handlers, raise_server_exceptions=False)


def test_guardrailed_base_500_handler(client: TestClient):
    response = client.get("/guardrailed-base-500")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["safety_code"] == SafetyCode.UNEXPECTED
    assert data["message"] == "An unexpected error occurred. Please try again later."
    assert data["action"] == Action.OVERRIDE.value


def test_guardrailed_base_400_user_handler(client: TestClient):
    response = client.get("/guardrailed-base-400-user")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["safety_code"] == SafetyCode.GENERIC_UNSAFE
    assert data["message"] == "Base 400 (User)"
    assert data["action"] == Action.OVERRIDE.value


def test_guardrailed_base_400_internal_handler(client: TestClient):
    response = client.get("/guardrailed-base-400-internal")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["safety_code"] == SafetyCode.GENERIC_UNSAFE
    assert data["message"] == "An unexpected error occurred. Please try again later."
    assert data["action"] == Action.OVERRIDE.value


def test_authentication_error_handler(client: TestClient):
    response = client.get("/auth-error")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    data = response.json()
    assert data["safety_code"] == SafetyCode.GENERIC_UNSAFE
    assert data["message"] == "Invalid API Key"
    assert data["action"] == Action.OVERRIDE.value


def test_validation_error_handler(client: TestClient):
    response = client.get("/validation-error")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["safety_code"] == SafetyCode.GENERIC_UNSAFE
    assert data["message"] == "Input field 'email' is invalid"
    assert data["action"] == Action.OVERRIDE.value


def test_initialization_error_handler(client: TestClient):
    response = client.get("/init-error")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["safety_code"] == SafetyCode.UNEXPECTED
    assert data["message"] == "An unexpected error occurred. Please try again later."
    assert data["action"] == Action.OVERRIDE.value


def test_not_initialized_error_handler(client: TestClient):
    response = client.get("/not-init-error")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["safety_code"] == SafetyCode.UNEXPECTED
    assert data["message"] == "An unexpected error occurred. Please try again later."
    assert data["action"] == Action.OVERRIDE.value


def test_guardrailed_http_exception_handler(client: TestClient):
    response = client.get("/guardrailed-http-exception")
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    data = response.json()
    assert data["safety_code"] == SafetyCode.GENERIC_UNSAFE
    assert data["message"] == "Rate limit exceeded"
    assert data["action"] == Action.RETRY.value


def test_unhandled_value_error_handler(client: TestClient):
    response = client.get("/unhandled-value-error")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["safety_code"] == SafetyCode.UNEXPECTED
    assert data["message"] == "An unexpected internal server error occurred."
    assert data["action"] == Action.OVERRIDE.value


def test_unhandled_generic_exception_handler(client: TestClient):
    response = client.get("/unhandled-generic-exception")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["safety_code"] == SafetyCode.UNEXPECTED
    assert data["message"] == "An unexpected internal server error occurred."
    assert data["action"] == Action.OVERRIDE.value


def test_json_decode_error_handler(client: TestClient):
    response = client.post("/json-decode-error", data="{invalid json")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["safety_code"] == SafetyCode.GENERIC_UNSAFE
    assert "Invalid JSON payload" in data["message"]
    assert data["action"] == Action.OVERRIDE.value


def test_fastapi_validation_error_handler(client: TestClient):
    response = client.post("/pydantic-validation-error", json={"name": "test"})

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["safety_code"] == SafetyCode.GENERIC_UNSAFE
    assert "message" in data
    assert data["action"] == Action.OVERRIDE.value

    assert data["message"].startswith("Invalid input for field")
    assert "'body.price'" in data["message"]
    assert "Field required" in data["message"]
    assert "Invalid request input." != data["message"]


def test_starlette_http_exception_handler(client: TestClient):
    response = client.get("/starlette-http-exception")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    data = response.json()
    assert data["safety_code"] == SafetyCode.GENERIC_UNSAFE
    assert data["message"] == "Native Not Found"
    assert data["action"] == Action.OVERRIDE.value
