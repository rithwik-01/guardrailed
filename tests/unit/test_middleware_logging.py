"""
Tests for LoggingMiddleware.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.middleware.logging_middleware import LoggingMiddleware


@pytest.fixture
def app_with_middleware():
    """
    Create a FastAPI app with LoggingMiddleware for testing.
    """
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        if not hasattr(request, "state"):
            request.state = MagicMock()
            if not hasattr(request.state, "request_id"):
                request.state.request_id = "test-req-id"
        return {"message": "test"}

    @app.get("/error")
    async def error_endpoint(request: Request):
        if not hasattr(request, "state"):
            request.state = MagicMock()
            if not hasattr(request.state, "request_id"):
                request.state.request_id = "test-req-id-error"
        raise ValueError("Test error")

    return app


@patch("src.middleware.logging_middleware.logger")
def test_logging_middleware_request_logging(mock_logger, app_with_middleware):
    """
    Test that LoggingMiddleware logs request details.
    """
    client = TestClient(app_with_middleware)

    response = client.get("/test", headers={"user-agent": "test-agent"})

    assert response.status_code == 200

    request_log_call = next(
        (c for c in mock_logger.info.call_args_list if "Request received" in c[0][0]),
        None,
    )
    assert request_log_call is not None, "Info log for 'Request received' not found"
    log_message_req, log_kwargs_req = request_log_call[0], request_log_call[1]

    assert "Request received" in log_message_req[0]
    assert "GET" in log_message_req[0]
    assert "/test" in log_message_req[0]

    extra_context_req = log_kwargs_req.get("extra", {})
    assert extra_context_req.get("method") == "GET"
    assert extra_context_req.get("path") == "/test"
    assert extra_context_req.get("user_agent") == "test-agent"
    assert "request_id" in extra_context_req

    response_log_call = next(
        (c for c in mock_logger.info.call_args_list if "Response sent" in c[0][0]),
        None,
    )
    assert response_log_call is not None, "Info log for 'Response sent' not found"
    log_message_res, log_kwargs_res = response_log_call[0], response_log_call[1]

    assert mock_logger.info.call_count == 2

    assert "Response sent" in log_message_res[0]
    assert "GET" in log_message_res[0]
    assert "/test" in log_message_res[0]
    assert "200" in log_message_res[0]

    extra_context_res = log_kwargs_res.get("extra", {})
    assert extra_context_res.get("status_code") == 200
    assert "process_time_ms" in extra_context_res
    assert "request_id" in extra_context_res

    assert "X-Process-Time" in response.headers
    assert float(response.headers["X-Process-Time"]) >= 0

    mock_logger.debug.assert_not_called()


@patch("src.middleware.logging_middleware.logger")
def test_logging_middleware_error_logging(mock_logger, app_with_middleware):
    """
    Test that LoggingMiddleware logs errors.
    """
    client = TestClient(app_with_middleware)

    with pytest.raises(ValueError, match="Test error"):
        client.get("/error")

    request_log_call = next(
        (c for c in mock_logger.info.call_args_list if "Request received" in c[0][0]),
        None,
    )
    assert request_log_call is not None, "Info log for 'Request received' not found"

    mock_logger.error.assert_called_once()
    log_message, log_kwargs = (
        mock_logger.error.call_args[0],
        mock_logger.error.call_args[1],
    )

    assert "Request failed" in log_message[0]
    assert "GET" in log_message[0]
    assert "/error" in log_message[0]

    extra_context = log_kwargs.get("extra", {})
    assert extra_context.get("error") == "Test error"
    assert "process_time_ms" in extra_context
    assert "request_id" in extra_context

    assert log_kwargs.get("exc_info") is True

    response_log_call = next(
        (c for c in mock_logger.info.call_args_list if "Response sent" in c[0][0]),
        None,
    )
    assert (
        response_log_call is None
    ), "Info log for 'Response sent' should not be present on error"


class TimeProvider:
    def __init__(self, start_time=10.0, increment=0.5):
        self.current_time = start_time
        self.increment = increment
        self.call_count = 0

    def get_time(self):
        self.call_count += 1
        t = self.current_time
        self.current_time += self.increment
        return t


@patch("time.time")
@patch("src.middleware.logging_middleware.logger")
def test_logging_middleware_timing(mock_logger, mock_time, app_with_middleware):
    """
    Test that LoggingMiddleware properly records processing time using a function side_effect.
    """
    client = TestClient(app_with_middleware)

    time_provider = TimeProvider(start_time=10.0, increment=0.5)
    mock_time.side_effect = time_provider.get_time

    response = client.get("/test")

    assert response.status_code == 200
    assert "X-Process-Time" in response.headers
    assert response.headers["X-Process-Time"] == "0.5000"

    assert mock_logger.info.call_count == 2

    response_log_call = next(
        (c for c in mock_logger.info.call_args_list if "Response sent" in c[0][0]),
        None,
    )
    assert response_log_call is not None, "Info log for 'Response sent' not found"

    extra_args = response_log_call[1].get("extra", {})
    assert "process_time_ms" in extra_args
    assert extra_args["process_time_ms"] == 500.0

    assert mock_time.call_count >= 2
