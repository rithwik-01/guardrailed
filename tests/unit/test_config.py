import logging
import os
from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import ValidationError as PydanticValidationError
from pydantic import ValidationInfo

from src.core.config import (
    AppConfig,
    Environment,
    ValidationConfig,
    load_config,
)


@pytest.fixture(autouse=True)
def clear_env_before_test():
    """Clears environment variables before each test and restores after."""
    original_environ = os.environ.copy()
    os.environ.clear()
    yield
    os.environ.clear()
    os.environ.update(original_environ)


@pytest.fixture
def mock_env_vars():
    """Context manager to temporarily set environment variables."""
    env_vars_set = {}

    def _set_env(key, value):
        if value is None:
            if key in os.environ:
                del os.environ[key]
            if key in env_vars_set:
                del env_vars_set[key]
        else:
            os.environ[key] = str(value)
            env_vars_set[key] = value

    yield _set_env


def test_validation_config_check_overlap_valid():
    mock_info = MagicMock(spec=ValidationInfo)
    mock_info.data = {"max_chunk_chars": 1000}
    result = ValidationConfig.check_overlap(200, mock_info)
    assert result == 200


def test_validation_config_check_overlap_too_large(caplog):
    max_chars = 1000
    overlap = 1100
    expected_new_overlap = max_chars // 4  # 250
    mock_info = MagicMock(spec=ValidationInfo)
    mock_info.data = {"max_chunk_chars": max_chars}
    with caplog.at_level(logging.WARNING):
        result = ValidationConfig.check_overlap(overlap, mock_info)
    assert result == expected_new_overlap
    assert (
        f"Chunk overlap ({overlap}) >= max chars ({max_chars}). Adjusting to {expected_new_overlap}"
        in caplog.text
    )


def test_validation_config_check_overlap_negative(caplog):
    overlap = -50
    mock_info = MagicMock(spec=ValidationInfo)
    mock_info.data = {"max_chunk_chars": 1000}
    with caplog.at_level(logging.WARNING):
        result = ValidationConfig.check_overlap(overlap, mock_info)
    assert result == 0
    assert (
        f"Chunk overlap ({overlap}) cannot be negative. Adjusting to 0." in caplog.text
    )


def test_app_config_check_empty_url():
    assert (
        AppConfig.check_empty_url("http://example.com/model")
        == "http://example.com/model"
    )
    assert AppConfig.check_empty_url("some/path") == "some/path"
    assert AppConfig.check_empty_url("") is None
    assert AppConfig.check_empty_url("   ") is None
    assert AppConfig.check_empty_url(None) is None


def test_app_config_ensure_valid_url_string_non_string(caplog):
    mock_info_oai = MagicMock(spec=ValidationInfo, field_name="openai_api_base_url")
    with caplog.at_level(logging.WARNING):
        res_oai = AppConfig.ensure_valid_url_string(None, mock_info_oai)
    assert res_oai == "https://api.openai.com/v1"
    assert "Invalid type for openai_api_base_url" in caplog.text
    caplog.clear()

    mock_info_gem = MagicMock(spec=ValidationInfo, field_name="gemini_api_base_url")
    with caplog.at_level(logging.WARNING):
        res_gem = AppConfig.ensure_valid_url_string(123, mock_info_gem)
    assert res_gem == "https://generativelanguage.googleapis.com"
    assert "Invalid type for gemini_api_base_url" in caplog.text
    caplog.clear()

    mock_info_cla = MagicMock(spec=ValidationInfo, field_name="claude_api_base_url")
    with caplog.at_level(logging.WARNING):
        res_cla = AppConfig.ensure_valid_url_string(True, mock_info_cla)
    assert res_cla == "https://api.anthropic.com"
    assert "Invalid type for claude_api_base_url" in caplog.text


def test_app_config_ensure_valid_url_string_missing_scheme(caplog):
    mock_info_oai = MagicMock(spec=ValidationInfo, field_name="openai_api_base_url")
    with caplog.at_level(logging.WARNING):
        res_oai = AppConfig.ensure_valid_url_string("api.example.com/v1", mock_info_oai)
    assert res_oai == "https://api.example.com/v1"
    assert "missing scheme (http/https). Prepending https://" in caplog.text
    caplog.clear()

    mock_info_gem = MagicMock(spec=ValidationInfo, field_name="gemini_api_base_url")
    with caplog.at_level(logging.WARNING):
        res_gem = AppConfig.ensure_valid_url_string("google.com", mock_info_gem)
    assert res_gem == "https://google.com"
    assert "missing scheme (http/https). Prepending https://" in caplog.text


def test_app_config_instantiation_strips_trailing_slash():
    """Verify AppConfig instantiation correctly applies strip_trailing_slash validator."""
    base_args = {"policies_file_path": "dummy.yaml"}

    cfg_with_slashes = AppConfig(
        **base_args,
        openai_api_base_url="https://api.openai.com/v1/",
        gemini_api_base_url="https://gemini.example.com/",
        claude_api_base_url="https://claude.example.com/",
    )
    assert str(cfg_with_slashes.openai_api_base_url) == "https://api.openai.com/v1"
    assert str(cfg_with_slashes.gemini_api_base_url) == "https://gemini.example.com"
    assert str(cfg_with_slashes.claude_api_base_url) == "https://claude.example.com"

    cfg_without_slashes = AppConfig(
        **base_args,
        openai_api_base_url="https://api.openai.com/v1",
        gemini_api_base_url="https://gemini.example.com",
        claude_api_base_url="https://claude.example.com",
    )
    assert str(cfg_without_slashes.openai_api_base_url) == "https://api.openai.com/v1"
    assert str(cfg_without_slashes.gemini_api_base_url) == "https://gemini.example.com"
    assert str(cfg_without_slashes.claude_api_base_url) == "https://claude.example.com"

    cfg_path = AppConfig(
        **base_args, gemini_api_base_url="https://gemini.example.com/some/path"
    )
    assert str(cfg_path.gemini_api_base_url) == "https://gemini.example.com/some/path"

    cfg_path_slash = AppConfig(
        **base_args, gemini_api_base_url="https://gemini.example.com/some/path/"
    )
    assert (
        str(cfg_path_slash.gemini_api_base_url)
        == "https://gemini.example.com/some/path"
    )


@patch("src.core.config.Path")
@patch("src.core.config.load_dotenv")
@patch("src.core.config.logger")
def test_load_config_defaults(
    mock_logger, mock_load_dotenv, mock_path_cls, mock_env_vars
):
    """Test loading config with default values by directly calling load_config."""
    mock_path_instance = mock_path_cls.return_value
    mock_path_instance.exists.return_value = False

    cfg = load_config()

    assert cfg.environment == Environment.PRODUCTION
    assert cfg.port == 8000
    assert cfg.allowed_origins == {"*"}
    assert cfg.gemini_api_version == "v1beta"
    assert cfg.logging.log_dir == "logs"
    assert str(cfg.openai_api_base_url) == "https://api.openai.com/v1"
    assert str(cfg.gemini_api_base_url) == "https://generativelanguage.googleapis.com/"
    assert str(cfg.claude_api_base_url) == "https://api.anthropic.com/"

    mock_load_dotenv.assert_called_once_with(".env")
    mock_path_cls.assert_called_once_with(".env.production")
    mock_path_instance.exists.assert_called_once()
    mock_logger.debug.assert_any_call(
        f"Effective path timeouts: {cfg.middleware.timeout.path_timeouts}"
    )


@patch("src.core.config.Path")
@patch("src.core.config.load_dotenv")
@patch("src.core.config.logger")
def test_load_config_env_vars(
    mock_logger, mock_load_dotenv, mock_path_cls, mock_env_vars
):
    """Test loading config with env vars overriding defaults by calling load_config."""
    mock_env_vars("ENVIRONMENT", "development")
    mock_env_vars("PORT", "9999")
    mock_env_vars("HOST", "127.0.0.1")
    mock_env_vars("POLICIES_FILE_PATH", "env_policies.yml")
    mock_env_vars("TOXICITY_MODEL_URL", "env/model")
    mock_env_vars("NER_MODEL_URL", "")
    mock_env_vars("GEMINI_API_VERSION", "v1")
    mock_env_vars("CLAUDE_API_VERSION", "beta-v2")
    mock_env_vars("OPENAI_API_BASE_URL", "https://proxy.example.com/v1/")
    mock_env_vars("GEMINI_API_BASE_URL", "http://localhost:8080/")
    mock_env_vars("CLAUDE_API_BASE_URL", "claude.internal/api/")
    mock_env_vars("DEFAULT_TIMEOUT", "45")
    mock_env_vars(
        "PATH_TIMEOUTS",
        " /slow:100, /fast:10 , /v1/chat/completions:150, invalid-entry ",
    )
    mock_env_vars(
        "ALLOWED_ORIGINS", " http://app.example.com, https://api.example.com "
    )
    mock_env_vars("LOG_DIR", "/var/log/app")
    mock_env_vars("LOG_LEVEL", "10")
    mock_env_vars("ENABLE_CHUNK_VALIDATION", "false")
    mock_env_vars("MAX_CHUNK_CHARS", "500")
    mock_env_vars("CHUNK_OVERLAP_CHARS", "50")
    mock_env_vars("TIMEOUT_ENABLED", "0")
    mock_env_vars("SECURITY_ENABLED", "no")
    mock_env_vars("REQUEST_ID_ENABLED", "False")
    mock_env_vars("REQUEST_LOGGING_ENABLED", "f")

    mock_path_instance = mock_path_cls.return_value
    mock_path_instance.exists.return_value = True

    cfg = load_config()

    assert cfg.environment == Environment.DEVELOPMENT
    assert cfg.port == 9999
    assert cfg.host == "127.0.0.1"
    assert cfg.policies_file_path == "env_policies.yml"
    assert cfg.toxicity_model_url == "env/model"
    assert cfg.ner_model_url is None
    assert cfg.gemini_api_version == "v1"
    assert cfg.claude_api_version == "beta-v2"
    assert str(cfg.openai_api_base_url) == "https://proxy.example.com/v1"
    assert str(cfg.gemini_api_base_url) == "http://localhost:8080"
    assert str(cfg.claude_api_base_url) == "https://claude.internal/api"
    assert cfg.allowed_origins == {"http://app.example.com", "https://api.example.com"}
    assert cfg.middleware.timeout.default_timeout == 45
    assert cfg.logging.log_dir == "/var/log/app"
    assert cfg.logging.level == 10
    assert cfg.validation.enable_chunking is False
    assert cfg.validation.max_chunk_chars == 500
    assert cfg.validation.chunk_overlap_chars == 50
    assert cfg.middleware.timeout.enabled is False
    assert cfg.middleware.security.enabled is False
    assert cfg.middleware.request_id.get("enabled") is False
    assert cfg.middleware.logging.get("enabled") is False

    mock_logger.warning.assert_any_call(
        "Ignoring invalid PATH_TIMEOUTS item (missing ':'): 'invalid-entry'"
    )
    mock_logger.warning.assert_any_call(
        "claude_api_base_url 'claude.internal/api/' missing scheme (http/https). Prepending https://"
    )
    mock_load_dotenv.assert_has_calls(
        [call(".env"), call(".env.development", override=True)]
    )
    mock_path_cls.assert_called_once_with(".env.development")
    mock_path_instance.exists.assert_called_once()
    mock_logger.debug.assert_any_call(
        f"Effective path timeouts: {cfg.middleware.timeout.path_timeouts}"
    )


@patch("src.core.config.AppConfig")
@patch("src.core.config.logger")
@patch("src.core.config.load_dotenv")
@patch("src.core.config.Path")
def test_load_config_validation_error(
    mock_path_cls, mock_load_dotenv, mock_logger, mock_appconfig_cls, mock_env_vars
):
    """Test SystemExit on AppConfig validation failure when calling load_config."""
    mock_env_vars("ENVIRONMENT", "production")
    mock_path_instance = mock_path_cls.return_value
    mock_path_instance.exists.return_value = False

    validation_exception = PydanticValidationError.from_exception_data(
        title="AppConfig",
        line_errors=[
            {
                "input": "not-an-int",
                "loc": ("port",),
                "msg": "Input should be a valid integer",
                "type": "int_parsing",
            }
        ],
    )
    mock_appconfig_cls.side_effect = validation_exception

    with pytest.raises(SystemExit) as exc_info:
        load_config()

    assert "Configuration validation failed" in str(exc_info.value)
    assert "port" in str(exc_info.value)
    assert "int_parsing" in str(exc_info.value)

    mock_logger.critical.assert_called_once()
    critical_log_message = mock_logger.critical.call_args[0][0]
    assert "Configuration validation failed" in critical_log_message
    assert "port" in critical_log_message
    assert "int_parsing" in critical_log_message

    mock_load_dotenv.assert_called_once_with(".env")
    mock_path_cls.assert_called_once_with(".env.production")
    mock_path_instance.exists.assert_called_once()

    mock_appconfig_cls.assert_called_once()
