import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

from dotenv import load_dotenv
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    HttpUrl,
    ValidationInfo,
    field_validator,
)

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class LogConfig(BaseModel):
    level: int = 20
    log_dir: Optional[str] = "logs"
    log_max_bytes: int = 10_000_000
    log_backup_count: int = 30


class SecurityConfig(BaseModel):
    enabled: bool = True
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }
    )


class TimeoutConfig(BaseModel):
    enabled: bool = True
    default_timeout: int = 30
    path_timeouts: Dict[str, int] = Field(
        default_factory=lambda: {
            "/safeguard": 60,
            "/v1/chat/completions": 120,
            f"/{os.getenv('GEMINI_API_VERSION', 'v1beta')}/models/*": 120,
            "/anthropic/v1/messages": 120,
        }
    )


class MiddlewareConfig(BaseModel):
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    request_id: Dict[str, bool] = Field(default_factory=lambda: {"enabled": True})
    logging: Dict[str, bool] = Field(default_factory=lambda: {"enabled": True})
    security: SecurityConfig = Field(default_factory=SecurityConfig)


class ValidationConfig(BaseModel):
    enable_chunking: bool = True
    max_chunk_chars: int = 1800
    chunk_overlap_chars: int = 200

    @field_validator("chunk_overlap_chars")
    @classmethod
    def check_overlap(cls, v: int, info: ValidationInfo) -> int:
        max_chars = info.data.get("max_chunk_chars")
        if max_chars is not None and v >= max_chars:
            new_overlap: int = max(0, max_chars // 4)
            logger.warning(
                f"Chunk overlap ({v}) >= max chars ({max_chars}). Adjusting to {new_overlap}"
            )
            return new_overlap
        elif v < 0:
            logger.warning(f"Chunk overlap ({v}) cannot be negative. Adjusting to 0.")
            return 0
        return v


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    environment: Environment = Environment.PRODUCTION
    policies_file_path: str = "policies.yaml"
    toxicity_model_url: Optional[str] = "s-nlp/roberta_toxicity_classifier"
    ner_model_url: Optional[str] = "dslim/bert-base-NER"
    middleware: MiddlewareConfig = Field(default_factory=MiddlewareConfig)
    logging: LogConfig = Field(default_factory=LogConfig)
    allowed_origins: Set[str] = Field(default_factory=lambda: {"*"})
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    openai_api_base_url: HttpUrl = Field(default=HttpUrl("https://api.openai.com/v1"))
    gemini_api_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://generativelanguage.googleapis.com")
    )
    gemini_api_version: str = "v1beta"
    claude_api_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://api.anthropic.com")
    )
    claude_api_version: str = "2023-06-01"

    @field_validator("toxicity_model_url", "ner_model_url")
    @classmethod
    def check_empty_url(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator(
        "openai_api_base_url",
        "gemini_api_base_url",
        "claude_api_base_url",
        mode="before",
    )
    @classmethod
    def ensure_valid_url_string(cls, v: Any, info: ValidationInfo) -> str:
        """Ensure the input is a string and has a scheme before Pydantic validation."""
        field_name = info.field_name if info.field_name else "URL field"
        default_url = ""
        if "openai" in field_name:
            default_url = "https://api.openai.com/v1"
        elif "gemini" in field_name:
            default_url = "https://generativelanguage.googleapis.com"
        elif "claude" in field_name:
            default_url = "https://api.anthropic.com"
        else:
            default_url = "https://example.com"

        if not isinstance(v, str):
            logger.warning(
                f"Invalid type for {field_name}: {type(v)}. Using default {default_url}."
            )
            return default_url

        if not v.startswith(("http://", "https://")):
            logger.warning(
                f"{field_name} '{v}' missing scheme (http/https). Prepending https://"
            )
            return f"https://{v}"
        return v

    @field_validator(
        "openai_api_base_url", "gemini_api_base_url", "claude_api_base_url"
    )
    @classmethod
    def strip_trailing_slash(cls, v: Union[HttpUrl, AnyHttpUrl]) -> str:
        """
        Remove trailing slash from the URL string after initial validation
        and return the string for Pydantic to perform final coercion.
        """
        return str(v).rstrip("/")


def load_config() -> AppConfig:
    load_dotenv(".env")
    env = os.getenv("ENVIRONMENT", "production")
    env_file = f".env.{env}"
    if Path(env_file).exists():
        logger.info(f"Loading environment variables from {env_file}")
        load_dotenv(env_file, override=True)
    else:
        logger.info(
            f"Environment file {env_file} not found, using defaults and OS env vars."
        )

    gemini_version = os.getenv("GEMINI_API_VERSION", "v1beta")
    claude_version = os.getenv("CLAUDE_API_VERSION", "2023-06-01")

    default_path_timeouts = {
        "/safeguard": 60,
        "/v1/chat/completions": 120,
        f"/{gemini_version}/models/*": 120,
        "/anthropic/v1/messages": 120,
    }
    path_timeouts_str = os.getenv("PATH_TIMEOUTS")
    path_timeouts_env = {}
    if path_timeouts_str:
        for item in path_timeouts_str.split(","):
            item_stripped = item.strip()
            if ":" in item_stripped:
                try:
                    path, timeout_str = item_stripped.split(":", 1)
                    path_cleaned = path.strip()
                    if path_cleaned and not path_cleaned.startswith("/"):
                        path_cleaned = "/" + path_cleaned
                    path_timeouts_env[path_cleaned] = int(timeout_str.strip())
                except ValueError:
                    logger.warning(
                        f"Ignoring invalid PATH_TIMEOUTS item: '{item_stripped}'"
                    )
            elif item_stripped:
                logger.warning(
                    f"Ignoring invalid PATH_TIMEOUTS item (missing ':'): '{item_stripped}'"
                )

    final_path_timeouts = {**default_path_timeouts, **path_timeouts_env}

    timeout_config = TimeoutConfig(
        enabled=os.getenv("TIMEOUT_ENABLED", "True").lower() in ("true", "1", "yes"),
        default_timeout=int(os.getenv("DEFAULT_TIMEOUT", "30")),
        path_timeouts=final_path_timeouts,
    )
    logger.debug(f"Effective path timeouts: {final_path_timeouts}")

    security_config = SecurityConfig(
        enabled=os.getenv("SECURITY_ENABLED", "True").lower() in ("true", "1", "yes")
    )

    middleware_config = MiddlewareConfig(
        timeout=timeout_config,
        request_id={
            "enabled": os.getenv("REQUEST_ID_ENABLED", "True").lower()
            in ("true", "1", "yes")
        },
        logging={
            "enabled": os.getenv("REQUEST_LOGGING_ENABLED", "True").lower()
            in ("true", "1", "yes")
        },
        security=security_config,
    )

    log_dir_env = os.getenv("LOG_DIR")
    log_level_env = int(os.getenv("LOG_LEVEL", "20"))
    log_config = LogConfig(
        level=log_level_env,
        log_dir=(log_dir_env if log_dir_env is not None else "logs"),
        log_max_bytes=int(os.getenv("LOG_MAX_BYTES", "10000000")),
        log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "30")),
    )

    allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
    allowed_origins = {
        origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()
    }
    if not allowed_origins and allowed_origins_str == "*":
        allowed_origins = {"*"}

    validation_config = ValidationConfig(
        enable_chunking=os.getenv("ENABLE_CHUNK_VALIDATION", "True").lower()
        in ("true", "1", "yes"),
        max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "1800")),
        chunk_overlap_chars=int(os.getenv("CHUNK_OVERLAP_CHARS", "200")),
    )

    app_config_kwargs: Dict[str, Any] = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "8000")),
        "environment": Environment(os.getenv("ENVIRONMENT", "production")),
        "policies_file_path": os.getenv("POLICIES_FILE_PATH", "policies.yaml"),
        "toxicity_model_url": os.getenv(
            "TOXICITY_MODEL_URL", "s-nlp/roberta_toxicity_classifier"
        ),
        "ner_model_url": os.getenv("NER_MODEL_URL", "dslim/bert-base-NER"),
        "middleware": middleware_config,
        "logging": log_config,
        "allowed_origins": allowed_origins,
        "validation": validation_config,
        "gemini_api_version": gemini_version,
        "claude_api_version": claude_version,
    }

    openai_url = os.getenv("OPENAI_API_BASE_URL")
    if openai_url is not None:
        app_config_kwargs["openai_api_base_url"] = openai_url

    gemini_url = os.getenv("GEMINI_API_BASE_URL")
    if gemini_url is not None:
        app_config_kwargs["gemini_api_base_url"] = gemini_url

    claude_url = os.getenv("CLAUDE_API_BASE_URL")
    if claude_url is not None:
        app_config_kwargs["claude_api_base_url"] = claude_url

    try:
        loaded_app_config = AppConfig(**app_config_kwargs)
        logger.debug(f"Full Config: {loaded_app_config.model_dump(mode='json')}")
        return loaded_app_config
    except Exception as e:
        logger.critical(f"Configuration validation failed: {e}", exc_info=True)
        raise SystemExit(f"Configuration validation failed: {e}")
