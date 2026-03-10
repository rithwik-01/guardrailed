import contextvars
import json
import logging
import os
import sys
import uuid
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Any, Dict, List, Optional, Type, Union

from src.core.config import Environment
from src.core.state import app_state

from .constants import LOG_RECORD_STANDARD_ATTRS

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


class ContextFilter(logging.Filter):
    def __init__(self, service_name: str = "guardrailed-os-core"):
        super().__init__()
        self.service_name = service_name
        if app_state.config is None:
            raise RuntimeError("app_state.config is not initialized")
        self.environment = app_state.config.environment.value

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self.service_name
        record.environment = self.environment
        explicit_request_id = getattr(record, "request_id", None)
        context_request_id = request_id_var.get()
        final_correlation_id = (
            explicit_request_id
            or context_request_id
            or getattr(record, "correlation_id", None)
            or str(uuid.uuid4())
        )
        record.correlation_id = final_correlation_id
        return True


class JsonFormatter(logging.Formatter):
    DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

    def format(self, record: logging.LogRecord) -> str:
        correlation_id = getattr(record, "correlation_id", "unassigned-correlation-id")
        default_environment = (
            "unknown"
            if app_state.config is None
            else app_state.config.environment.value
        )

        log_object: Dict[str, Any] = {
            "timestamp": self.formatTime(
                record, self.datefmt or self.DEFAULT_TIMESTAMP_FORMAT
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": getattr(record, "service", "unknown-service"),
            "environment": getattr(record, "environment", default_environment),
            "correlation_id": correlation_id,
            "source_location": f"{record.pathname}:{record.lineno}",
        }
        if record.exc_info:
            log_object["exception_traceback"] = self.formatException(record.exc_info)
            exception_type: Optional[Type[BaseException]] = record.exc_info[0]
            exception_value: Optional[BaseException] = record.exc_info[1]
            if exception_type:
                log_object["exception_type"] = exception_type.__name__
            if exception_value:
                log_object["exception_message"] = str(exception_value)

        extra_attrs = {
            k: v
            for k, v in record.__dict__.items()
            if k not in LOG_RECORD_STANDARD_ATTRS and not k.startswith("_")
        }
        if extra_attrs:
            log_object["extra"] = extra_attrs
        try:
            return json.dumps(log_object, default=str)
        except TypeError as e:
            error_log = {
                "timestamp": self.formatTime(
                    record, self.datefmt or self.DEFAULT_TIMESTAMP_FORMAT
                ),
                "level": "ERROR",
                "logger": "logging.JsonFormatter",
                "message": f"Failed to serialize log record to JSON: {e}",
                "original_level": record.levelname,
                "original_logger": record.name,
                "correlation_id": correlation_id,
            }
            return json.dumps(error_log)


class ConsoleFormatter(logging.Formatter):
    DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(
            record, self.datefmt or self.DEFAULT_TIMESTAMP_FORMAT
        )
        correlation_id = getattr(record, "correlation_id", "no-cid")
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in LOG_RECORD_STANDARD_ATTRS and not k.startswith("_")
        }
        extras_str = ""
        if extras:
            extras_str = " | " + " ".join([f"{k}={v!r}" for k, v in extras.items()])
        log_message = record.getMessage()
        location = f"{record.module}:{record.lineno}"
        formatted_line = f"{timestamp} - {record.levelname:<8} - [{correlation_id}] - {location} - {log_message}{extras_str}"
        if record.exc_info:
            formatted_line += "\n" + self.formatException(record.exc_info)
        return formatted_line


def setup_logger(level: Optional[Union[int, str]] = None) -> logging.Logger:
    if app_state.config is None:
        raise RuntimeError(
            "app_state.config is not initialized. Cannot setup logger without configuration."
        )

    log_level: int
    if level is None:
        log_level = app_state.config.logging.level
    elif isinstance(level, str):
        log_level_num = logging.getLevelName(level.upper())
        if not isinstance(log_level_num, int):
            print(
                f"Warning: Invalid log level string '{level}'. Using default level {app_state.config.logging.level}."
            )
            log_level = app_state.config.logging.level
        else:
            log_level = log_level_num
    else:
        log_level = level

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        print(
            f"Clearing {len(root_logger.handlers)} existing handlers for root logger."
        )
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    root_logger.setLevel(log_level)
    context_filter = ContextFilter()
    root_logger.addFilter(context_filter)
    print("ContextFilter added to root logger.")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    console_formatter: logging.Formatter
    if app_state.config.environment == Environment.DEVELOPMENT:
        console_formatter = ConsoleFormatter()
    else:
        console_formatter = JsonFormatter(
            datefmt=JsonFormatter.DEFAULT_TIMESTAMP_FORMAT
        )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    error_console_handler = logging.StreamHandler(sys.stderr)
    error_console_handler.setLevel(logging.ERROR)
    error_console_handler.setFormatter(console_formatter)
    root_logger.addHandler(error_console_handler)

    if app_state.config.logging.log_dir:
        try:
            log_dir = app_state.config.logging.log_dir
            os.makedirs(log_dir, exist_ok=True)

            json_file_formatter = JsonFormatter(
                datefmt=JsonFormatter.DEFAULT_TIMESTAMP_FORMAT
            )

            app_log_file = os.path.join(log_dir, "app.log")
            file_handler = RotatingFileHandler(
                app_log_file,
                maxBytes=app_state.config.logging.log_max_bytes,
                backupCount=app_state.config.logging.log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(json_file_formatter)
            root_logger.addHandler(file_handler)

            error_log_file = os.path.join(log_dir, "error.log")
            error_file_handler = TimedRotatingFileHandler(
                error_log_file,
                when="midnight",
                interval=1,
                backupCount=app_state.config.logging.log_backup_count,
                encoding="utf-8",
            )
            error_file_handler.setLevel(logging.ERROR)
            error_file_handler.setFormatter(json_file_formatter)
            root_logger.addHandler(error_file_handler)

            print(f"File logging enabled. Log directory: {log_dir}")
        except OSError as e:
            print(
                f"Error setting up file logging in {app_state.config.logging.log_dir}: {e}",
                file=sys.stderr,
            )
    else:
        print("File logging disabled (LOG_DIR not set).")

    loggers_to_silence: List[str] = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "httpcore",
        "httpx",
        "aiohttp",
        "asyncio",
        "urllib3",
        "huggingface_hub",
        "filelock",
        "transformers",
    ]
    silence_level = logging.WARNING
    print(
        f"Setting log level for {len(loggers_to_silence)} third-party loggers to {logging.getLevelName(silence_level)}"
    )
    for logger_name in loggers_to_silence:
        logging.getLogger(logger_name).setLevel(silence_level)

    print(
        f"Root logger configured with level: {logging.getLevelName(root_logger.level)}"
    )
    root_logger.debug("Logger setup complete. This is a debug message.")
    root_logger.info("Logger setup complete. This is an info message.")
    return root_logger
