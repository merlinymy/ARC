"""Comprehensive logging configuration for the backend API.

Provides structured logging with:
- Request/response logging
- User operation tracking
- Error context
- File rotation
- Configurable log levels
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName", "relativeCreated",
                "thread", "threadName", "exc_info", "exc_text", "stack_info"
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        # Format the message
        formatted = super().format(record)

        # Reset levelname for next use
        record.levelname = levelname

        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_json: bool = False,
    enable_file: bool = True,
) -> None:
    """Setup comprehensive logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_json: Use JSON formatting for logs
        enable_file: Enable file logging
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if enable_json:
        console_formatter = JSONFormatter()
    else:
        console_formatter = ColoredFormatter(
            '%(levelname)s | %(asctime)s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (if enabled)
    if enable_file:
        # General log file (INFO and above)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(levelname)s | %(asctime)s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Error log file (ERROR and above only)
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "error.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

        # API access log file (for request/response tracking)
        access_handler = logging.handlers.RotatingFileHandler(
            log_path / "access.log",
            maxBytes=20 * 1024 * 1024,  # 20 MB
            backupCount=10,
            encoding='utf-8'
        )
        access_handler.setLevel(logging.INFO)
        access_handler.setFormatter(file_formatter)

        # Create access logger
        access_logger = logging.getLogger("access")
        access_logger.addHandler(access_handler)
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False  # Don't propagate to root

    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("voyageai").setLevel(logging.WARNING)
    logging.getLogger("cohere").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.info(f"Logging initialized: level={log_level}, file_logging={enable_file}, json={enable_json}")


def get_access_logger() -> logging.Logger:
    """Get the access logger for API request/response logging."""
    return logging.getLogger("access")


class RequestLogger:
    """Helper for logging API requests with context."""

    def __init__(self, request_id: str, user_id: str | None = None):
        self.request_id = request_id
        self.user_id = user_id
        self.logger = get_access_logger()

    def log_request(
        self,
        method: str,
        path: str,
        query_params: Dict[str, Any] | None = None,
        client_ip: str | None = None,
    ) -> None:
        """Log an incoming request."""
        extra = {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "client_ip": client_ip,
            "query_params": query_params,
        }
        self.logger.info(
            f"REQUEST | {method} {path}",
            extra=extra
        )

    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Log a response."""
        extra = {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }

        # Log level based on status code
        if status_code >= 500:
            level = logging.ERROR
        elif status_code >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO

        self.logger.log(
            level,
            f"RESPONSE | {method} {path} | {status_code} | {duration_ms:.2f}ms",
            extra=extra
        )

    def log_error(
        self,
        method: str,
        path: str,
        error: Exception,
        status_code: int = 500,
    ) -> None:
        """Log an error during request processing."""
        extra = {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "error_type": type(error).__name__,
            "status_code": status_code,
        }
        self.logger.error(
            f"ERROR | {method} {path} | {error}",
            extra=extra,
            exc_info=True
        )

    def log_user_operation(
        self,
        operation: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Log a user operation (upload, delete, etc)."""
        extra = {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "operation": operation,
            "details": details,
        }
        self.logger.info(
            f"USER_OP | {operation} | user={self.user_id}",
            extra=extra
        )
