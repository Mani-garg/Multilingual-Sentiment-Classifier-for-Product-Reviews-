"""Structured logging configuration for development and production observability."""

import json
import logging
import sys
from datetime import UTC, datetime


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter with ISO-8601 timestamps and request correlation."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add correlation ID if present in record extra
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms
        if hasattr(record, "backend"):
            log_obj["backend"] = record.backend
        if hasattr(record, "batch_size"):
            log_obj["batch_size"] = record.batch_size
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def configure_logging(level: str = "INFO", use_json: bool = False) -> None:
    """Configure root logger with either standard text or structured JSON output."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    if use_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt))

    root_logger.addHandler(console_handler)

    # Silence verbose 3rd party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
