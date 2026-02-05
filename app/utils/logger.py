"""
Structured Logging Module
=========================

Provides structured logging using structlog with JSON output for production
and colored console output for development.

Usage:
    from app.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing request", user_id=123, action="tap")
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from app.config import get_settings


def add_app_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Add application context to log entries.

    Args:
        logger: The wrapped logger object.
        method_name: The name of the log method called.
        event_dict: The event dictionary to process.

    Returns:
        The modified event dictionary.
    """
    event_dict["app"] = "android-ai-agent"
    event_dict["version"] = "1.0.0"
    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging for the application.

    Sets up structlog with appropriate processors based on the environment:
    - Development: Colored console output with pretty printing
    - Production: JSON output for log aggregation

    This should be called once at application startup.
    """
    settings = get_settings()
    is_debug = settings.server.debug

    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        add_app_context,
    ]

    if is_debug:
        # Development: Pretty console output
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            ),
        ]
    else:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.server.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging for third-party libs
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.server.log_level),
    )

    # Reduce noise from common libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: The name for the logger, typically __name__.

    Returns:
        A bound structlog logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("User logged in", user_id=123)
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding temporary context to logs.

    Usage:
        with LogContext(request_id="abc123", user="john"):
            logger.info("Processing")  # Will include request_id and user
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize log context.

        Args:
            **kwargs: Key-value pairs to add to log context.
        """
        self.context = kwargs
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        """Enter the context, binding variables."""
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context, unbinding variables."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())
