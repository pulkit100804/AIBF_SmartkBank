"""
Logging configuration for SmartBank AI.

Provides a centralized logger that respects LOG_LEVEL from environment.
Avoids logging PII or sensitive banking data.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    from app.config.settings import LOG_LEVEL

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger
