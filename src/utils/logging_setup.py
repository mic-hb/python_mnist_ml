"""Centralized logging configuration for the MNIST CLI project."""

import logging
from typing import Optional


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Setup basic logging configuration for the application.

    Configures a logger to output to console with a standardized format
    including timestamp, log level, and message.

    Args:
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
                        Defaults to logging.INFO.
    """
    # Define the log format with timestamp, level, and message
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],  # Output to console
    )

    # Get the root logger and log the setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log level set to: {logging.getLevelName(log_level)}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (Optional[str]): Name for the logger. If None, uses the calling module's name.

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
