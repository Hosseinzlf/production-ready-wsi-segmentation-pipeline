"""Logging configuration using loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Configure loguru for the pipeline.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path for file logging (with rotation).
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=level,
            rotation="20 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        )
        logger.info(f"File logging enabled → {log_file}")
