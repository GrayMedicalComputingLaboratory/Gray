"""Consistent console and file logger construction."""
from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def get_logger(name: str, output_dir: Path) -> logging.Logger:
    """Create or reuse an isolated console and file logger.

    Repeated calls with the same name and resolved directory reuse the logger
    without adding duplicate handlers. Log records are written to
    ``<output_dir>/run.log`` and displayed through Rich.

    Args:
        name: Logical logger name.
        output_dir: Directory in which to create ``run.log``.

    Returns:
        A non-propagating logger configured at ``INFO`` level.

    Raises:
        OSError: If the output directory or log file cannot be created.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}:{output_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        console_handler = RichHandler(show_path=False, rich_tracebacks=True, markup=True)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger
