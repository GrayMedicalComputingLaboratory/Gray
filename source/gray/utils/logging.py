"""Consistent console and file logger construction."""
from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def get_logger(name: str, output_dir: Path) -> logging.Logger:
    """Return an isolated console/file logger for one output directory."""
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
