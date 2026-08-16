"""Consistent console and file logger construction."""
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, output_dir: Path) -> logging.Logger:
    """Return a non-propagating logger writing both console and ``run.log``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger
