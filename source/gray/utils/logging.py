"""Consistent console and file logger construction."""
from __future__ import annotations

import logging
import math
from numbers import Real
from pathlib import Path
from typing import Any

from rich.logging import RichHandler
from rich.text import Text


SUCCESS = 25
METRIC = 21
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(METRIC, "METRIC")


class _GrayRichHandler(RichHandler):
    """Apply semantic console styles without modifying stored messages."""

    _message_styles = {
        SUCCESS: "bold green",
        METRIC: "bold cyan",
        logging.WARNING: "yellow",
        logging.ERROR: "bold red",
        logging.CRITICAL: "bold white on red",
    }

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        rendered = super().render_message(record, message)
        style = self._message_styles.get(record.levelno)
        if style:
            rendered.stylize(style)
        return rendered


class GrayLogger(logging.LoggerAdapter):
    """Logger with semantic success and numeric metric operations."""

    def success(self, message: object, *args: object, **kwargs: Any) -> None:
        """Log a successful operation with a green emphasized console style.

        Args:
            message: Message or format string accepted by :mod:`logging`.
            *args: Values interpolated into a format string.
            **kwargs: Standard logging options such as ``exc_info`` and ``extra``.

        Returns:
            None.
        """
        self.log(SUCCESS, message, *args, **kwargs)

    def metric(self, name: str, value: Real) -> None:
        """Log one finite numeric metric using a stable ``name=value`` format.

        Args:
            name: Non-empty metric name, for example ``validation_auc``.
            value: Finite numeric metric value.

        Returns:
            None.

        Raises:
            TypeError: If ``name`` is not a string or ``value`` is not numeric.
            ValueError: If ``name`` is empty or ``value`` is not finite.
        """
        if not isinstance(name, str):
            raise TypeError("metric name must be a string")
        name = name.strip()
        if not name:
            raise ValueError("metric name must not be empty")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("metric value must be a number")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("metric value must be finite")
        self.log(METRIC, "%s=%s", name, numeric_value)


def get_logger(name: str, output_dir: Path) -> GrayLogger:
    """Create or reuse an isolated console and file logger.

    Repeated calls with the same name and resolved directory reuse the logger
    without adding duplicate handlers. Log records are written to
    ``<output_dir>/run.log`` and displayed through Rich.

    Args:
        name: Logical logger name.
        output_dir: Directory in which to create ``run.log``.

    Returns:
        A non-propagating :class:`GrayLogger` configured at ``INFO`` level.
        It provides ``success`` and ``metric`` in addition to standard logging
        methods such as ``info``, ``warning`` and ``error``.

    Raises:
        OSError: If the output directory or log file cannot be created.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}:{output_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        console_handler = _GrayRichHandler(show_path=False, rich_tracebacks=True, markup=False)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return GrayLogger(logger, {})
