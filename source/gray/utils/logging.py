"""Consistent console and file logger construction."""

from __future__ import annotations

import logging
import math
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

from rich.logging import RichHandler
from rich.text import Text

SUCCESS = 25
METRIC = 21
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(METRIC, "METRIC")


class _GrayRichHandler(RichHandler):
    """Apply semantic console styles without modifying stored messages."""

    _message_styles: ClassVar[dict[int, str]] = {
        SUCCESS: "bold green",
        METRIC: "bold cyan",
        logging.WARNING: "yellow",
        logging.ERROR: "bold red",
        logging.CRITICAL: "bold white on red",
    }

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        """Render and color one console log message by semantic level.

        Args:
            record: Standard-library log record containing the numeric level.
            message: Already formatted plain-text log message.

        Returns:
            Rich ``Text`` with the configured semantic style applied.
        """
        rendered = super().render_message(record, message)
        style = self._message_styles.get(record.levelno)
        if style:
            rendered.stylize(style)
        return rendered


class GrayLogger(logging.LoggerAdapter):
    """Logger with semantic success and numeric metric operations."""

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None, *, tracker: Any = None) -> None:
        """Wrap a standard logger and optionally mirror records to a tracker.

        Args:
            logger: Configured standard-library logger.
            extra: Optional contextual fields passed to ``LoggerAdapter``.
            tracker: Optional object providing ClearML-compatible
                ``report_text`` and ``report_scalar`` methods.
        """
        super().__init__(logger, extra or {})
        self._tracker = tracker
        self._forwarded_logger_names: set[str] = set()

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        """Write locally and mirror the formatted plain-text record.

        Args:
            level: Standard-library numeric logging level.
            msg: Message object or percent-style format string.
            *args: Values interpolated into ``msg``.
            **kwargs: Standard logging keyword arguments.

        Returns:
            None.
        """
        if not self.isEnabledFor(level):
            return
        super().log(level, msg, *args, **kwargs)
        if self._tracker is not None:
            record = logging.LogRecord(
                name=self.logger.name,
                level=level,
                pathname="",
                lineno=0,
                msg=msg,
                args=args,
                exc_info=None,
            )
            self._tracker.report_text(record.getMessage(), level=level, print_console=False)

    def close(self) -> None:
        """Close and detach this logger's handlers.

        Returns:
            None. Subsequent calls are safe and have no effect.
        """
        for name in tuple(self._forwarded_logger_names):
            external = logging.getLogger(name)
            for handler in tuple(external.handlers):
                if (
                    isinstance(handler, _GrayForwardHandler)
                    and handler.target_logger is self.logger
                ):
                    external.removeHandler(handler)
                    handler.close()
            self._forwarded_logger_names.remove(name)
        for handler in tuple(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

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

    def metric(self, name: str, value: Real, *, iteration: int = 0, title: str = "Metrics") -> None:
        """Log one finite numeric metric using a stable ``name=value`` format.

        Args:
            name: Non-empty metric name, for example ``validation_auc``.
            value: Finite numeric metric value.
            iteration: Non-negative training step or epoch used by ClearML.
            title: Non-empty ClearML chart title.

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
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise TypeError("metric iteration must be an integer")
        if iteration < 0:
            raise ValueError("metric iteration must be non-negative")
        if not isinstance(title, str):
            raise TypeError("metric title must be a string")
        title = title.strip()
        if not title:
            raise ValueError("metric title must not be empty")
        self.log(METRIC, "%s=%s", name, numeric_value)
        if self._tracker is not None:
            self._tracker.report_scalar(title=title, series=name, value=numeric_value, iteration=iteration)


class _GrayForwardHandler(logging.Handler):
    """Forward standard-library records into one configured Gray logger."""

    def __init__(self, target: GrayLogger) -> None:
        """Store the Gray logger receiving forwarded records.

        Args:
            target: Configured Gray logger for console and file output.
        """
        super().__init__(level=logging.NOTSET)
        self._target = target

    @property
    def target_logger(self) -> logging.Logger:
        """Return the underlying logger used by the forwarding target."""
        return self._target.logger

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one external record without duplicating its logger prefix.

        Args:
            record: Record emitted by an external standard-library logger.
        """
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{self._format_exception(record.exc_info)}"
            self._target.log(record.levelno, message)
        except Exception:  # noqa: BLE001 - logging failures must not break callers
            self.handleError(record)

    @staticmethod
    def _format_exception(exc_info: Any) -> str:
        """Format exception details using the standard logging formatter."""
        return logging.Formatter().formatException(exc_info)


def forward_standard_logger(name: str, target: GrayLogger) -> None:
    """Route an external standard-library logger through Gray handlers.

    Existing handlers on the named logger are detached so console output is
    emitted once through the target Gray logger and stored in its ``run.log``.
    Repeated calls replace an earlier Gray forwarding handler safely.

    Args:
        name: External logger namespace, for example ``clearml``.
        target: Configured Gray logger receiving external records.

    Raises:
        ValueError: If the logger name is empty.
        TypeError: If target is not a :class:`GrayLogger`.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("external logger name must be non-empty")
    if not isinstance(target, GrayLogger):
        raise TypeError("external logger target must be a GrayLogger")
    logger_name = name.strip()
    external = logging.getLogger(logger_name)
    logger_names = [
        logger_name,
        *[
            child_name
            for child_name in logging.Logger.manager.loggerDict
            if child_name.startswith(f"{logger_name}.")
        ],
    ]
    for child_name in logger_names:
        child = logging.getLogger(child_name)
        for handler in tuple(child.handlers):
            child.removeHandler(handler)
            handler.close()
        child.setLevel(logging.NOTSET if child_name != logger_name else logging.INFO)
        child.propagate = child_name != logger_name
    external.addHandler(_GrayForwardHandler(target))
    external.setLevel(logging.INFO)
    external.propagate = False
    target._forwarded_logger_names.add(logger_name)


def get_logger(name: str, output_dir: Path, *, tracker: Any = None) -> GrayLogger:
    """Create or reuse an isolated console and file logger.

    Repeated calls with the same name and resolved directory reuse the logger
    without adding duplicate handlers. Log records are written to
    ``<output_dir>/run.log`` and displayed through Rich.

    Args:
        name: Logical logger name.
        output_dir: Directory in which to create ``run.log``.
        tracker: Optional ClearML Logger-compatible object. When provided, all
            messages are mirrored with ``report_text`` and metrics are also
            reported as scalar series.

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
    return GrayLogger(logger, tracker=tracker)
