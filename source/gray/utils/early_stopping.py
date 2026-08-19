"""Metric-based early stopping for iterative training workflows."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal


class EarlyStopping:
    """Stop training after a monitored metric stops improving.

    The class is framework-independent: call :meth:`step` once after each
    validation epoch and stop when it returns ``True``.

    Args:
        patience: Number of consecutive non-improving steps tolerated after the
            best value. ``0`` stops on the first non-improving step.
        mode: ``"min"`` when lower values are better, or ``"max"`` when higher
            values are better.
        min_delta: Minimum absolute improvement required to reset patience.
            Must be non-negative.

    Raises:
        ValueError: If ``patience`` is negative, ``mode`` is unsupported, or
            ``min_delta`` is negative.
        TypeError: If constructor arguments have invalid types.
    """

    def __init__(
        self,
        patience: int = 5,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise TypeError("patience must be an integer")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)):
            raise TypeError("min_delta must be a number")
        if not math.isfinite(float(min_delta)) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        self.patience = patience
        self.mode = mode
        self.min_delta = float(min_delta)
        self.reset()

    def reset(self) -> None:
        """Reset the best value, wait counter, and stopped state.

        Returns:
            None.
        """
        self.best_value: float | None = None
        self.best_step: int | None = None
        self.num_bad_steps = 0
        self.stopped = False

    def step(self, value: float) -> bool:
        """Record one metric value and report whether training should stop.

        Args:
            value: Finite validation metric for the current step.

        Returns:
            ``True`` once patience is exhausted; otherwise ``False``. Once
            stopped, subsequent calls continue returning ``True``.

        Raises:
            TypeError: If ``value`` is not numeric.
            ValueError: If ``value`` is NaN or infinite.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("value must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if self.stopped:
            return True

        current_step = 0 if self.best_step is None else self.best_step + self.num_bad_steps + 1
        if self._is_improvement(value):
            self.best_value = value
            self.best_step = current_step
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1
            if self.num_bad_steps > self.patience:
                self.stopped = True
        return self.stopped

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing or experiment resumes.

        Returns:
            Dictionary containing configuration and current stopping state.
        """
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_step": self.best_step,
            "num_bad_steps": self.num_bad_steps,
            "stopped": self.stopped,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state previously returned by :meth:`state_dict`.

        Args:
            state: Mapping containing the serialized stopping state.

        Returns:
            None.

        Raises:
            TypeError: If ``state`` is not a mapping.
            ValueError: If restored counters or metric values are invalid.
            KeyError: If required state keys are missing.
        """
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("patience") != self.patience or state.get("mode") != self.mode:
            raise ValueError("state configuration does not match this EarlyStopping instance")
        best_value = state["best_value"]
        if best_value is not None:
            if not isinstance(best_value, (int, float)) or not math.isfinite(float(best_value)):
                raise ValueError("state best_value must be finite or None")
            best_value = float(best_value)
        best_step = state["best_step"]
        bad_steps = state["num_bad_steps"]
        if best_step is not None and (isinstance(best_step, bool) or not isinstance(best_step, int) or best_step < 0):
            raise ValueError("state best_step must be a non-negative integer or None")
        if isinstance(bad_steps, bool) or not isinstance(bad_steps, int) or bad_steps < 0:
            raise ValueError("state num_bad_steps must be a non-negative integer")
        self.best_value = best_value
        self.best_step = best_step
        self.num_bad_steps = bad_steps
        self.stopped = bool(state["stopped"])

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta
