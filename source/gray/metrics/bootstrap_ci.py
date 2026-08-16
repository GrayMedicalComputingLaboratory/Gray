"""Non-parametric bootstrap confidence interval for one scalar metric."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def bootstrap_ci(targets: Sequence[Any], values: Sequence[Any], metric: Callable[[np.ndarray, np.ndarray], float | None], n_bootstrap: int = 2_000, confidence: float = 0.95, seed: int = 42) -> dict[str, float | int | None]:
    """Return percentile bootstrap CI, skipping resamples where a metric is undefined."""
    y_true = np.asarray(list(targets), dtype=object)
    metric_values = np.asarray(list(values), dtype=object)
    if y_true.size == 0 or y_true.shape[0] != metric_values.shape[0]:
        raise ValueError("targets and values must be non-empty and aligned")
    if n_bootstrap < 100 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be at least 100 and confidence must be in (0, 1)")
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        index = generator.integers(0, y_true.size, size=y_true.size)
        value = metric(y_true[index], metric_values[index])
        if value is not None and np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return {"estimate": None, "lower": None, "upper": None, "confidence": confidence, "bootstrap_samples": 0}
    alpha = (1 - confidence) / 2
    point = metric(y_true, metric_values)
    return {
        "estimate": float(point) if point is not None else None,
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": len(estimates),
    }
