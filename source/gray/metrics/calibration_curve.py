"""Binary probability calibration curve and calibration errors."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._binary import binary_inputs


def calibration_curve(targets: Sequence[Any], probabilities: Sequence[float] | np.ndarray, positive_label: Any | None = None, n_bins: int = 10) -> dict[str, Any]:
    """Return reliability-curve points, expected calibration error and Brier score."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y_true, values, positive = binary_inputs(targets, probabilities, positive_label)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    binary_target = (y_true == positive).astype(int)
    bin_index = np.minimum((values * n_bins).astype(int), n_bins - 1)
    points: list[dict[str, float | int]] = []
    expected_calibration_error = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        count = int(mask.sum())
        if not count:
            continue
        observed = float(binary_target[mask].mean())
        predicted = float(values[mask].mean())
        expected_calibration_error += count / len(values) * abs(observed - predicted)
        points.append({"lower": index / n_bins, "upper": (index + 1) / n_bins, "count": count, "mean_prediction": predicted, "positive_rate": observed})
    return {
        "positive_label": str(positive),
        "n_bins": n_bins,
        "points": points,
        "ece": float(expected_calibration_error),
        "brier_score": float(np.mean((values - binary_target) ** 2)),
    }
