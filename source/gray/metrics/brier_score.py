"""Binary Brier calibration score."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def brier_score(targets: Sequence[Any], scores: Sequence[float] | Sequence[Sequence[float]] | np.ndarray, labels: Sequence[Any] | None = None) -> float | None:
    """Return binary Brier score for the second label, or ``None`` when invalid."""
    y_true = np.asarray(list(targets), dtype=object)
    values = np.asarray(scores, dtype=float)
    class_labels = list(labels) if labels is not None else sorted(set(y_true.tolist()), key=str)
    if (
        len(class_labels) != 2
        or len(set(class_labels)) != 2
        or y_true.size == 0
        or values.ndim not in (1, 2)
        or values.shape[0] != y_true.size
        or (values.ndim == 2 and values.shape[1] != 2)
        or not np.all(np.isfinite(values))
    ):
        return None
    positive = values if values.ndim == 1 else values[:, 1]
    if np.any((positive < 0) | (positive > 1)):
        return None
    return float(np.mean((positive - (y_true == class_labels[1]).astype(int)) ** 2))
