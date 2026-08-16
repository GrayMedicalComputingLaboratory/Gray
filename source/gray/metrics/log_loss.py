"""Classification cross-entropy / log loss."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss


def log_loss(targets: Sequence[Any], probabilities: Sequence[float] | Sequence[Sequence[float]] | np.ndarray, labels: Sequence[Any]) -> float | None:
    """Return log loss for label-ordered probabilities, or ``None`` if invalid."""
    y_true = np.asarray(list(targets), dtype=object)
    values = np.asarray(probabilities, dtype=float)
    class_labels = list(labels)
    if y_true.size == 0 or values.shape[0] != y_true.size or not class_labels:
        return None
    if len(class_labels) == 2 and values.ndim == 1:
        values = np.column_stack((1.0 - values, values))
    try:
        return float(sklearn_log_loss(y_true, values, labels=class_labels))
    except ValueError:
        return None
