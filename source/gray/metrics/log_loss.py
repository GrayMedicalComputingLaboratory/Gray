"""Classification cross-entropy / log loss."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss


def log_loss(targets: Sequence[Any], probabilities: Sequence[float] | Sequence[Sequence[float]] | np.ndarray, labels: Sequence[Any] | None = None) -> float | None:
    """Return log loss for label-ordered probabilities, or ``None`` if invalid."""
    y_true = np.asarray(list(targets), dtype=object)
    values = np.asarray(probabilities, dtype=float)
    class_labels = list(labels) if labels is not None else sorted(set(y_true.tolist()), key=str)
    if y_true.size == 0 or values.ndim not in (1, 2) or values.shape[0] != y_true.size or not class_labels or len(set(class_labels)) != len(class_labels):
        return None
    if len(class_labels) == 2 and values.ndim == 1:
        values = np.column_stack((1.0 - values, values))
    if values.ndim != 2 or values.shape[1] != len(class_labels) or not np.all(np.isfinite(values)):
        return None
    if np.any((values < 0) | (values > 1)):
        return None
    try:
        label_to_index = {label: index for index, label in enumerate(class_labels)}
        target_indices = np.asarray([label_to_index[label] for label in y_true.tolist()], dtype=int)
        return float(sklearn_log_loss(target_indices, values, labels=list(range(len(class_labels)))))
    except (KeyError, TypeError):
        return None
    except ValueError:
        return None
