"""Private class-order normalization shared by classification metrics."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def resolve_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Any] | None) -> list[Any]:
    """Return explicitly supplied labels or a stable observed-label ordering."""
    class_labels = list(labels) if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
    if not class_labels:
        raise ValueError("at least one class label is required")
    return class_labels
