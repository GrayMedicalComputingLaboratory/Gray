"""Private binary target and hard-prediction normalization."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def binary_predictions(targets: Sequence[Any], predictions: Sequence[Any], positive_label: Any | None = None) -> tuple[np.ndarray, np.ndarray, Any]:
    """Return aligned binary target/prediction labels and positive label."""
    y_true = np.asarray(list(targets), dtype=object)
    y_pred = np.asarray(list(predictions), dtype=object)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must be non-empty and aligned")
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
    if len(labels) != 2:
        raise ValueError("binary clinical metrics require exactly two observed classes")
    if positive_label is None:
        positive_label = labels[1]
    if positive_label not in labels:
        raise ValueError("targets and predictions must contain positive_label")
    return y_true, y_pred, positive_label
