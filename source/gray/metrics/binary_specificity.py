"""Clinical binary specificity (true-negative rate)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._binary_predictions import binary_predictions


def binary_specificity(targets: Sequence[Any], predictions: Sequence[Any], positive_label: Any | None = None) -> float:
    """Return one-vs-rest specificity for the selected positive class."""
    y_true, y_pred, positive = binary_predictions(targets, predictions, positive_label)
    negative = next(label for label in set(y_true.tolist()) if label != positive)
    true_negative = np.sum((y_true == negative) & (y_pred == negative))
    false_positive = np.sum((y_true == negative) & (y_pred == positive))
    return float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else 0.0
