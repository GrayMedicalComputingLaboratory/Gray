"""Clinical negative predictive value."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._binary_predictions import binary_predictions


def npv(targets: Sequence[Any], predictions: Sequence[Any], positive_label: Any | None = None) -> float:
    """Return negative predictive value for the selected positive class."""
    y_true, y_pred, positive = binary_predictions(targets, predictions, positive_label)
    negative = next(label for label in set(y_true.tolist()) if label != positive)
    true_negative = np.sum((y_true == negative) & (y_pred == negative))
    false_negative = np.sum((y_true == positive) & (y_pred == negative))
    return float(true_negative / (true_negative + false_negative)) if true_negative + false_negative else 0.0
