"""Clinical positive predictive value."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import precision_score

from ._binary_predictions import binary_predictions


def ppv(targets: Sequence[Any], predictions: Sequence[Any], positive_label: Any | None = None) -> float:
    """Return positive predictive value for the selected positive class."""
    y_true, y_pred, positive = binary_predictions(targets, predictions, positive_label)
    return float(precision_score(y_true, y_pred, pos_label=positive, average="binary", zero_division=0))
