"""Clinical binary sensitivity (true-positive rate)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import recall_score

from ._binary_predictions import binary_predictions


def sensitivity(targets: Sequence[Any], predictions: Sequence[Any], positive_label: Any | None = None) -> float:
    """Return sensitivity for the selected positive class."""
    y_true, y_pred, positive = binary_predictions(targets, predictions, positive_label)
    return float(recall_score(y_true, y_pred, pos_label=positive, average="binary", zero_division=0))
