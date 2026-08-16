"""Classification accuracy."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import accuracy_score

from ._validate import validate_targets_predictions


def accuracy(targets: Sequence[Any], predictions: Sequence[Any]) -> float:
    """Return the fraction of correct predictions."""
    y_true, y_pred = validate_targets_predictions(targets, predictions)
    return float(accuracy_score(y_true, y_pred))
