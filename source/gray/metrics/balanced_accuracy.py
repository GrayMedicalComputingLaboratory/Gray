"""Balanced classification accuracy."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings

from sklearn.metrics import balanced_accuracy_score

from ._validate import validate_targets_predictions


def balanced_accuracy(targets: Sequence[Any], predictions: Sequence[Any]) -> float:
    """Return scikit-learn balanced accuracy without single-class warnings."""
    y_true, y_pred = validate_targets_predictions(targets, predictions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(balanced_accuracy_score(y_true, y_pred))
