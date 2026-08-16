"""Classification confusion matrix."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from ._labels import resolve_labels
from ._validate import validate_targets_predictions


def confusion_matrix(targets: Sequence[Any], predictions: Sequence[Any], labels: Sequence[Any] | None = None) -> list[list[int]]:
    """Return a label-ordered integer confusion matrix."""
    y_true, y_pred = validate_targets_predictions(targets, predictions)
    matrix = sklearn_confusion_matrix(y_true, y_pred, labels=resolve_labels(y_true, y_pred, labels))
    return matrix.astype(int).tolist()
