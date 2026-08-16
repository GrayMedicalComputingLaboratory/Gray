"""Per-class classification specificity."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from ._labels import resolve_labels
from ._validate import validate_targets_predictions


def specificity_per_class(targets: Sequence[Any], predictions: Sequence[Any], labels: Sequence[Any] | None = None) -> dict[str, float]:
    """Return one-vs-rest specificity for every class."""
    y_true, y_pred = validate_targets_predictions(targets, predictions)
    class_labels = resolve_labels(y_true, y_pred, labels)
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    values: dict[str, float] = {}
    for index, label in enumerate(class_labels):
        true_positive = matrix[index, index]
        false_negative = matrix[index, :].sum() - true_positive
        false_positive = matrix[:, index].sum() - true_positive
        true_negative = matrix.sum() - true_positive - false_negative - false_positive
        values[str(label)] = float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else 0.0
    return values
