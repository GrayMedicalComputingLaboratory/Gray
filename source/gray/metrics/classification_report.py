"""Per-class precision, recall and F1 report."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import classification_report as sklearn_classification_report

from ._labels import resolve_labels
from ._validate import validate_targets_predictions


def classification_report(targets: Sequence[Any], predictions: Sequence[Any], labels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Return the JSON-serializable scikit-learn classification report."""
    y_true, y_pred = validate_targets_predictions(targets, predictions)
    class_labels = resolve_labels(y_true, y_pred, labels)
    return sklearn_classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=[str(label) for label in class_labels],
        output_dict=True,
        zero_division=0,
    )
