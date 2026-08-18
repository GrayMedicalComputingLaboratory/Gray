"""One-call JSON-serializable classification metric summary."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._labels import resolve_labels
from ._validate import validate_targets_predictions
from .accuracy import accuracy
from .balanced_accuracy import balanced_accuracy
from .brier_score import brier_score
from .classification_report import classification_report
from .confusion_matrix import confusion_matrix
from .f1 import f1
from .log_loss import log_loss
from .pr_auc import pr_auc
from .precision import precision
from .recall import recall
from .roc_auc import roc_auc
from .specificity import specificity
from .specificity_per_class import specificity_per_class


def classification_metrics(
    targets: Sequence[Any],
    predictions: Sequence[Any],
    probabilities: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Combine independently callable metrics into one saved experiment report."""
    y_true, y_pred = validate_targets_predictions(targets, predictions)
    class_labels = resolve_labels(y_true, y_pred, labels)
    result: dict[str, Any] = {
        "samples": int(y_true.size),
        "classes": [str(label) for label in class_labels],
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "precision_macro": precision(y_true, y_pred, class_labels, "macro"),
        "precision_weighted": precision(y_true, y_pred, class_labels, "weighted"),
        "recall_macro": recall(y_true, y_pred, class_labels, "macro"),
        "recall_weighted": recall(y_true, y_pred, class_labels, "weighted"),
        "f1_macro": f1(y_true, y_pred, class_labels, "macro"),
        "f1_weighted": f1(y_true, y_pred, class_labels, "weighted"),
        "specificity_macro": specificity(y_true, y_pred, class_labels),
        "specificity_per_class": specificity_per_class(y_true, y_pred, class_labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, class_labels),
        "classification_report": classification_report(y_true, y_pred, class_labels),
        "roc_auc": None,
        "roc_auc_ovr_macro": None,
        "pr_auc": None,
        "average_precision": None,
        "log_loss": None,
        "brier_score": None,
    }
    if probabilities is None:
        return result
    result["roc_auc"] = roc_auc(y_true, probabilities, class_labels)
    result["roc_auc_ovr_macro"] = result["roc_auc"]
    result["pr_auc"] = pr_auc(y_true, probabilities, class_labels)
    result["average_precision"] = result["pr_auc"]
    result["log_loss"] = log_loss(y_true, probabilities, class_labels)
    if len(class_labels) == 2:
        result["brier_score"] = brier_score(y_true, probabilities, class_labels)
    return result
