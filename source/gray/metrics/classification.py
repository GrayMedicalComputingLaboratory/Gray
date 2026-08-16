"""Reusable classification metrics backed by scikit-learn."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _optional_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _binary_scores(scores: np.ndarray, positive_index: int) -> np.ndarray:
    if scores.ndim == 1:
        return scores.astype(float)
    if scores.ndim == 2 and scores.shape[1] >= 2:
        return scores[:, positive_index].astype(float)
    raise ValueError("binary scores must have shape [N] or [N, 2]")


def classification_metrics(
    targets: Sequence[Any],
    predictions: Sequence[Any],
    scores: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute hard-label and score-based metrics for binary or multiclass data.

    ``scores`` is a positive-class score for binary tasks or an ``[N, C]``
    probability/score matrix for multiclass tasks. Undefined metrics, such as
    ROC-AUC when only one class is present, are returned as ``None``.
    """
    y_true = np.asarray(list(targets), dtype=object)
    y_pred = np.asarray(list(predictions), dtype=object)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must be non-empty and aligned")
    class_labels = list(labels) if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
    if not class_labels:
        raise ValueError("at least one class label is required")

    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=[str(label) for label in class_labels],
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "samples": int(y_true.size),
        "classes": [str(label) for label in class_labels],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": 0.0,
        "precision_macro": float(precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": report,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    specificities: list[float] = []
    for index in range(len(class_labels)):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        specificities.append(float(tn / (tn + fp)) if tn + fp else 0.0)
    result["specificity_macro"] = float(np.mean(specificities))
    result["specificity_per_class"] = {str(label): value for label, value in zip(class_labels, specificities)}

    result.update({"roc_auc": None, "roc_auc_ovr_macro": None, "average_precision": None, "pr_auc": None, "log_loss": None})
    if scores is None:
        return result

    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim not in (1, 2) or score_array.shape[0] != y_true.size:
        raise ValueError("scores must have shape [N] or [N, C] and align with targets")
    try:
        if len(class_labels) == 2:
            positive = _binary_scores(score_array, 1)
            positive_target = (y_true == class_labels[1]).astype(int)
            if np.unique(positive_target).size < 2:
                return result
            result["roc_auc"] = _optional_float(roc_auc_score(positive_target, positive))
            result["roc_auc_ovr_macro"] = result["roc_auc"]
            result["average_precision"] = _optional_float(average_precision_score(positive_target, positive))
            result["pr_auc"] = result["average_precision"]
            probabilities = np.column_stack((1.0 - positive, positive)) if score_array.ndim == 1 else score_array
            result["log_loss"] = _optional_float(log_loss(y_true, probabilities, labels=class_labels))
            result["brier_score"] = _optional_float(np.mean((positive - positive_target) ** 2))
        else:
            if score_array.ndim != 2 or score_array.shape[1] != len(class_labels):
                raise ValueError("multiclass scores must have shape [N, number_of_labels]")
            if np.unique(y_true).size < 2:
                return result
            result["roc_auc_ovr_macro"] = _optional_float(roc_auc_score(y_true, score_array, labels=class_labels, multi_class="ovr", average="macro"))
            result["roc_auc_ovo_macro"] = _optional_float(roc_auc_score(y_true, score_array, labels=class_labels, multi_class="ovo", average="macro"))
            result["average_precision_macro"] = _optional_float(average_precision_score(y_true, score_array, average="macro"))
            result["pr_auc_macro"] = result["average_precision_macro"]
            result["log_loss"] = _optional_float(log_loss(y_true, score_array, labels=class_labels))
    except ValueError:
        # A fold can legitimately contain one class. Keep the report usable.
        result.update({"roc_auc": None, "roc_auc_ovr_macro": None, "average_precision": None, "pr_auc": None, "log_loss": None})
    return result
