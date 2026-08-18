"""Precision-recall AUC for binary and multiclass classification."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


def pr_auc(targets: Sequence[Any], scores: Sequence[float] | Sequence[Sequence[float]] | np.ndarray, labels: Sequence[Any] | None = None) -> float | None:
    """Return average precision / PR-AUC, or ``None`` when undefined."""
    y_true = np.asarray(list(targets), dtype=object)
    score_array = np.asarray(scores, dtype=float)
    class_labels = list(labels) if labels is not None else sorted(set(y_true.tolist()), key=str)
    if y_true.size == 0 or score_array.ndim not in (1, 2) or score_array.shape[0] != y_true.size or len(set(y_true.tolist())) < 2:
        return None
    if not np.all(np.isfinite(score_array)) or len(set(class_labels)) != len(class_labels):
        return None
    try:
        if len(class_labels) == 2:
            if score_array.ndim == 2 and score_array.shape[1] != 2:
                return None
            positive = score_array if score_array.ndim == 1 else score_array[:, 1]
            return float(average_precision_score(y_true == class_labels[1], positive))
        if score_array.ndim != 2 or score_array.shape[1] != len(class_labels):
            raise ValueError("multiclass scores must have shape [N, number_of_labels]")
        return float(average_precision_score(label_binarize(y_true, classes=class_labels), score_array, average="macro"))
    except ValueError:
        return None
