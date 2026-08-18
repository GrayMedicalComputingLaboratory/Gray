"""Private binary-label and score normalization helpers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def binary_inputs(targets: Sequence[Any], scores: Sequence[float] | np.ndarray, positive_label: Any | None = None, labels: Sequence[Any] | None = None) -> tuple[np.ndarray, np.ndarray, Any]:
    """Return binary targets, one-dimensional positive scores and positive label."""
    y_true = np.asarray(list(targets), dtype=object)
    values = np.asarray(scores, dtype=float)
    if y_true.size == 0 or values.ndim not in (1, 2) or values.shape[0] != y_true.size:
        raise ValueError("targets and scores must be non-empty and aligned")
    class_labels = list(labels) if labels is not None else sorted(set(y_true.tolist()), key=str)
    if len(class_labels) != 2 or len(set(class_labels)) != 2:
        raise ValueError("binary clinical metrics require exactly two observed classes")
    observed_labels = set(y_true.tolist())
    if observed_labels != set(class_labels):
        raise ValueError("targets must contain both labels declared for binary metrics")
    if positive_label is None:
        positive_label = class_labels[1]
    if positive_label not in class_labels:
        raise ValueError("positive_label must be one of labels")
    if values.ndim == 2:
        if values.shape[1] != 2:
            raise ValueError("binary score matrices must have shape [N, 2]")
        values = values[:, class_labels.index(positive_label)]
    if positive_label not in set(y_true.tolist()):
        raise ValueError("targets must contain positive_label")
    return y_true, values, positive_label
