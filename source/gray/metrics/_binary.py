"""Private binary-label and score normalization helpers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def binary_inputs(targets: Sequence[Any], scores: Sequence[float] | np.ndarray, positive_label: Any | None = None) -> tuple[np.ndarray, np.ndarray, Any]:
    """Return binary targets, one-dimensional positive scores and positive label."""
    y_true = np.asarray(list(targets), dtype=object)
    values = np.asarray(scores, dtype=float)
    if y_true.size == 0 or values.ndim not in (1, 2) or values.shape[0] != y_true.size:
        raise ValueError("targets and scores must be non-empty and aligned")
    labels = sorted(set(y_true.tolist()), key=str)
    if len(labels) != 2:
        raise ValueError("binary clinical metrics require exactly two observed classes")
    if positive_label is None:
        positive_label = labels[1]
    if values.ndim == 2:
        if values.shape[1] != 2:
            raise ValueError("binary score matrices must have shape [N, 2]")
        values = values[:, 1]
    if positive_label not in set(y_true.tolist()):
        raise ValueError("targets must contain positive_label")
    return y_true, values, positive_label
