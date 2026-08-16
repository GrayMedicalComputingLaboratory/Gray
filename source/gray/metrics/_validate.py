"""Private validation shared by hard-label metrics."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def validate_targets_predictions(targets: Sequence[Any], predictions: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned, non-empty target and prediction arrays."""
    y_true = np.asarray(list(targets), dtype=object)
    y_pred = np.asarray(list(predictions), dtype=object)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must be non-empty and aligned")
    return y_true, y_pred
