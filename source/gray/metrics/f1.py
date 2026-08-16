"""Classification F1 score."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import f1_score

from ._labels import resolve_labels
from ._validate import validate_targets_predictions


def f1(targets: Sequence[Any], predictions: Sequence[Any], labels: Sequence[Any] | None = None, average: str = "macro") -> float:
    """Return F1 using ``macro``, ``weighted``, ``micro`` or ``binary`` averaging."""
    y_true, y_pred = validate_targets_predictions(targets, predictions)
    return float(f1_score(y_true, y_pred, labels=resolve_labels(y_true, y_pred, labels), average=average, zero_division=0))
