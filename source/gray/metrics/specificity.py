"""Macro classification specificity."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .specificity_per_class import specificity_per_class


def specificity(targets: Sequence[Any], predictions: Sequence[Any], labels: Sequence[Any] | None = None) -> float:
    """Return the macro mean of one-vs-rest per-class specificity."""
    values = specificity_per_class(targets, predictions, labels)
    return float(np.mean(list(values.values())))
