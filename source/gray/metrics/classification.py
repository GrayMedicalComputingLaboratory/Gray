"""Dependency-light classification metrics suitable for framework smoke tests."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def classification_metrics(targets: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    labels = sorted(set(targets) | set(predictions))
    if not targets or len(targets) != len(predictions): raise ValueError("targets/predictions must be non-empty and aligned")
    accuracy = float(np.mean(np.asarray(targets) == np.asarray(predictions)))
    f1s = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if not denom else 2 * tp / denom)
    return {"accuracy": accuracy, "f1_macro": float(np.mean(f1s)), "samples": float(len(targets))}
