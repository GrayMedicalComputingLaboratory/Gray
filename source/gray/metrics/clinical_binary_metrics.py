"""One-call clinical binary classification assessment."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._binary_predictions import binary_predictions
from .accuracy import accuracy
from .binary_specificity import binary_specificity
from .bootstrap_ci import bootstrap_ci
from .calibration_curve import calibration_curve
from .npv import npv
from .ppv import ppv
from .pr_auc import pr_auc
from .roc_auc import roc_auc
from .sensitivity import sensitivity
from .threshold_report import threshold_report


def clinical_binary_metrics(
    targets: Sequence[Any],
    predictions: Sequence[Any],
    probabilities: Sequence[float] | np.ndarray,
    positive_label: Any | None = None,
    n_bins: int = 10,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return a clinical binary report with calibration, CIs and threshold analysis."""
    y_true, y_pred, positive = binary_predictions(targets, predictions, positive_label)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
    negative = next(label for label in labels if label != positive)
    values = np.asarray(probabilities, dtype=float)
    return {
        "positive_label": str(positive),
        "negative_label": str(negative),
        "samples": int(y_true.size),
        "positive_prevalence": float(np.mean(y_true == positive)),
        "accuracy": accuracy(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred, positive),
        "specificity": binary_specificity(y_true, y_pred, positive),
        "ppv": ppv(y_true, y_pred, positive),
        "npv": npv(y_true, y_pred, positive),
        "roc_auc": roc_auc(y_true, values, [negative, positive]),
        "pr_auc": pr_auc(y_true, values, [negative, positive]),
        "calibration": calibration_curve(y_true, values, positive, n_bins),
        "confidence_intervals": {
            "sensitivity": bootstrap_ci(y_true, y_pred, lambda target, prediction: sensitivity(target, prediction, positive), n_bootstrap, confidence, seed),
            "specificity": bootstrap_ci(y_true, y_pred, lambda target, prediction: binary_specificity(target, prediction, positive), n_bootstrap, confidence, seed),
            "ppv": bootstrap_ci(y_true, y_pred, lambda target, prediction: ppv(target, prediction, positive), n_bootstrap, confidence, seed),
            "npv": bootstrap_ci(y_true, y_pred, lambda target, prediction: npv(target, prediction, positive), n_bootstrap, confidence, seed),
            "roc_auc": bootstrap_ci(y_true, values, lambda target, score: roc_auc(target, score.astype(float), [negative, positive]), n_bootstrap, confidence, seed),
            "pr_auc": bootstrap_ci(y_true, values, lambda target, score: pr_auc(target, score.astype(float), [negative, positive]), n_bootstrap, confidence, seed),
        },
        "threshold_report": threshold_report(y_true, values, positive, thresholds),
    }
