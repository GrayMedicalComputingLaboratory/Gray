"""Binary clinical threshold sweep and selection report."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._binary import binary_inputs
from .binary_specificity import binary_specificity
from .f1 import f1
from .npv import npv
from .ppv import ppv
from .sensitivity import sensitivity


def threshold_report(targets: Sequence[Any], probabilities: Sequence[float] | np.ndarray, positive_label: Any | None = None, thresholds: Sequence[float] | None = None, labels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Evaluate thresholds and select Youden-J and positive-class-F1 points."""
    y_true, values, positive = binary_inputs(targets, probabilities, positive_label, labels)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    candidate_thresholds = list(thresholds) if thresholds is not None else np.linspace(0.01, 0.99, 99).tolist()
    if not candidate_thresholds or any(not 0 <= threshold <= 1 for threshold in candidate_thresholds):
        raise ValueError("thresholds must be non-empty values in [0, 1]")
    negative = next(label for label in set(y_true.tolist()) if label != positive)
    rows: list[dict[str, float]] = []
    for threshold in sorted(set(float(value) for value in candidate_thresholds)):
        predictions = np.where(values >= threshold, positive, negative)
        sensitivity_value = sensitivity(y_true, predictions, positive)
        specificity_value = binary_specificity(y_true, predictions, positive)
        f1_macro = f1(y_true, predictions, [negative, positive], "macro")
        f1_positive = f1(y_true, predictions, [negative, positive], "binary", positive)
        rows.append({
            "threshold": threshold,
            "sensitivity": sensitivity_value,
            "specificity": specificity_value,
            "ppv": ppv(y_true, predictions, positive),
            "npv": npv(y_true, predictions, positive),
            "f1_macro": f1_macro,
            "f1_positive": f1_positive,
            "f1": f1_positive,
            "youden_j": sensitivity_value + specificity_value - 1,
        })
    youden = max(rows, key=lambda row: (row["youden_j"], row["sensitivity"], -row["threshold"]))
    best_f1 = max(rows, key=lambda row: (row["f1_positive"], row["sensitivity"], -row["threshold"]))
    best_f1_macro = max(rows, key=lambda row: (row["f1_macro"], row["sensitivity"], -row["threshold"]))
    return {"positive_label": str(positive), "rows": rows, "youden_optimal": youden, "f1_optimal": best_f1, "f1_macro_optimal": best_f1_macro}
