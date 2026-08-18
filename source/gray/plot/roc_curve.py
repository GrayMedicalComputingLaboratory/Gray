"""ROC curve and bootstrap confidence-band visualization."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

from ._common import finish_figure, get_axes


def plot_roc_auc(
    targets: Sequence[Any],
    probabilities: Sequence[float],
    positive_label: Any | None = None,
    *,
    ci: bool = False,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
    ax: Any = None,
    save_path: str | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """Plot a binary ROC curve, optionally with a bootstrap AUC CI.

    ``probabilities`` must be the probability of ``positive_label``. The
    confidence interval is computed for the scalar AUC and shown in the legend;
    it is not a pointwise curve band.
    """
    y = np.asarray(list(targets), dtype=object)
    scores = np.asarray(probabilities, dtype=float)
    if y.size == 0 or y.size != scores.size or not np.all(np.isfinite(scores)):
        raise ValueError("targets and probabilities must be non-empty, aligned and finite")
    labels = sorted(set(y.tolist()), key=str)
    if positive_label is None:
        if len(labels) != 2:
            raise ValueError("positive_label is required unless exactly two labels are observed")
        positive_label = labels[-1]
    if len(set(y.tolist())) != 2 or positive_label not in labels or np.any((scores < 0) | (scores > 1)):
        raise ValueError("ROC plotting requires two labels and probabilities in [0, 1]")
    truth = (y == positive_label).astype(int)
    fpr, tpr, _ = roc_curve(truth, scores)
    estimate = float(roc_auc_score(truth, scores))
    legend = f"AUC = {estimate:.3f}"
    if ci:
        if n_bootstrap < 100 or not 0 < confidence < 1:
            raise ValueError("n_bootstrap must be at least 100 and confidence must be in (0, 1)")
        generator = np.random.default_rng(seed)
        values: list[float] = []
        for _ in range(n_bootstrap):
            index = generator.integers(0, y.size, y.size)
            if np.unique(truth[index]).size < 2:
                continue
            values.append(float(roc_auc_score(truth[index], scores[index])))
        if values:
            alpha = (1 - confidence) / 2
            lower, upper = np.quantile(values, [alpha, 1 - alpha])
            legend += f" ({confidence:.0%} CI {lower:.3f}-{upper:.3f})"
    fig, axis = get_axes(ax)
    axis.plot(fpr, tpr, color="#6db8ea", lw=2, label=legend)
    axis.plot([0, 1], [0, 1], "--", color="#89939d", lw=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    return finish_figure(fig, save_path, dpi)
