"""Precision-recall curve visualization."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from ._common import SavePath, finish_figure, get_axes


def plot_pr_curve(
    targets: Sequence[Any],
    probabilities: Sequence[float],
    positive_label: Any | None = None,
    *,
    ax: Any = None,
    save_path: SavePath | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """Plot a binary precision-recall curve and display average precision."""
    y = np.asarray(list(targets), dtype=object)
    values = np.asarray(probabilities, dtype=float)
    labels = sorted(set(y.tolist()), key=str)
    if y.size == 0 or y.size != values.size or len(labels) != 2 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("PR plotting requires aligned binary targets and probabilities in [0, 1]")
    positive_label = labels[-1] if positive_label is None else positive_label
    if positive_label not in labels:
        raise ValueError("positive_label must be one of the observed labels")
    truth = (y == positive_label).astype(int)
    precision, recall, _ = precision_recall_curve(truth, values)
    average_precision = float(average_precision_score(truth, values))
    fig, axis = get_axes(ax)
    axis.plot(recall, precision, color="#72c48d", lw=2, label=f"AP = {average_precision:.3f}")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axis.legend(loc="lower left")
    axis.grid(alpha=0.25)
    return finish_figure(fig, save_path, dpi)
