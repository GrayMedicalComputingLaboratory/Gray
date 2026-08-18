"""Clinical threshold-sweep visualization."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt

from gray.metrics.threshold_report import threshold_report

from ._common import finish_figure, get_axes


def plot_threshold_report(targets: Sequence[Any], probabilities: Sequence[float], positive_label: Any | None = None, *, thresholds: Sequence[float] | None = None, ax: Any = None, save_path: str | None = None, dpi: int = 180) -> plt.Figure:
    """Plot sensitivity, specificity, PPV, NPV and F1 across thresholds."""
    report = threshold_report(targets, probabilities, positive_label, thresholds)
    rows = report["rows"]
    fig, axis = get_axes(ax)
    values = {name: [row[name] for row in rows] for name in ("sensitivity", "specificity", "ppv", "npv", "f1")}
    for name, series in values.items():
        axis.plot([row["threshold"] for row in rows], series, lw=1.8, label=name.upper() if name in {"ppv", "npv"} else name.title())
    axis.axvline(report["youden_optimal"]["threshold"], color="#e1b75d", ls="--", lw=1, label="Youden optimal")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Decision threshold", ylabel="Score", title="Clinical Threshold Analysis")
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)
    return finish_figure(fig, save_path, dpi)
