"""Reliability-curve visualization."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt

from gray.metrics.calibration_curve import calibration_curve

from ._common import SavePath, finish_figure, get_axes


def plot_calibration_curve(targets: Sequence[Any], probabilities: Sequence[float], positive_label: Any | None = None, *, n_bins: int = 10, ax: Any = None, save_path: SavePath | None = None, dpi: int = 180) -> plt.Figure:
    """Plot mean predicted probability against observed positive rate."""
    report = calibration_curve(targets, probabilities, positive_label, n_bins)
    points = report["points"]
    fig, axis = get_axes(ax)
    axis.plot([0, 1], [0, 1], "--", color="#89939d", lw=1, label="Perfect calibration")
    axis.plot([point["mean_prediction"] for point in points], [point["positive_rate"] for point in points], "o-", color="#e1b75d", label=f"ECE = {report['ece']:.3f}")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive rate", title="Calibration Curve")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)
    return finish_figure(fig, save_path, dpi)
