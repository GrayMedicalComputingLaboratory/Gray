"""Metric and confidence-interval visualizations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ._common import finish_figure, get_axes


def plot_metrics(
    metrics: Mapping[str, float],
    ci: Mapping[str, Mapping[str, float | None]] | None = None,
    *,
    title: str = "Classification Metrics",
    ylim: tuple[float, float] | None = (0.0, 1.0),
    ax: Any = None,
    save_path: str | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """Plot scalar metrics with optional asymmetric confidence intervals.

    ``ci`` follows the framework report shape, for example
    ``{"roc_auc": {"estimate": .81, "lower": .74, "upper": .87}}``.
    Missing or ``None`` bounds are rendered without an error bar.
    """
    if not metrics:
        raise ValueError("metrics must be non-empty")
    names, values = list(metrics), np.asarray([float(metrics[name]) for name in metrics], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("metrics must contain finite values")
    fig, axis = get_axes(ax)
    positions = np.arange(len(names))
    axis.bar(positions, values, color="#6db8ea", edgecolor="#bfe5ff")
    if ci:
        for position, name in zip(positions, names):
            bound = ci.get(name)
            if not bound or bound.get("lower") is None or bound.get("upper") is None:
                continue
            lower, upper = float(bound["lower"]), float(bound["upper"])
            axis.errorbar(position, values[position], yerr=[[values[position] - lower], [upper - values[position]]], fmt="none", ecolor="#e1b75d", capsize=4, capthick=1.5)
    axis.set(xticks=positions, xticklabels=names, ylabel="Score", title=title)
    axis.tick_params(axis="x", rotation=35)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(axis="y", alpha=0.25)
    return finish_figure(fig, save_path, dpi)
