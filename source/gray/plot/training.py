"""Training-history visualization."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt

from ._common import finish_figure, get_axes


def plot_training_history(history: Mapping[str, Sequence[float]], *, metrics: Sequence[str] = ("loss", "f1_macro"), ax: Any = None, save_path: str | None = None, dpi: int = 180) -> plt.Figure:
    """Plot train/validation series from a history mapping.

    Keys should follow ``train.loss``, ``valid.loss`` or nested-free names such
    as ``train_loss``. Missing requested series are ignored; no series raises
    ``ValueError``.
    """
    fig, axis = get_axes(ax)
    plotted = 0
    for metric in metrics:
        for prefix, color in (("train", "#6db8ea"), ("valid", "#e1b75d")):
            key = f"{prefix}.{metric}" if f"{prefix}.{metric}" in history else f"{prefix}_{metric}"
            if key not in history:
                continue
            axis.plot(list(range(1, len(history[key]) + 1)), history[key], label=key, color=color, lw=1.8)
            plotted += 1
    if not plotted:
        raise ValueError("history does not contain any requested train/valid series")
    axis.set(xlabel="Epoch", title="Training History")
    axis.legend()
    axis.grid(alpha=0.25)
    return finish_figure(fig, save_path, dpi)
