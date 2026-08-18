"""Training-history visualization."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt

from ._common import SavePath, finish_figure, get_axes


def plot_training_history(history: Mapping[str, Sequence[float]], *, metrics: Sequence[str] = ("loss", "f1_macro"), ax: Any = None, save_path: SavePath | None = None, dpi: int = 180) -> plt.Figure:
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
            series = list(history[key])
            if not series:
                raise ValueError(f"history series {key!r} must be non-empty")
            try:
                import numpy as np
                if not np.all(np.isfinite(np.asarray(series, dtype=float))):
                    raise ValueError(f"history series {key!r} must contain finite numbers")
            except (TypeError, ValueError) as error:
                raise ValueError(f"history series {key!r} must contain numeric values") from error
            axis.plot(list(range(1, len(series) + 1)), series, label=key, color=color, lw=1.8)
            plotted += 1
    if not plotted:
        raise ValueError("history does not contain any requested train/valid series")
    axis.set(xlabel="Epoch", title="Training History")
    axis.legend()
    axis.grid(alpha=0.25)
    return finish_figure(fig, save_path, dpi)
