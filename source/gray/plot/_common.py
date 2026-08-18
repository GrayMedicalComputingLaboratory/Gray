"""Shared plotting validation and figure-saving helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def finish_figure(fig: plt.Figure, save_path: str | Path | None, dpi: int) -> plt.Figure:
    """Tighten and optionally save a figure without closing the caller's figure."""
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def get_axes(ax: Any, figsize: tuple[float, float] = (7.0, 5.0)) -> tuple[plt.Figure, Any]:
    """Return a figure and axes, creating them when the caller did not provide axes."""
    if ax is None:
        return plt.subplots(figsize=figsize)
    if not hasattr(ax, "figure"):
        raise TypeError("ax must be a matplotlib Axes")
    return ax.figure, ax
