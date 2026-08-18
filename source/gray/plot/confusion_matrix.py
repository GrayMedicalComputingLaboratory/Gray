"""Confusion-matrix visualization."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from ._common import SavePath, finish_figure, get_axes


def plot_confusion_matrix(
    targets: Sequence[Any],
    predictions: Sequence[Any],
    labels: Sequence[Any] | None = None,
    *,
    normalize: bool = False,
    cmap: str = "Blues",
    annotate: bool = True,
    ax: Any = None,
    save_path: SavePath | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """Plot a label-ordered confusion matrix and return its Figure.

    ``normalize=True`` displays row-wise proportions; otherwise integer counts
    are shown. ``labels`` fixes axis order and is recommended for fold reports.
    """
    y_true, y_pred = list(targets), list(predictions)
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("targets and predictions must be non-empty and aligned")
    ordered = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred), key=str)
    if not ordered:
        raise ValueError("labels must not be empty")
    if len(set(ordered)) != len(ordered):
        raise ValueError("labels must not contain duplicates")
    matrix = sklearn_confusion_matrix(y_true, y_pred, labels=ordered).astype(float)
    values = matrix / matrix.sum(axis=1, keepdims=True) if normalize else matrix
    values = np.nan_to_num(values)
    fig, axis = get_axes(ax)
    image = axis.imshow(values, interpolation="nearest", cmap=cmap, vmin=0, vmax=1 if normalize else None)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Proportion" if normalize else "Count")
    axis.set(xticks=np.arange(len(ordered)), yticks=np.arange(len(ordered)), xticklabels=ordered, yticklabels=ordered, xlabel="Predicted label", ylabel="True label", title="Confusion Matrix")
    axis.set_ylim(len(ordered) - 0.5, -0.5)
    if annotate:
        threshold = values.max() / 2 if values.size else 0
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                text = f"{values[row, column]:.2f}" if normalize else f"{int(values[row, column])}"
                axis.text(column, row, text, ha="center", va="center", color="white" if values[row, column] > threshold else "black")
    return finish_figure(fig, save_path, dpi)
