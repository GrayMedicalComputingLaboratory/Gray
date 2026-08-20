"""Probability averaging for classification model ensembles."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def Ensemble(
    *probabilities: ArrayLike,
    method: Literal["mean", "weighted"] = "mean",
    weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Combine classification probabilities from one or more models.

    Pass each model output as a separate positional argument. Binary outputs use
    shape ``[n_samples]`` and represent the positive class. Multiclass outputs
    use shape ``[n_samples, n_classes]`` with every row summing to one. This
    function does not accept logits or execute model forward passes.

    Args:
        *probabilities: One probability array per model. Arrays must have equal
            shape and contain finite values within ``[0, 1]``.
        method: ``"mean"`` for equal averaging or ``"weighted"`` for an
            average using ``weights``.
        weights: One non-negative weight per model when ``method="weighted"``.
            Values are normalized to sum to one. Must be ``None`` for ``mean``.

    Returns:
        Float64 probabilities with the same shape as one model output. Passing
        one model returns an independent copy of its probabilities.

    Raises:
        TypeError: If weights are not a sequence of numbers.
        ValueError: If no model output is supplied, ``method`` is unsupported,
            shapes differ, arrays are not one- or two-dimensional, probabilities
            are invalid, multiclass rows do not sum to one, or weights violate
            the selected method contract.
    """
    if method not in {"mean", "weighted"}:
        raise ValueError("method must be 'mean' or 'weighted'")
    if not probabilities:
        raise ValueError("at least one model probability array is required")

    arrays: list[NDArray[np.float64]] = []
    for value in probabilities:
        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise ValueError("probabilities must be real-valued")
        try:
            arrays.append(array.astype(np.float64, copy=False))
        except (TypeError, ValueError) as error:
            raise TypeError("probabilities must be numeric") from error
    expected_shape = arrays[0].shape
    if arrays[0].ndim not in {1, 2} or not expected_shape or expected_shape[0] == 0:
        raise ValueError("each model output must have shape [samples] or [samples, classes]")
    for array in arrays:
        if array.shape != expected_shape:
            raise ValueError("all model outputs must have the same shape")
        if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
            raise ValueError("probabilities must contain finite values within [0, 1]")
        if array.ndim == 2:
            if array.shape[1] < 2:
                raise ValueError("multiclass probabilities must contain at least two classes")
            if not np.allclose(array.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError("each multiclass probability row must sum to one")

    stacked = np.stack(arrays, axis=0)
    if method == "mean":
        if weights is not None:
            raise ValueError("weights must be None when method='mean'")
        return stacked.mean(axis=0)

    normalized_weights = _normalize_weights(weights, len(arrays))
    return np.average(stacked, axis=0, weights=normalized_weights)


def _normalize_weights(weights: Sequence[float] | None, model_count: int) -> NDArray[np.float64]:
    if weights is None:
        raise ValueError("weights are required when method='weighted'")
    if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
        raise TypeError("weights must be a sequence of numbers")
    try:
        values = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("weights must be a sequence of numbers") from error
    if values.ndim != 1 or values.size != model_count:
        raise ValueError("weights length must match the number of model outputs")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weights must contain finite non-negative values")
    maximum = float(values.max())
    if maximum == 0.0:
        raise ValueError("weights must sum to a positive value")
    scaled = values / maximum
    return scaled / scaled.sum()
