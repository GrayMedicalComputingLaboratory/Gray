"""Probability averaging for classification model ensembles."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Ensemble:
    """Combine classification probabilities from one or more models.

    This class only aggregates predictions that have already been produced. It
    does not load models, execute forward passes, choose thresholds, or convert
    logits to probabilities.

    Args:
        weights: Optional non-negative weight for each model. Weights are
            normalized to sum to one. ``None`` uses an equal-weight mean.

    Raises:
        TypeError: If ``weights`` is not a sequence of numbers.
        ValueError: If weights are empty, non-finite, negative, or sum to zero.
    """

    def __init__(self, weights: Sequence[float] | None = None) -> None:
        self.weights = self._validate_weights(weights) if weights is not None else None

    def __call__(self, probabilities: Sequence[ArrayLike]) -> NDArray[np.float64]:
        """Delegate to :meth:`predict` so an ensemble instance is callable.

        Args:
            probabilities: One probability array per model.

        Returns:
            Aggregated probabilities with the same shape as one model output.

        Raises:
            TypeError: If ``probabilities`` is not a sequence of arrays.
            ValueError: If prediction arrays violate the probability contract.
        """
        return self.predict(probabilities)

    def predict(self, probabilities: Sequence[ArrayLike]) -> NDArray[np.float64]:
        """Average binary or multiclass classification probabilities.

        Every model output must have the same shape. Binary probabilities use
        shape ``[n_samples]`` and represent the positive class. Multiclass
        probabilities use shape ``[n_samples, n_classes]`` with rows summing to
        one. A single model is supported by passing a one-item sequence.

        Args:
            probabilities: Sequence containing one probability array per model.
                Values must be finite and within ``[0, 1]``.

        Returns:
            Float64 equal-weight or weighted-mean probabilities. A single model
            returns an independent copy of its probability array.

        Raises:
            TypeError: If the outer value is not a sequence.
            ValueError: If the sequence is empty, shapes differ, arrays are not
                one- or two-dimensional, values are invalid, multiclass rows do
                not sum to one, or the number of weights does not match models.
        """
        if isinstance(probabilities, (str, bytes)) or not isinstance(probabilities, Sequence):
            raise TypeError("probabilities must be a sequence with one array per model")
        if not probabilities:
            raise ValueError("probabilities must contain at least one model output")

        arrays = [np.asarray(value, dtype=np.float64) for value in probabilities]
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

        if self.weights is not None and len(self.weights) != len(arrays):
            raise ValueError("weights length must match the number of model outputs")
        stacked = np.stack(arrays, axis=0)
        if self.weights is None:
            return stacked.mean(axis=0)
        return np.average(stacked, axis=0, weights=self.weights)

    @staticmethod
    def _validate_weights(weights: Sequence[float]) -> NDArray[np.float64]:
        if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
            raise TypeError("weights must be a sequence of numbers")
        try:
            values = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError("weights must be a sequence of numbers") from error
        if values.ndim != 1 or values.size == 0:
            raise ValueError("weights must be a non-empty one-dimensional sequence")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("weights must contain finite non-negative values")
        total = float(values.sum())
        if total == 0.0:
            raise ValueError("weights must sum to a positive value")
        return values / total
