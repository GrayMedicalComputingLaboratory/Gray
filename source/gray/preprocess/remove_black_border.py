"""Cropping of empty black image borders."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def remove_black_border(
    image: ArrayLike,
    *,
    threshold: float = 0.0,
    spatial_axes: Sequence[int] = (-2, -1),
) -> NDArray[np.generic]:
    """Crop black margins from selected spatial axes of an image array.

    Foreground is any value whose absolute magnitude exceeds ``threshold``.
    All non-spatial axes are reduced together, so the default supports ``H, W``,
    ``C, H, W``, and ``B, C, H, W`` arrays while preserving leading dimensions.
    For a three-dimensional volume, use ``spatial_axes=(0, 1, 2)`` to crop all
    volume axes. An entirely black input is returned unchanged to avoid an empty
    result.

    Args:
        image: Numeric NumPy-compatible array to crop.
        threshold: Non-negative finite magnitude below which pixels are treated
            as black.
        spatial_axes: At least two distinct axes to crop. Negative axes follow
            standard NumPy indexing rules.

    Returns:
        A NumPy array cropped to the smallest bounding box that contains every
        foreground value on the selected axes. The cropped result is a view of
        the converted input array. If no foreground exists, the converted input
        array is returned unchanged.

    Raises:
        TypeError: If ``image`` is non-numeric, ``threshold`` is not numeric,
            or ``spatial_axes`` is not a sequence of integer axes.
        ValueError: If ``image`` has fewer than two dimensions, ``threshold`` is
            negative or non-finite, or spatial axes are invalid or duplicated.
    """
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must have a numeric dtype")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
        raise TypeError("threshold must be a number")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    axes = _normalize_axes(spatial_axes, array.ndim)
    foreground = np.abs(array) > threshold
    reduction_axes = tuple(axis for axis in range(array.ndim) if axis not in axes)
    spatial_foreground = np.any(foreground, axis=reduction_axes)
    if not spatial_foreground.any():
        return array

    slices: list[slice] = [slice(None)] * array.ndim
    for coordinate_axis, array_axis in enumerate(axes):
        occupied = np.any(spatial_foreground, axis=tuple(index for index in range(len(axes)) if index != coordinate_axis))
        indices = np.flatnonzero(occupied)
        slices[array_axis] = slice(int(indices[0]), int(indices[-1]) + 1)
    return array[tuple(slices)]


def _normalize_axes(spatial_axes: Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(spatial_axes, (str, bytes)) or not isinstance(spatial_axes, Sequence):
        raise TypeError("spatial_axes must be a sequence of integer axes")
    if len(spatial_axes) < 2:
        raise ValueError("spatial_axes must contain at least two axes")
    axes: list[int] = []
    for axis in spatial_axes:
        if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
            raise TypeError("spatial_axes must contain integer axes")
        normalized = int(axis) % ndim
        if axis < -ndim or axis >= ndim:
            raise ValueError("spatial_axes contains an axis outside image dimensions")
        if normalized in axes:
            raise ValueError("spatial_axes must not contain duplicate axes")
        axes.append(normalized)
    # NumPy keeps remaining dimensions in ascending axis order after reduction.
    # Matching that order keeps coordinates aligned for user-provided axis orders.
    return tuple(sorted(axes))
