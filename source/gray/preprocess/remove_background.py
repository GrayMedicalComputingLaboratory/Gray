"""Cropping of uniform image backgrounds."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def remove_background(
    image: ArrayLike,
    *,
    background_colors: ArrayLike = 0,
    tolerance: float = 0.0,
    spatial_axes: Sequence[int] = (-2, -1),
) -> NDArray[np.generic]:
    """Crop margins whose pixels match one or more background colors.

    Non-spatial axes are treated as channels or other feature dimensions. For
    example, an ``(C, H, W)`` image accepts a color shaped ``(C,)`` and multiple
    colors shaped ``(N, C)``. A pixel is background only when every channel is
    within ``tolerance`` of at least one supplied color.

    Args:
        image: Numeric NumPy-compatible image or volume array.
        background_colors: One color or a collection of colors. Scalars apply
            to grayscale data; color vectors match the non-spatial dimensions.
        tolerance: Non-negative finite absolute tolerance for color matching.
        spatial_axes: At least two distinct axes to crop. Negative axes follow
            standard NumPy indexing rules.

    Returns:
        A view cropped to the smallest bounding box containing every foreground
        value. If the input contains only background, its shape is unchanged.

    Raises:
        TypeError: If the image, colors, tolerance, or axes are not numeric and
            structurally valid.
        ValueError: If dimensions, axes, tolerance, or color shapes are invalid.
    """
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must have a numeric dtype")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float, np.number)):
        raise TypeError("tolerance must be a number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    axes = _normalize_axes(spatial_axes, array.ndim)
    non_spatial_axes = tuple(axis for axis in range(array.ndim) if axis not in axes)
    ordered = np.transpose(array, non_spatial_axes + axes)
    prefix_shape = ordered.shape[: len(non_spatial_axes)]
    channel_count = prefix_shape[-1] if prefix_shape else None
    colors = _normalize_colors(background_colors, channel_count)
    if channel_count is None:
        candidates = colors.reshape((colors.shape[0],) + (1,) * len(axes))
        matches = np.any(np.abs(ordered[None, ...] - candidates) <= tolerance, axis=0)
    else:
        batch_shape = prefix_shape[:-1]
        candidates = colors.reshape(
            (colors.shape[0],) + (1,) * len(batch_shape) + (channel_count,) + (1,) * len(axes)
        )
        matches = np.any(np.abs(ordered[None, ...] - candidates) <= tolerance, axis=0)
        matches = np.all(matches, axis=len(batch_shape))
    foreground = ~matches
    if prefix_shape:
        foreground = np.any(foreground, axis=tuple(range(len(prefix_shape) - 1)))
    spatial_foreground = foreground
    if not spatial_foreground.any():
        return array

    slices: list[slice] = [slice(None)] * array.ndim
    for coordinate_axis, array_axis in enumerate(axes):
        occupied = np.any(
            spatial_foreground,
            axis=tuple(index for index in range(len(axes)) if index != coordinate_axis),
        )
        indices = np.flatnonzero(occupied)
        slices[array_axis] = slice(int(indices[0]), int(indices[-1]) + 1)
    return array[tuple(slices)]


def _normalize_colors(colors: ArrayLike, channel_count: int | None) -> NDArray[np.float64]:
    try:
        values = np.asarray(colors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("background_colors must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError("background_colors must contain finite values")
    if channel_count is None:
        return values.reshape(-1)
    if values.ndim == 0:
        return np.full((1, channel_count), float(values))
    if values.shape == (channel_count,):
        return values.reshape((1, channel_count))
    if values.ndim == 2 and values.shape[1] == channel_count:
        return values
    raise ValueError(
        "background_colors must be a scalar, a channel-shaped color, "
        "or an array shaped [colors, channels]"
    )


def _normalize_axes(spatial_axes: Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(spatial_axes, (str, bytes)) or not isinstance(spatial_axes, Sequence):
        raise TypeError("spatial_axes must be a sequence of integer axes")
    if len(spatial_axes) < 2:
        raise ValueError("spatial_axes must contain at least two axes")
    axes: list[int] = []
    for axis in spatial_axes:
        if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
            raise TypeError("spatial_axes must contain integer axes")
        if axis < -ndim or axis >= ndim:
            raise ValueError("spatial_axes contains an axis outside image dimensions")
        normalized = int(axis) % ndim
        if normalized in axes:
            raise ValueError("spatial_axes must not contain duplicate axes")
        axes.append(normalized)
    return tuple(sorted(axes))
