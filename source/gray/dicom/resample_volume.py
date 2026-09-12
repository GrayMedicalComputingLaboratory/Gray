"""Resample a NumPy array to target voxel spacing."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import zoom


def resample_volume(
    pixels: ArrayLike,
    source_spacing: Sequence[float],
    target_spacing: Sequence[float],
    interpolator: str = "linear",
    default_value: float = 0.0,
) -> NDArray[np.generic]:
    """Resample an image to new physical spacing while preserving its geometry.

    Nearest-neighbour output keeps the input pixel type, which is appropriate
    for masks and labels. Linear and B-spline interpolation return float32 so
    fractional interpolated intensities are not quantized back to integers.

    Args:
        pixels: Non-empty real-valued numeric image array.
        source_spacing: Positive spacing for every input array dimension.
        target_spacing: Positive spacing for every image dimension, in
            the same axis order as ``pixels``.
        interpolator: ``"nearest"``, ``"linear"``, or ``"bspline"``.
        default_value: Pixel value used outside the input image domain.

    Returns:
        A NumPy array with adjusted shape. Nearest-neighbour interpolation
        preserves the input dtype; linear and cubic B-spline interpolation
        return float32.

    Raises:
        TypeError: If pixels are not a real-valued numeric array or a spacing
            argument is not a numeric sequence.
        ValueError: If pixels are empty, spacing has the wrong dimension or a
            non-positive value, or the interpolator name is unsupported.
    """
    array = np.asarray(pixels)
    if array.size == 0:
        raise ValueError("pixels must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("pixels must be a real-valued numeric array")
    if isinstance(source_spacing, (str, bytes)) or isinstance(
        target_spacing, (str, bytes)
    ):
        raise TypeError(
            "source_spacing and target_spacing must be sequences of numbers"
        )
    try:
        source = tuple(float(value) for value in source_spacing)
        target = tuple(float(value) for value in target_spacing)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "source_spacing and target_spacing must be sequences of numbers"
        ) from error
    if any(
        len(spacing) != array.ndim
        or any(not math.isfinite(value) or value <= 0 for value in spacing)
        for spacing in (source, target)
    ):
        raise ValueError(
            "spacing must match pixels.ndim and contain finite positive values"
        )
    orders = {"nearest": 0, "linear": 1, "bspline": 3}
    if interpolator not in orders:
        raise ValueError("interpolator must be nearest, linear or bspline")
    try:
        fill_value = float(default_value)
    except (TypeError, ValueError) as error:
        raise TypeError("default_value must be a number") from error
    if not math.isfinite(fill_value):
        raise ValueError("default_value must be finite")
    target_shape = tuple(
        max(1, round(size * old / new))
        for size, old, new in zip(array.shape, source, target)
    )
    factors = tuple(
        size / original for size, original in zip(target_shape, array.shape)
    )
    values = (
        array if interpolator == "nearest" else array.astype(np.float32, copy=False)
    )
    result = zoom(
        values,
        zoom=factors,
        order=orders[interpolator],
        mode="constant",
        cval=fill_value,
        prefilter=interpolator == "bspline",
        grid_mode=False,
    )
    if interpolator == "nearest":
        result = result.astype(array.dtype, copy=False)
    else:
        result = result.astype(np.float32, copy=False)
    return result
