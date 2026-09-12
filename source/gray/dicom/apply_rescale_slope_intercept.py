"""Apply DICOM Rescale Slope and Rescale Intercept to pixel values."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def apply_rescale_slope_intercept(
    pixels: ArrayLike,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> NDArray[np.float64]:
    """Convert stored DICOM pixel values into modality values.

    The transformation is ``output = pixels * slope + intercept``. For CT,
    this commonly converts stored values into Hounsfield units.

    Args:
        pixels: Real-valued numeric pixel array.
        slope: Finite DICOM ``RescaleSlope`` value. It must not be zero.
        intercept: Finite DICOM ``RescaleIntercept`` value.

    Returns:
        A new float64 NumPy array containing the rescaled modality values.

    Raises:
        TypeError: If pixels are not a real-valued numeric array or slope and
            intercept are not numbers.
        ValueError: If pixels are empty, slope or intercept is non-finite, or
            slope is zero.
    """
    array = np.asarray(pixels)
    if array.size == 0:
        raise ValueError("pixels must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("pixels must be a real-valued numeric array")
    try:
        scale = float(slope)
        offset = float(intercept)
    except (TypeError, ValueError) as error:
        raise TypeError("slope and intercept must be numbers") from error
    if not math.isfinite(scale) or not math.isfinite(offset):
        raise ValueError("slope and intercept must be finite")
    if scale == 0.0:
        raise ValueError("slope must not be zero")
    return array.astype(np.float64, copy=False) * scale + offset
