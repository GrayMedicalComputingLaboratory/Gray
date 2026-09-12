"""Apply DICOM Window Width and Window Center to pixel values."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def apply_window_level(
    pixels: ArrayLike,
    window_width: float,
    window_center: float,
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> NDArray[np.float32]:
    """Window image intensities and map them into a requested output range.

    Args:
        pixels: Non-empty real-valued numeric array containing modality values.
        window_width: Positive width of the input intensity window.
        window_center: Center of the input intensity window.
        output_min: Value assigned to intensities at or below the lower bound.
        output_max: Value assigned to intensities at or above the upper bound.

    Returns:
        A new float32 NumPy array clipped and mapped into
        ``[output_min, output_max]``.

    Raises:
        TypeError: If pixels are not a real-valued numeric array or numeric
            arguments have invalid types.
        ValueError: If pixels are empty, a value is non-finite, the window
            width is not positive, or the output range is not increasing.
    """
    array = np.asarray(pixels)
    if array.size == 0:
        raise ValueError("pixels must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("pixels must be a real-valued numeric array")
    try:
        width = float(window_width)
        center = float(window_center)
        minimum = float(output_min)
        maximum = float(output_max)
    except (TypeError, ValueError) as error:
        raise TypeError("window and output range values must be numbers") from error
    if not all(math.isfinite(value) for value in (width, center, minimum, maximum)):
        raise ValueError("window and output range values must be finite")
    if width <= 0:
        raise ValueError("window_width must be positive")
    if maximum <= minimum:
        raise ValueError("output_max must be greater than output_min")
    values = array.astype(np.float32, copy=False)
    if width == 1.0:
        threshold = center - 0.5
        selected = (values > threshold).astype(np.float32)
        result = selected * (maximum - minimum) + minimum
        return result.astype(np.float32, copy=False)
    lower = center - 0.5 - (width - 1.0) / 2.0
    upper = center - 0.5 + (width - 1.0) / 2.0
    clipped = np.clip(values, lower, upper)
    scaled = (clipped - lower) / (upper - lower)
    result = scaled * (maximum - minimum) + minimum
    return result.astype(np.float32, copy=False)
