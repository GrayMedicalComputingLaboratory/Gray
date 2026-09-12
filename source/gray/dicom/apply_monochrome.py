"""Normalize DICOM MONOCHROME1 pixel polarity."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def apply_monochrome(
    pixels: ArrayLike,
    photometric_interpretation: str,
) -> NDArray[np.generic]:
    """Normalize a monochrome DICOM image to MONOCHROME2 polarity.

    Args:
        pixels: Non-empty real-valued numeric pixel array.
        photometric_interpretation: DICOM ``MONOCHROME1`` or ``MONOCHROME2``.

    Returns:
        A NumPy array with MONOCHROME2 polarity and the input dtype. A copy is
        returned so callers may safely mutate it.

    Raises:
        TypeError: If pixels are not a real-valued numeric array or the
            interpretation is not a string.
        ValueError: If pixels are empty or the interpretation is unsupported.
    """
    array = np.asarray(pixels)
    if array.size == 0:
        raise ValueError("pixels must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("pixels must be a real-valued numeric array")
    if not isinstance(photometric_interpretation, str):
        raise TypeError("photometric_interpretation must be a string")
    interpretation = photometric_interpretation.strip().upper()
    if interpretation not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(
            "photometric_interpretation must be MONOCHROME1 or MONOCHROME2"
        )
    if interpretation == "MONOCHROME2":
        return array.copy()
    numeric = array.astype(np.float64, copy=False)
    minimum = float(np.min(numeric))
    maximum = float(np.max(numeric))
    return (maximum + minimum - numeric).astype(array.dtype, copy=False)
