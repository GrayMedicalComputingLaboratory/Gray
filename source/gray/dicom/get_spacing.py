"""Read physical voxel spacing from a SimpleITK image."""
from __future__ import annotations

import SimpleITK as sitk


def get_spacing(image: sitk.Image) -> tuple[float, ...]:
    """Read and validate the physical voxel spacing of an image.

    Args:
        image: Input SimpleITK image.

    Returns:
        Positive spacing values in SimpleITK axis order, such as ``(x, y, z)``.

    Raises:
        TypeError: If ``image`` is not a SimpleITK image.
        ValueError: If spacing is empty or contains a non-positive value.
    """
    if not isinstance(image, sitk.Image):
        raise TypeError("image must be a SimpleITK Image")
    spacing = tuple(float(value) for value in image.GetSpacing())
    if not spacing or any(value <= 0 for value in spacing):
        raise ValueError("image spacing must contain positive values")
    return spacing
