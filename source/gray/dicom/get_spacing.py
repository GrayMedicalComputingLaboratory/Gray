"""Read physical voxel spacing from a SimpleITK image."""
from __future__ import annotations

import SimpleITK as sitk


def get_spacing(image: sitk.Image) -> tuple[float, ...]:
    """Return spacing in SimpleITK axis order as positive floats."""
    if not isinstance(image, sitk.Image):
        raise TypeError("image must be a SimpleITK Image")
    spacing = tuple(float(value) for value in image.GetSpacing())
    if not spacing or any(value <= 0 for value in spacing):
        raise ValueError("image spacing must contain positive values")
    return spacing
