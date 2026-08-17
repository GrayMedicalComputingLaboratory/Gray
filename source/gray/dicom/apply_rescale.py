"""Apply DICOM Rescale Slope and Rescale Intercept with SimpleITK metadata."""
from __future__ import annotations

import SimpleITK as sitk


def apply_rescale(image: sitk.Image) -> sitk.Image:
    """Convert stored values to physical values using DICOM slope/intercept tags."""
    if not isinstance(image, sitk.Image):
        raise TypeError("image must be a SimpleITK Image")
    slope = float(image.GetMetaData("0028|1053")) if image.HasMetaDataKey("0028|1053") else 1.0
    intercept = float(image.GetMetaData("0028|1052")) if image.HasMetaDataKey("0028|1052") else 0.0
    if slope == 0:
        raise ValueError("Rescale Slope must not be zero")
    return sitk.Cast(image, sitk.sitkFloat32) * slope + intercept
