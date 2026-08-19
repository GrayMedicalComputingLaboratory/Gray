"""Apply DICOM Rescale Slope and Rescale Intercept with SimpleITK metadata."""
from __future__ import annotations

import SimpleITK as sitk


def apply_rescale(image: sitk.Image) -> sitk.Image:
    """Convert stored DICOM pixels to physical values using rescale metadata.

    Args:
        image: Input SimpleITK image. Tags ``0028|1053`` and ``0028|1052`` are
            read as slope and intercept; absent tags default to 1 and 0.

    Returns:
        A float image calculated as ``image * slope + intercept``.

    Raises:
        TypeError: If ``image`` is not a SimpleITK image.
        ValueError: If metadata is non-numeric or the slope is zero.
    """
    if not isinstance(image, sitk.Image):
        raise TypeError("image must be a SimpleITK Image")
    slope = float(image.GetMetaData("0028|1053")) if image.HasMetaDataKey("0028|1053") else 1.0
    intercept = float(image.GetMetaData("0028|1052")) if image.HasMetaDataKey("0028|1052") else 0.0
    if slope == 0:
        raise ValueError("Rescale Slope must not be zero")
    return sitk.Cast(image, sitk.sitkFloat32) * slope + intercept
