"""Normalize DICOM MONOCHROME1 polarity with SimpleITK."""
from __future__ import annotations

import SimpleITK as sitk


def apply_monochrome(image: sitk.Image, photometric_interpretation: str | None = None) -> sitk.Image:
    """Invert MONOCHROME1 images; leave MONOCHROME2 images unchanged."""
    if not isinstance(image, sitk.Image):
        raise TypeError("image must be a SimpleITK Image")
    interpretation = photometric_interpretation
    if interpretation is None and image.HasMetaDataKey("0028|0004"):
        interpretation = image.GetMetaData("0028|0004")
    interpretation = (interpretation or "MONOCHROME2").strip().upper()
    if interpretation not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("photometric_interpretation must be MONOCHROME1 or MONOCHROME2")
    if interpretation == "MONOCHROME2":
        return image
    statistics = sitk.StatisticsImageFilter()
    statistics.Execute(image)
    minimum, maximum = statistics.GetMinimum(), statistics.GetMaximum()
    inverted = float(maximum + minimum) - sitk.Cast(image, sitk.sitkFloat32)
    return sitk.Cast(inverted, image.GetPixelID())
