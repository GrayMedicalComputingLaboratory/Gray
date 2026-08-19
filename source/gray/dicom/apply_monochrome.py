"""Normalize DICOM MONOCHROME1 polarity with SimpleITK."""
from __future__ import annotations

import SimpleITK as sitk


def apply_monochrome(image: sitk.Image, photometric_interpretation: str | None = None) -> sitk.Image:
    """Normalize a monochrome DICOM image to MONOCHROME2 polarity.

    Args:
        image: Input SimpleITK image. When ``photometric_interpretation`` is
            omitted, metadata tag ``0028|0004`` is used when available.
        photometric_interpretation: Explicit ``MONOCHROME1`` or
            ``MONOCHROME2`` value. Missing metadata defaults to ``MONOCHROME2``.

    Returns:
        A polarity-inverted image with the original pixel type for
        ``MONOCHROME1``; the original image object for ``MONOCHROME2``.

    Raises:
        TypeError: If ``image`` is not a SimpleITK image.
        ValueError: If the photometric interpretation is unsupported.
    """
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
