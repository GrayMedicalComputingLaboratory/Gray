"""Apply DICOM Window Width and Window Center with SimpleITK."""
from __future__ import annotations

import SimpleITK as sitk


def apply_window_level(image: sitk.Image, window_width: float, window_center: float, output_min: float = 0.0, output_max: float = 1.0) -> sitk.Image:
    """Clip a volume to the requested window and map it to an output range."""
    if not isinstance(image, sitk.Image):
        raise TypeError("image must be a SimpleITK Image")
    if window_width <= 0:
        raise ValueError("window_width must be positive")
    if output_max <= output_min:
        raise ValueError("output_max must be greater than output_min")
    lower = float(window_center) - float(window_width) / 2.0
    upper = float(window_center) + float(window_width) / 2.0
    return sitk.IntensityWindowing(
        sitk.Cast(image, sitk.sitkFloat32),
        windowMinimum=lower,
        windowMaximum=upper,
        outputMinimum=float(output_min),
        outputMaximum=float(output_max),
    )
