"""Resample a SimpleITK volume to a target physical spacing."""
from __future__ import annotations

from collections.abc import Sequence

import SimpleITK as sitk

from .get_spacing import get_spacing


def resample_volume(image: sitk.Image, target_spacing: Sequence[float], interpolator: str = "linear", default_value: float = 0.0) -> sitk.Image:
    """Resample an image to new physical spacing while preserving its geometry.

    Nearest-neighbour output keeps the input pixel type, which is appropriate
    for masks and labels. Linear and B-spline interpolation return float32 so
    fractional interpolated intensities are not quantized back to integers.

    Args:
        image: Input SimpleITK image.
        target_spacing: Positive spacing for every image dimension, in
            SimpleITK axis order.
        interpolator: ``"nearest"``, ``"linear"``, or ``"bspline"``.
        default_value: Pixel value used outside the input image domain.

    Returns:
        A resampled image with adjusted size and requested spacing. Its origin
        and direction match the input image.

    Raises:
        TypeError: If ``image`` is not a SimpleITK image.
        ValueError: If target spacing has the wrong dimension, contains a
            non-positive value, or the interpolator name is unsupported.
    """
    if not isinstance(image, sitk.Image):
        raise TypeError("image must be a SimpleITK Image")
    source_spacing = get_spacing(image)
    spacing = tuple(float(value) for value in target_spacing)
    if len(spacing) != image.GetDimension() or any(value <= 0 for value in spacing):
        raise ValueError("target_spacing must match image dimension and contain positive values")
    interpolators = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
    }
    if interpolator not in interpolators:
        raise ValueError("interpolator must be nearest, linear or bspline")
    source_size = image.GetSize()
    target_size = [max(1, int(round(size * old / new))) for size, old, new in zip(source_size, source_spacing, spacing)]
    output_pixel_id = image.GetPixelID() if interpolator == "nearest" else sitk.sitkFloat32
    return sitk.Resample(
        image,
        target_size,
        sitk.Transform(),
        interpolators[interpolator],
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        float(default_value),
        output_pixel_id,
    )
