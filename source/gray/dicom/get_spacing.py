"""Read physical voxel spacing from an ordered DICOM series."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from pydicom import dcmread
from pydicom.dataset import Dataset

from .read_series import _normalize_paths, _pixel_spacing, _slice_spacing


def get_spacing(file_paths: Sequence[str | Path]) -> tuple[float, float, float]:
    """Read and validate the physical voxel spacing of an image.

    Args:
        file_paths: Non-empty sequence of DICOM files in the same slice order
            used by :func:`read_series`.

    Returns:
        ``(slice, row, column)`` spacing as finite positive floats, matching
        the ``(z, y, x)`` array returned by :func:`read_series`.

    Raises:
        TypeError: If ``file_paths`` is a single path or contains a non-path.
        ValueError: If the path list is empty, slices belong to different
            series, or spacing is missing, inconsistent, or invalid.
        FileNotFoundError: If any supplied path is not an existing file.
        RuntimeError: If pydicom cannot read a DICOM header.
    """
    paths = _normalize_paths(file_paths)
    datasets: list[Dataset] = []
    for path in paths:
        try:
            datasets.append(dcmread(path, stop_before_pixels=True))
        except Exception as error:
            raise RuntimeError(f"failed to read DICOM header: {path}") from error
    first_spacing = _pixel_spacing(datasets[0])
    first_uid = str(datasets[0].get("SeriesInstanceUID", ""))
    for dataset in datasets[1:]:
        if _pixel_spacing(dataset) != first_spacing:
            raise ValueError("all DICOM slices must have the same PixelSpacing")
        current_uid = str(dataset.get("SeriesInstanceUID", ""))
        if first_uid and current_uid and current_uid != first_uid:
            raise ValueError(
                "all DICOM slices must belong to the same SeriesInstanceUID"
            )
    slice_spacing = _slice_spacing(datasets)
    spacing = (slice_spacing, *first_spacing)
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("DICOM spacing must contain finite positive values")
    return spacing
