"""Read an already selected and ordered DICOM series with pydicom."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixels import apply_modality_lut

from .apply_rescale_slope_intercept import apply_rescale_slope_intercept


def read_series(file_paths: Sequence[str | Path]) -> NDArray[np.generic]:
    """Read an explicitly ordered DICOM series into an image volume.

    Args:
        file_paths: Non-empty sequence of DICOM files in desired slice order.
            User-home markers such as ``~`` are expanded.

    Returns:
        A NumPy array ordered as ``(z, y, x)``. Stored pixels are decoded by
        pydicom and transformed with the DICOM Modality LUT or Rescale
        Slope/Intercept when present.

    Raises:
        ValueError: If ``file_paths`` is empty.
        FileNotFoundError: If any supplied path is not an existing file.
        TypeError: If ``file_paths`` is a single path or contains a non-path.
        ValueError: If files differ in pixel shape, pixel spacing, photometric
            interpretation, Series Instance UID, or slice spacing.
        RuntimeError: If pydicom cannot read or decode a pixel array, including
            when a compressed transfer syntax needs an unavailable plugin.
    """
    paths = _normalize_paths(file_paths)
    datasets: list[Dataset] = []
    arrays: list[np.ndarray] = []
    for path in paths:
        try:
            dataset = dcmread(path)
            stored = np.asarray(dataset.pixel_array)
            if "ModalityLUTSequence" in dataset:
                decoded = np.asarray(apply_modality_lut(stored, dataset))
            else:
                decoded = apply_rescale_slope_intercept(
                    stored,
                    dataset.get("RescaleSlope", 1.0),
                    dataset.get("RescaleIntercept", 0.0),
                )
        except Exception as error:
            raise RuntimeError(f"failed to decode DICOM pixels: {path}") from error
        if decoded.ndim != 2:
            raise ValueError(f"each DICOM file must contain one 2D frame: {path}")
        datasets.append(dataset)
        arrays.append(decoded)

    _validate_series(datasets, arrays)
    _slice_spacing(datasets)
    return np.stack(arrays, axis=0)


def _normalize_paths(file_paths: Sequence[str | Path]) -> list[Path]:
    """Validate and normalize an ordered DICOM path sequence."""
    if isinstance(file_paths, (str, Path)):
        raise TypeError("file_paths must be a sequence of paths, not a single path")
    try:
        paths = [Path(path).expanduser() for path in file_paths]
    except TypeError as error:
        raise TypeError(
            "file_paths must contain only strings or pathlib.Path values"
        ) from error
    if not paths:
        raise ValueError("file_paths must contain at least one DICOM file")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DICOM files not found: {missing[0]}")
    return paths


def _validate_series(datasets: Sequence[Dataset], arrays: Sequence[np.ndarray]) -> None:
    """Validate geometry and identity shared by decoded DICOM slices.

    Args:
        datasets: Non-empty source datasets in requested slice order.
        arrays: Decoded two-dimensional pixel arrays paired with ``datasets``.

    Returns:
        None.

    Raises:
        ValueError: If shape, PixelSpacing, PhotometricInterpretation, or a
            present SeriesInstanceUID differs between slices.
    """
    shape = arrays[0].shape
    interpretation = (
        str(datasets[0].get("PhotometricInterpretation", "MONOCHROME2")).strip().upper()
    )
    series_uid = str(datasets[0].get("SeriesInstanceUID", ""))
    spacing = _pixel_spacing(datasets[0])
    for dataset, array in zip(datasets[1:], arrays[1:]):
        if array.shape != shape:
            raise ValueError("all DICOM slices must have the same pixel shape")
        if _pixel_spacing(dataset) != spacing:
            raise ValueError("all DICOM slices must have the same PixelSpacing")
        current_interpretation = (
            str(dataset.get("PhotometricInterpretation", "MONOCHROME2")).strip().upper()
        )
        if current_interpretation != interpretation:
            raise ValueError(
                "all DICOM slices must have the same PhotometricInterpretation"
            )
        current_uid = str(dataset.get("SeriesInstanceUID", ""))
        if series_uid and current_uid and current_uid != series_uid:
            raise ValueError(
                "all DICOM slices must belong to the same SeriesInstanceUID"
            )


def _pixel_spacing(dataset: Dataset) -> tuple[float, float]:
    """Extract validated row and column spacing from one dataset.

    Args:
        dataset: pydicom dataset containing a two-value ``PixelSpacing``.

    Returns:
        ``(row_spacing, column_spacing)`` as finite positive floats.

    Raises:
        ValueError: If PixelSpacing is missing, malformed, non-finite, or not
            strictly positive.
    """
    values = dataset.get("PixelSpacing")
    try:
        valid_length = values is not None and len(values) == 2
    except TypeError:
        valid_length = False
    if not valid_length:
        raise ValueError("DICOM PixelSpacing must contain row and column spacing")
    try:
        spacing = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("DICOM PixelSpacing must contain numeric values") from error
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("DICOM PixelSpacing must contain finite positive values")
    return spacing


def _slice_spacing(datasets: Sequence[Dataset]) -> float:
    """Derive uniform slice spacing from ordered DICOM datasets.

    Args:
        datasets: Non-empty datasets in final slice order.

    Returns:
        Positive spacing between adjacent slices. ImagePositionPatient is used
        when available; otherwise SpacingBetweenSlices, SliceThickness, or
        ``1.0`` is used in that order.

    Raises:
        ValueError: If positions are non-uniform or the derived spacing is not
            finite and positive.
    """
    if len(datasets) > 1 and all(
        "ImagePositionPatient" in dataset for dataset in datasets
    ):
        positions = [
            np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
            for dataset in datasets
        ]
        if all(position.shape == (3,) for position in positions):
            distances = [
                float(np.linalg.norm(second - first))
                for first, second in pairwise(positions)
            ]
            if distances and all(
                math.isfinite(value) and value > 0 for value in distances
            ):
                reference = distances[0]
                if not np.allclose(distances, reference, rtol=1e-4, atol=1e-6):
                    raise ValueError(
                        "DICOM slice positions do not have uniform spacing"
                    )
                return reference
    first = datasets[0]
    value = first.get("SpacingBetweenSlices", first.get("SliceThickness", 1.0))
    try:
        spacing = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("DICOM slice spacing must be numeric") from error
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("DICOM slice spacing must be finite and positive")
    return spacing
