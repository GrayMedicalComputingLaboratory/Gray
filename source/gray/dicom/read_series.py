"""Read an already selected and ordered DICOM Series with SimpleITK."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import SimpleITK as sitk


def read_series(file_paths: Sequence[str | Path]) -> sitk.Image:
    """Read an explicitly ordered DICOM series into an image volume.

    Args:
        file_paths: Non-empty sequence of DICOM files in desired slice order.
            User-home markers such as ``~`` are expanded.

    Returns:
        The volume produced by :class:`SimpleITK.ImageSeriesReader`.

    Raises:
        ValueError: If ``file_paths`` is empty.
        FileNotFoundError: If any supplied path is not an existing file.
        RuntimeError: If SimpleITK cannot decode or assemble the series.
    """
    if isinstance(file_paths, (str, Path)):
        raise TypeError("file_paths must be a sequence of paths, not a single path")
    paths = [str(Path(path).expanduser()) for path in file_paths]
    if not paths:
        raise ValueError("file_paths must contain at least one DICOM file")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"DICOM files not found: {missing[0]}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    image = reader.Execute()
    metadata_reader = sitk.ImageFileReader()
    metadata_reader.SetFileName(paths[0])
    metadata_reader.ReadImageInformation()
    photometric_key = "0028|0004"
    if metadata_reader.HasMetaDataKey(photometric_key):
        image.SetMetaData(photometric_key, metadata_reader.GetMetaData(photometric_key))
    return image
