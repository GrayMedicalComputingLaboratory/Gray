"""Explicit dimension conversion, casting, and intensity scaling."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InputError
from .types import BATCH_AXES, ImageInput


def prepare(
    image: ImageInput, spec: dict[str, Any], *, allow_batch_size: bool = False
) -> np.ndarray:
    """Convert axes explicitly; only singleton N/C axes may be inserted/removed.

    No resize, color conversion, windowing, interpolation or orientation changes
    are performed. Those remain explicit project/DICOM adapter operations.
    """
    data = image.data
    axes = list(image.dim)
    target = spec["dim"]
    source_batch = next((axis for axis in axes if axis in BATCH_AXES), None)
    target_batch = next((axis for axis in target if axis in BATCH_AXES), None)
    if source_batch is not None and target_batch is not None:
        axes[axes.index(source_batch)] = target_batch
    if set(axes) - set(target) - {"N", "B", "C"} or set(target) - set(axes) - {
        "N",
        "B",
        "C",
    }:
        raise InputError("dim spatial axes differ from model contract")
    for axis in tuple(axes):
        if axis not in target:
            index = axes.index(axis)
            if data.shape[index] != 1:
                raise InputError("Only singleton batch/channel axes can be removed")
            data = np.squeeze(data, axis=index)
            axes.remove(axis)
    for axis in target:
        if axis not in axes:
            data = np.expand_dims(data, -1)
            axes.append(axis)
    data = np.transpose(data, [axes.index(axis) for axis in target])
    pre = spec.get("preprocessing", {})
    if "cast_dtype" in pre:
        data = data.astype(pre["cast_dtype"])
    if data.dtype.name != spec["dtype"]:
        raise InputError(
            "Input dtype differs from manifest; configure cast_dtype explicitly"
        )
    if "normalization" in pre:
        norm = pre["normalization"]
        with np.errstate(over="ignore", invalid="ignore"):
            data = (data * norm["scale"] + norm.get("offset", 0)).astype(spec["dtype"])
    if not np.isfinite(data).all():
        raise InputError("Preprocessing produced nonfinite values")
    batch_token = next((axis for axis in "BN" if axis in target), None)
    batch_axis = target.index(batch_token) if allow_batch_size and batch_token else None
    if any(
        expected is not None and axis != batch_axis and actual != expected
        for axis, (actual, expected) in enumerate(zip(data.shape, spec["shape"]))
    ):
        raise InputError("Input shape differs from manifest")
    return np.ascontiguousarray(data)
