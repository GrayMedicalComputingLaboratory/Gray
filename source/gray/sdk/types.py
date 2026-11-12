"""Explicit image and result contracts; no channel or modality guessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .exceptions import InputError

_BASE_INPUT_DIMS = {
    "HW",
    "HWC",
    "CHW",
    "NHWC",
    "NCHW",
    "DHW",
    "NDHW",
    "DHWC",
    "CDHW",
    "NDHWC",
    "NCDHW",
}
INPUT_DIMS = _BASE_INPUT_DIMS | {
    dim.replace("N", "B", 1) for dim in _BASE_INPUT_DIMS if "N" in dim
}
BATCH_AXES = {"B", "N"}


@dataclass(frozen=True)
class ImageInput:
    """Decoded real array. Spatial metadata follows D/H/W axis order.

    dtype defaults to the array's actual dtype and, when supplied, is asserted.
    direction is a spatial square direction-cosine matrix; no reorientation is
    performed. window_applied records caller-owned windowing (None = unknown).
    """

    data: np.ndarray
    dim: str
    dtype: str | None = None
    spacing: tuple[float, ...] | None = None
    direction: tuple[tuple[float, ...], ...] | None = None
    window_applied: bool | None = None

    def __post_init__(self) -> None:
        """Normalize dim case and validate array and optional spatial metadata."""
        if not isinstance(self.data, np.ndarray):
            raise InputError("data must be a NumPy ndarray")
        if isinstance(self.dim, str):
            object.__setattr__(self, "dim", self.dim.upper())
        if (
            not isinstance(self.dim, str)
            or self.dim not in INPUT_DIMS
            or self.data.ndim != len(self.dim)
        ):
            raise InputError(
                "dim must be supported (case-insensitive) and match array dimensions"
            )
        if self.data.dtype.kind not in "uif" or not self.data.size:
            raise InputError("data must be a nonempty real numeric array")
        if not np.isfinite(self.data).all():
            raise InputError("data must contain only finite values")
        if self.dtype is not None and self.dtype != self.data.dtype.name:
            raise InputError("declared dtype differs from array dtype")
        object.__setattr__(self, "dtype", self.data.dtype.name)
        spatial = sum(axis in self.dim for axis in "DHW")
        if self.spacing is not None:
            try:
                values = np.asarray(self.spacing, dtype=float)
            except (TypeError, ValueError) as error:
                raise InputError("spacing must be numeric") from error
            if (
                values.shape != (spatial,)
                or not np.isfinite(values).all()
                or (values <= 0).any()
            ):
                raise InputError(
                    "spacing must contain positive finite values per spatial axis"
                )
            object.__setattr__(self, "spacing", tuple(float(v) for v in values))
        if self.direction is not None:
            try:
                values = np.asarray(self.direction, dtype=float)
            except (TypeError, ValueError) as error:
                raise InputError("direction must be numeric") from error
            if values.shape != (spatial, spatial) or not np.isfinite(values).all():
                raise InputError("direction must be a finite spatial square matrix")
            if not np.allclose(values.T @ values, np.eye(spatial), atol=1e-5):
                raise InputError("direction must be orthonormal")
            object.__setattr__(
                self, "direction", tuple(tuple(float(v) for v in row) for row in values)
            )
        if self.window_applied is not None and not isinstance(
            self.window_applied, bool
        ):
            raise InputError("window_applied must be bool or None")


@dataclass(frozen=True)
class InferenceResult:
    """Named model tensors and non-patient execution provenance."""

    task_type: str
    outputs: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
