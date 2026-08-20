"""Inference-time spatial test-time augmentation."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np


def tta(
    sample: Any,
    *,
    dim: Literal["2d", "3d"],
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    rotate90_angles: Sequence[int] = (0,),
) -> dict[str, Any]:
    """Generate selected in-plane test-time augmentation variants.

    The two spatial axes are always the final ``H, W`` axes. For ``dim="3d"``,
    all leading axes, including depth/Z, remain in place; every slice receives
    the same in-plane transform. Variants are independent rather than a
    flip-rotation Cartesian product: ``original``, each selected flip, and each
    selected non-zero rotation are returned once.

    Args:
        sample: NumPy array or PyTorch tensor with spatial ``H, W`` as its final
            two axes.
        dim: ``"2d"`` for samples with at least two axes or ``"3d"`` for
            samples with at least three axes.
        horizontal_flip: Include a variant flipped along the final axis.
        vertical_flip: Include a variant flipped along the penultimate axis.
        rotate90_angles: Integer multiples of 90 degrees. Angles are normalized
            modulo 360 and duplicate rotations are removed.

    Returns:
        A dictionary from deterministic variant names to transformed samples.
        It always contains ``"original"``. NumPy transformations are copied;
        the original value is retained by reference.

    Raises:
        ValueError: If ``dim`` is invalid, the sample has too few dimensions,
            or a rotation is not an integer multiple of 90.
        TypeError: If flip flags are not booleans or ``sample`` is neither a
            NumPy array nor a PyTorch tensor.
    """
    if dim not in {"2d", "3d"}:
        raise ValueError("dim must be '2d' or '3d'")
    minimum_ndim = 2 if dim == "2d" else 3
    if not hasattr(sample, "ndim") or sample.ndim < minimum_ndim:
        raise ValueError(f"{dim} sample must have at least {minimum_ndim} dimensions with H, W last")
    if not isinstance(horizontal_flip, bool) or not isinstance(vertical_flip, bool):
        raise TypeError("horizontal_flip and vertical_flip must be booleans")

    try:
        import torch
    except ImportError:
        torch = None

    is_numpy = isinstance(sample, np.ndarray)
    is_torch = torch is not None and isinstance(sample, torch.Tensor)
    if not is_numpy and not is_torch:
        raise TypeError("sample must be a numpy.ndarray or torch.Tensor")

    normalized_angles: list[int] = []
    for angle in rotate90_angles:
        if not isinstance(angle, int) or angle % 90:
            raise ValueError("rotate90_angles must contain integer multiples of 90")
        normalized = angle % 360
        if normalized not in normalized_angles:
            normalized_angles.append(normalized)

    variants: dict[str, Any] = {"original": sample}
    if horizontal_flip:
        variants["horizontal_flip"] = np.flip(sample, axis=-1).copy() if is_numpy else torch.flip(sample, dims=(-1,))
    if vertical_flip:
        variants["vertical_flip"] = np.flip(sample, axis=-2).copy() if is_numpy else torch.flip(sample, dims=(-2,))
    for angle in normalized_angles:
        if angle:
            variants[f"rotate90_{angle}"] = np.rot90(sample, k=angle // 90, axes=(-2, -1)).copy() if is_numpy else torch.rot90(sample, k=angle // 90, dims=(-2, -1))
    return variants
