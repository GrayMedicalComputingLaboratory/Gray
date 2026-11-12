"""Decode common 2D image files into explicit SDK image inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .exceptions import InputError
from .types import ImageInput

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path, *, dim: str | None = None) -> ImageInput:
    """Decode a 2D image file as grayscale HW or RGB HWC/CHW pixels.

    EXIF orientation is applied. Palette, alpha, CMYK, and other color modes
    are converted to RGB; grayscale bit depth is retained where Pillow allows.
    Pixel values are not normalized or resized. Install ``gray[sdk]`` for the
    Pillow decoder.
    """
    source = Path(path).expanduser()
    if source.suffix.lower() not in _IMAGE_SUFFIXES:
        raise InputError(f"Unsupported image file extension: {source.suffix}")
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise InputError("Install gray[sdk] to decode image files") from error
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            grayscale = image.mode in {"L", "I", "I;16", "I;16L", "I;16B", "F"}
            if grayscale:
                data = np.array(image, copy=True)
                actual_dim = "HW"
            else:
                data = np.array(image.convert("RGB"), copy=True)
                actual_dim = "HWC"
    except (OSError, ValueError) as error:
        raise InputError(f"Cannot decode image file: {source}") from error
    if data.ndim != 2 and data.ndim != 3:
        raise InputError("Decoded image must be a 2D grayscale or RGB image")
    requested = actual_dim if dim is None else dim.upper() if isinstance(dim, str) else ""
    if requested == "CHW" and actual_dim == "HWC":
        data = np.moveaxis(data, -1, 0)
    elif requested != actual_dim and not (requested == "CHW" and actual_dim == "CHW"):
        raise InputError("Image dim must be HW, HWC, or CHW for the decoded image")
    return ImageInput(data, requested)
