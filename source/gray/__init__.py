"""Gray: small, extensible computer-vision research framework."""

from importlib.metadata import PackageNotFoundError, version

from .core.config import load_config

try:
    __version__ = version("gray")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["load_config"]
