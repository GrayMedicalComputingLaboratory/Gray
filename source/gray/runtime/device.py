"""Explicit runtime-device validation."""
from __future__ import annotations


def resolve_device(value: str | int) -> str:
    """Validate and normalize an explicitly selected runtime device.

    Args:
        value: ``"cpu"``, ``"cuda"``, ``"cuda:N"``, a numeric string, or a
            non-negative integer GPU index.

    Returns:
        ``"cpu"`` or a normalized CUDA identifier such as ``"cuda:0"``.

    Raises:
        ValueError: If ``value`` is not a supported device specification.
        RuntimeError: If CUDA is requested but PyTorch is unavailable, CUDA is
            disabled, or the requested GPU index does not exist.
    """
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        value = "cuda:0"
    if isinstance(value, str) and value.startswith("cuda:"):
        value = value.removeprefix("cuda:")
    if isinstance(value, bool):
        raise ValueError("device must be 'cpu', 'cuda', 'cuda:N', or a non-negative GPU index")
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        index = int(value)
        if index < 0:
            raise ValueError("GPU index must be non-negative")
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("CUDA device requested but PyTorch is unavailable") from error
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{index} is unavailable")
        return f"cuda:{index}"
    raise ValueError("device must be 'cpu', 'cuda', 'cuda:N', or a non-negative GPU index")
