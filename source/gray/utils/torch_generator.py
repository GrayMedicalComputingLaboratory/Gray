"""Seeded PyTorch generator construction."""
from __future__ import annotations

from typing import Any


def torch_generator(seed: int) -> Any:
    """Create a reproducibly seeded PyTorch random generator.

    Args:
        seed: Integer seed passed to ``torch.Generator.manual_seed``.

    Returns:
        A seeded ``torch.Generator`` suitable for the DataLoader ``generator``
        argument.

    Raises:
        RuntimeError: If PyTorch is not installed or rejects the seed value.
        TypeError: If PyTorch rejects the supplied seed type.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch_generator requires PyTorch") from error
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
