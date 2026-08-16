"""Seeded PyTorch generator construction."""
from __future__ import annotations

from typing import Any


def torch_generator(seed: int) -> Any:
    """Create a seeded generator for ``DataLoader(generator=...)``."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch_generator requires PyTorch") from error
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
