"""Training runtime reproducibility controls."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: Integer seed passed to every available random number generator.
        deterministic: Enable deterministic PyTorch and cuDNN behavior when
            PyTorch is installed.

    Returns:
        None. Missing PyTorch is treated as an optional dependency.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker.

    Args:
        worker_id: Non-negative worker identifier supplied by DataLoader.

    Returns:
        None. Missing PyTorch is treated as an optional dependency.

    Raises:
        ValueError: If ``worker_id`` is negative.
    """
    if worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    try:
        import torch
    except ImportError:
        return
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def torch_generator(seed: int) -> Any:
    """Create a reproducibly seeded PyTorch random generator.

    Args:
        seed: Integer seed passed to ``torch.Generator.manual_seed``.

    Returns:
        A seeded ``torch.Generator`` suitable for a DataLoader.

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
