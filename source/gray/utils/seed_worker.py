"""PyTorch DataLoader worker random-state initialization."""
from __future__ import annotations

import random

import numpy as np


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker.

    Args:
        worker_id: Non-negative worker identifier supplied by DataLoader. The
            random seed itself comes from :func:`torch.initial_seed`.

    Returns:
        None. Missing PyTorch is treated as an optional dependency.

    Raises:
        TypeError: If ``worker_id`` cannot be compared with an integer.
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
