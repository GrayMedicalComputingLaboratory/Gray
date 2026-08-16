"""PyTorch DataLoader worker random-state initialization."""
from __future__ import annotations

import random

import numpy as np


def seed_worker(worker_id: int) -> None:
    """Seed one PyTorch DataLoader worker from its PyTorch-assigned seed."""
    if worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    try:
        import torch
    except ImportError:
        return
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
