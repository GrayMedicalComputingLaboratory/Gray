"""Global training random-state initialization."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    The environment variable is exported for child processes, but changing it
    here cannot change the already-running interpreter's hash randomization.

    Args:
        seed: Integer seed passed to every available random number generator.
        deterministic: Enable deterministic PyTorch algorithms and deterministic
            cuDNN behavior when PyTorch is installed.

    Returns:
        None. Missing PyTorch is treated as an optional dependency.

    Raises:
        TypeError: If a runtime rejects the supplied seed type.
        ValueError: If a runtime rejects the supplied seed value.
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
