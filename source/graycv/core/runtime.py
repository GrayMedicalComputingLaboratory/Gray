"""Runtime validation and immutable artifact identity helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .config import artifact_dir


def resolve_device(value: str | int) -> str:
    """Validate an explicit CPU or numbered CUDA device; never silently fall back."""
    if value == "cpu":
        return "cpu"
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
    raise ValueError("device must be 'cpu' or a non-negative GPU index")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Build the minimum commercial model identity record beside a checkpoint."""
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "experiment_id": config["experiment_id"],
        "model_version": config["model"]["model_version"],
        "architecture": config["model"]["architecture"],
        "data_version": config["data"]["data_version"],
        "label_schema": config["data"]["label_schema"],
        "seed": config["train"]["seed"],
        "fold": config["train"].get("fold", "single"),
        "config_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest(),
        "git_commit": git_commit,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "status": "candidate",
    }


def stage_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Enumerate standard stages without creating them until a stage writes."""
    return {stage: artifact_dir(config, stage) for stage in ("prepare", "features", "logs", "models", "oof", "scores", "validation", "analysis")}
