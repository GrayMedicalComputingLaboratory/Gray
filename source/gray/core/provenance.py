"""Immutable model-artifact identity and file hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Build the minimum model identity record stored beside a checkpoint."""
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
