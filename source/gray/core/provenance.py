"""Immutable model-artifact identity and file hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _find_repo_root(*locations: Path) -> Path | None:
    """Find the nearest Git worktree containing any supplied location.

    Args:
        *locations: Files or directories from which to search upward.

    Returns:
        The first directory containing ``.git``, or ``None`` when no worktree
        can be found.
    """
    for location in locations:
        current = location if location.is_dir() else location.parent
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                return candidate
    return None


def sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file without loading it all at once.

    Args:
        path: File to hash.

    Returns:
        The lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        IsADirectoryError: If ``path`` is a directory.
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Build a reproducibility manifest for a trained model checkpoint.

    The manifest records model and data identity, a stable configuration hash,
    the current Git commit when discoverable, and the checkpoint checksum.

    Args:
        config: Resolved experiment configuration with model, data, and training
            metadata. Relative checkpoints use its optional ``_config_dir``.
        checkpoint: Model checkpoint file to identify and hash.

    Returns:
        A JSON-serializable dictionary describing the candidate model artifact.

    Raises:
        KeyError: If required configuration fields are missing.
        FileNotFoundError: If the checkpoint does not exist.
        OSError: If the checkpoint cannot be read.
        TypeError: If public configuration values are not JSON-serializable.
    """
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_absolute() and config.get("_config_dir"):
        checkpoint = Path(config["_config_dir"]) / checkpoint
    checkpoint = checkpoint.resolve()
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    repo_root = _find_repo_root(checkpoint, Path(config.get("_config_dir", ".")))
    try:
        if repo_root is None:
            raise OSError("no Git repository found")
        git_commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
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
