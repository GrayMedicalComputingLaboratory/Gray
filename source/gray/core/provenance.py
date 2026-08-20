"""Immutable model-artifact identity and file hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from collections.abc import Mapping
from typing import Any


_SENSITIVE_CONFIG_NAMES = {
    "access_key",
    "api_key",
    "client_secret",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
}


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


def experiment_manifest(
    config: Mapping[str, Any],
    checkpoint: Path,
    *,
    evaluation: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a structured lineage manifest for one completed training run.

    The returned hierarchy explicitly links the experiment to its dataset,
    source revision, resolved configuration, training run, model artifact, and
    optional evaluation result. It is suitable for JSON persistence or upload
    to an experiment tracker such as ClearML.

    Args:
        config: Resolved experiment configuration containing ``experiment_id``,
            ``data``, ``model`` and ``train`` sections.
        checkpoint: Trained model checkpoint. Relative paths are resolved from
            the optional internal ``_config_dir`` field.
        evaluation: Optional JSON-serializable evaluation summary or metrics.
        run_id: Optional unique training execution identifier, such as a ClearML
            Task ID. When omitted, ``config.run_id`` is used when available.

    Returns:
        A JSON-serializable nested dictionary with a stable ``schema_version``
        and explicit experiment lineage sections.

    Raises:
        KeyError: If required configuration fields are missing.
        TypeError: If inputs are invalid or public values are not JSON serializable.
        ValueError: If an explicitly supplied ``run_id`` is empty.
        FileNotFoundError: If the checkpoint does not exist.
        OSError: If the checkpoint cannot be read.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if evaluation is not None and not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping or None")
    selected_run_id = run_id if run_id is not None else config.get("run_id")
    if selected_run_id is not None and not isinstance(selected_run_id, str):
        raise TypeError("run_id must be a string or None")
    if isinstance(selected_run_id, str) and not selected_run_id.strip():
        raise ValueError("run_id must be a non-empty string or None")

    legacy = model_manifest(dict(config), checkpoint)
    public_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    config_version = config.get("config_version") or legacy["config_sha256"][:12]
    manifest = {
        "schema_version": 1,
        "experiment": {
            "id": legacy["experiment_id"],
            "run_id": selected_run_id.strip() if isinstance(selected_run_id, str) else None,
        },
        "dataset": {
            "version": legacy["data_version"],
            "label_schema": legacy["label_schema"],
        },
        "code": {"git_commit": legacy["git_commit"]},
        "config": {
            "version": str(config_version),
            "sha256": legacy["config_sha256"],
            "resolved": _redact_config(public_config),
        },
        "training": {
            "seed": legacy["seed"],
            "fold": legacy["fold"],
        },
        "model": {
            "version": legacy["model_version"],
            "architecture": legacy["architecture"],
            "checkpoint": legacy["checkpoint"],
            "checkpoint_sha256": legacy["checkpoint_sha256"],
            "status": legacy["status"],
        },
        "evaluation": dict(evaluation) if evaluation is not None else None,
    }
    json.dumps(manifest, sort_keys=True)
    return manifest


def _redact_config(value: Any, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    if normalized in _SENSITIVE_CONFIG_NAMES or any(
        normalized.endswith(f"_{suffix}") for suffix in _SENSITIVE_CONFIG_NAMES
    ):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(child_key): _redact_config(child_value, str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_config(item) for item in value]
    return value
