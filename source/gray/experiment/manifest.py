"""Structured experiment lineage manifests."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gray.core.provenance import model_manifest


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

    identity = model_manifest(dict(config), checkpoint)
    public_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    config_version = config.get("config_version") or identity["config_sha256"][:12]
    manifest = {
        "schema_version": 1,
        "experiment": {
            "id": identity["experiment_id"],
            "run_id": selected_run_id.strip() if isinstance(selected_run_id, str) else None,
        },
        "dataset": {
            "version": identity["data_version"],
            "label_schema": identity["label_schema"],
        },
        "code": {"git_commit": identity["git_commit"]},
        "config": {
            "version": str(config_version),
            "sha256": identity["config_sha256"],
            "resolved": _redact_config(public_config),
        },
        "training": {
            "seed": identity["seed"],
            "fold": identity["fold"],
        },
        "model": {
            "version": identity["model_version"],
            "architecture": identity["architecture"],
            "checkpoint": identity["checkpoint"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "status": identity["status"],
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
