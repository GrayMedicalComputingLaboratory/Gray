"""Deterministic nnU-Net artifact paths and process environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from gray.experiment.artifacts import artifact_dir


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    """Read the supported nnU-Net settings from either config namespace."""
    value = config.get("nnunet")
    if isinstance(value, dict):
        return value
    value = config.get("preprocessing", {}).get("nnunet")
    if not isinstance(value, dict):
        raise KeyError("configuration must contain nnunet or preprocessing.nnunet")
    return value


def dataset_dir(config: dict[str, Any]) -> Path:
    """Return the experiment-owned raw DatasetXXX directory."""
    settings = _settings(config)
    return artifact_dir(config, "nnunet_raw") / (
        f"Dataset{int(settings['dataset_id']):03d}_{settings['dataset_name']}"
    )


def model_dir(config: dict[str, Any]) -> Path:
    """Return the configured trainer/plans/configuration model directory."""
    settings = _settings(config)
    training = config.get("training", settings)
    return (
        artifact_dir(config, "nnunet_results")
        / dataset_dir(config).name
        / f"{training['trainer']}__{training['plans']}__{training['configuration']}"
    )


def nnunet_environment(
    config: dict[str, Any],
    *,
    visible_gpu: str | None = None,
    create_results: bool = False,
) -> dict[str, str]:
    """Build nnU-Net environment variables without mutating the caller."""
    environment = os.environ.copy()
    environment["nnUNet_raw"] = str(artifact_dir(config, "nnunet_raw"))
    environment["nnUNet_preprocessed"] = str(
        artifact_dir(config, "nnunet_preprocessed")
    )
    environment["nnUNet_results"] = str(
        artifact_dir(config, "nnunet_results", create=create_results)
    )
    if visible_gpu is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = visible_gpu
    return environment
