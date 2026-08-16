"""YAML configuration loading with explicit, portable path resolution."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # Minimal template remains runnable before dependency install.
    yaml = None


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a mapping config and annotate it with its source directory."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"config not found: {source}")
    with source.open(encoding="utf-8") as handle:
        config = (yaml.safe_load(handle) if yaml is not None else json.load(handle)) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a YAML mapping")
    result = deepcopy(config)
    experiment_id = source.stem
    configured = result.get("experiment_id")
    if configured not in (None, experiment_id):
        raise ValueError(f"experiment_id must be derived from config filename: expected {experiment_id}, got {configured}")
    result["experiment_id"] = experiment_id
    result["_config_dir"] = str(source.parent)
    return result, source


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve relative paths from the configuration location, never the CWD."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(config["_config_dir"]) / path).resolve()


def artifact_dir(config: dict[str, Any], stage: str, create: bool = False) -> Path:
    """Return ``output_root/<experiment_id>/<stage>`` for one isolated experiment."""
    if not stage or Path(stage).is_absolute() or ".." in Path(stage).parts:
        raise ValueError(f"invalid artifact stage: {stage!r}")
    output_root = resolve_path(config, config["project"]["output_root"])
    path = output_root / config["experiment_id"] / stage
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
