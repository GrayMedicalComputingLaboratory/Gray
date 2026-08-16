"""YAML configuration loading with explicit, portable path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> tuple[dict[str, Any], Path]:
    """Compose one experiment YAML through Hydra without config-group composition."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"config not found: {source}")
    raw = OmegaConf.load(source)
    if not isinstance(raw, DictConfig):
        raise ValueError("configuration root must be a YAML mapping")
    if "defaults" in raw:
        raise ValueError("Gray accepts one self-contained experiment YAML; Hydra defaults are not supported")
    with initialize_config_dir(version_base=None, config_dir=str(source.parent)):
        composed = compose(config_name=source.stem, overrides=list(overrides))
    result = OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
    if not isinstance(result, dict):
        raise ValueError("configuration root must resolve to a mapping")
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
