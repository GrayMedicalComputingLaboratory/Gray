"""YAML configuration loading with explicit, portable path resolution."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> tuple[dict[str, Any], Path]:
    """Load and resolve one self-contained experiment YAML file with Hydra.

    The configuration filename determines ``experiment_id``. Hydra overrides
    are applied before interpolation is resolved, and ``_config_dir`` is added
    to the returned mapping for portable relative-path resolution.

    Args:
        path: Path to the experiment YAML file.
        overrides: Hydra override expressions such as ``train.seed=42``.

    Returns:
        A pair containing the resolved configuration dictionary and the
        absolute path to its source file.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
        ValueError: If the YAML root is not a mapping, uses Hydra ``defaults``,
            resolves to an invalid value, or defines a conflicting
            ``experiment_id``.
        omegaconf.errors.OmegaConfBaseException: If parsing, an override, an
            interpolation, or a mandatory value is invalid.
    """
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
    if "_config_dir" in composed:
        raise ValueError("_config_dir is reserved and cannot be configured")
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


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the configuration file directory.

    Args:
        config: Configuration returned by :func:`load_config`. It must contain
            the internal ``_config_dir`` field.
        value: Absolute path or path relative to the configuration directory.

    Returns:
        The expanded input path when absolute, otherwise an absolute path
        resolved from ``config["_config_dir"]``.

    Raises:
        KeyError: If a relative path is supplied and ``_config_dir`` is absent.
        TypeError: If ``value`` is not path-like.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(config["_config_dir"]) / path).resolve()


__all__ = ["load_config", "resolve_path"]
