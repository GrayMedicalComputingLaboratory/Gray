"""Validation and parameter helpers for Optuna configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna


@dataclass(frozen=True)
class OptunaSettings:
    """Validated, execution-ready values from the ``optuna`` config block."""

    options: dict[str, Any]
    search_space: dict[str, dict[str, Any]]
    direction: str
    objective_key: str
    n_trials: int
    study_name: str
    seed: int
    sampler_name: str
    pruner_name: str


def validate_options(config: dict[str, Any]) -> OptunaSettings:
    """Validate the user-owned Optuna configuration before creating a study."""
    options = config.get("optuna")
    if not isinstance(options, dict):
        raise ValueError("optuna must be a YAML mapping")
    search_space = options.get("search_space")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("optuna.search_space must be a non-empty mapping")
    normalized_space: dict[str, dict[str, Any]] = {}
    for dotted_key, specification in search_space.items():
        if not isinstance(dotted_key, str) or not dotted_key or any(not part for part in dotted_key.split(".")):
            raise ValueError("each optuna.search_space key must be a non-empty dotted path")
        if not isinstance(specification, dict):
            raise ValueError("each optuna.search_space entry must be a dotted key and mapping")
        kind = specification.get("type")
        if kind not in {"float", "int", "categorical"}:
            raise ValueError(f"optuna.search_space.{dotted_key}.type must be float, int or categorical")
        if kind in {"float", "int"}:
            for field in ("low", "high"):
                if field not in specification:
                    raise ValueError(f"optuna.search_space.{dotted_key}.{field} is required")
        if kind == "categorical" and (not isinstance(specification.get("choices"), list) or not specification["choices"]):
            raise ValueError(f"optuna.search_space.{dotted_key}.choices must be non-empty")
        normalized_space[dotted_key] = specification
    direction = options.get("direction", "maximize")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("optuna.direction must be 'maximize' or 'minimize'")
    objective_key = options.get("objective_key")
    if not isinstance(objective_key, str) or not objective_key or any(not part for part in objective_key.split(".")):
        raise ValueError("optuna.objective_key must be a dotted result key, for example valid.f1_macro")
    n_trials = int(options.get("n_trials", 20))
    if n_trials < 1:
        raise ValueError("optuna.n_trials must be positive")
    sampler_name = str(options.get("sampler", "tpe")).lower()
    pruner_name = str(options.get("pruner", "median")).lower()
    if sampler_name not in {"tpe", "random"} or pruner_name not in {"median", "none"}:
        raise ValueError("optuna.sampler must be tpe/random and optuna.pruner must be median/none")
    return OptunaSettings(
        options=options,
        search_space=normalized_space,
        direction=direction,
        objective_key=objective_key,
        n_trials=n_trials,
        study_name=str(options.get("study_name", f"{config['experiment_id']}_study")),
        seed=int(options.get("seed", config.get("runtime", {}).get("seed", 42))),
        sampler_name=sampler_name,
        pruner_name=pruner_name,
    )


def resolve_storage(config: dict[str, Any], output_dir: Path, value: Any) -> str:
    """Resolve a configured SQLAlchemy URL or SQLite path."""
    if value is None:
        return f"sqlite:///{(output_dir / 'study.db').resolve().as_posix()}"
    if isinstance(value, str) and "://" not in value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(config["_config_dir"]) / path
        return f"sqlite:///{path.resolve().as_posix()}"
    if not isinstance(value, str):
        raise ValueError("optuna.storage must be a SQLAlchemy URL or a SQLite file path")
    return value


def suggest_parameter(trial: optuna.Trial, dotted_key: str, specification: dict[str, Any]) -> Any:
    """Suggest one validated search-space parameter."""
    kind = specification["type"]
    if kind == "float":
        return trial.suggest_float(dotted_key, float(specification["low"]), float(specification["high"]), log=bool(specification.get("log", False)), step=specification.get("step"))
    if kind == "int":
        return trial.suggest_int(dotted_key, int(specification["low"]), int(specification["high"]), log=bool(specification.get("log", False)), step=int(specification.get("step", 1)))
    return trial.suggest_categorical(dotted_key, specification["choices"])


def set_dotted_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set an existing dotted config key, rejecting unknown paths."""
    cursor: dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise KeyError(f"search-space key does not exist in config: {dotted_key}")
        cursor = child
    if parts[-1] not in cursor:
        raise KeyError(f"search-space key does not exist in config: {dotted_key}")
    cursor[parts[-1]] = value
