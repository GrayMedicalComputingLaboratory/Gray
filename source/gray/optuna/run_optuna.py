"""Hydra-configured Optuna study execution with Rich terminal reporting."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import inspect
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import optuna
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gray.core.config import artifact_dir
from gray.utils.io import write_json


def run_optuna(config: dict[str, Any], train_once: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """Optimize one project's ``train_once`` callable from its single experiment YAML."""
    options = config.get("optuna")
    if not isinstance(options, dict):
        raise ValueError("optuna must be a YAML mapping")
    search_space = options.get("search_space")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("optuna.search_space must be a non-empty mapping")
    direction = options.get("direction", "maximize")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("optuna.direction must be 'maximize' or 'minimize'")
    objective_key = options.get("objective_key")
    if not isinstance(objective_key, str) or not objective_key:
        raise ValueError("optuna.objective_key must be a dotted result key, for example valid.f1_macro")
    n_trials = int(options.get("n_trials", 20))
    if n_trials < 1:
        raise ValueError("optuna.n_trials must be positive")
    output_dir = artifact_dir(config, "optuna", create=True)
    trial_dir = output_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    study_name = str(options.get("study_name", f"{config['experiment_id']}_study"))
    storage = options.get("storage")
    if storage is None:
        storage = f"sqlite:///{(output_dir / 'study.db').resolve().as_posix()}"
    elif isinstance(storage, str) and "://" not in storage:
        storage_path = Path(storage).expanduser()
        if not storage_path.is_absolute():
            storage_path = Path(config["_config_dir"]) / storage_path
        storage = f"sqlite:///{storage_path.resolve().as_posix()}"
    if not isinstance(storage, str):
        raise ValueError("optuna.storage must be a SQLAlchemy URL or a SQLite file path")
    seed = int(options.get("seed", config.get("runtime", {}).get("seed", 42)))
    sampler_name = str(options.get("sampler", "tpe")).lower()
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True) if sampler_name == "tpe" else optuna.samplers.RandomSampler(seed=seed)
    pruner_name = str(options.get("pruner", "median")).lower()
    pruner = optuna.pruners.MedianPruner() if pruner_name == "median" else optuna.pruners.NopPruner()
    if sampler_name not in {"tpe", "random"} or pruner_name not in {"median", "none"}:
        raise ValueError("optuna.sampler must be tpe/random and optuna.pruner must be median/none")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(options.get("resume", True)),
    )
    console = Console()
    console.print(Panel(
        f"[bold bright_blue]Study:[/] [bold]{study_name}[/]\n"
        f"[bold bright_blue]Objective:[/] [bold]{direction} {objective_key}[/]\n"
        f"[bold bright_blue]Trials:[/] [bold]{n_trials}[/]\n"
        f"[bold bright_blue]Storage:[/] {storage}",
        title="[bold gold1]GRAY OPTUNA SEARCH[/]",
        border_style="bright_blue",
    ))
    accepts_trial = "trial" in inspect.signature(train_once).parameters

    def objective(trial: optuna.Trial) -> float:
        trial_config = deepcopy(config)
        trial_config["experiment_id"] = f"{config['experiment_id']}__trial_{trial.number:04d}"
        trial_config["optuna"] = {**options, "enabled": False, "trial_number": trial.number}
        for dotted_key, specification in search_space.items():
            if not isinstance(dotted_key, str) or not isinstance(specification, dict):
                raise ValueError("each optuna.search_space entry must be a dotted key and mapping")
            kind = specification.get("type")
            if kind == "float":
                value = trial.suggest_float(dotted_key, float(specification["low"]), float(specification["high"]), log=bool(specification.get("log", False)), step=specification.get("step"))
            elif kind == "int":
                value = trial.suggest_int(dotted_key, int(specification["low"]), int(specification["high"]), log=bool(specification.get("log", False)), step=int(specification.get("step", 1)))
            elif kind == "categorical":
                choices = specification.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ValueError(f"optuna.search_space.{dotted_key}.choices must be non-empty")
                value = trial.suggest_categorical(dotted_key, choices)
            else:
                raise ValueError(f"optuna.search_space.{dotted_key}.type must be float, int or categorical")
            cursor: dict[str, Any] = trial_config
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                child = cursor.get(part)
                if not isinstance(child, dict):
                    raise KeyError(f"search-space key does not exist in config: {dotted_key}")
                cursor = child
            if parts[-1] not in cursor:
                raise KeyError(f"search-space key does not exist in config: {dotted_key}")
            cursor[parts[-1]] = value
        snapshot = trial_dir / f"trial_{trial.number:04d}.yaml"
        OmegaConf.save(config=OmegaConf.create(trial_config), f=snapshot)
        console.print(f"[bold bright_blue]TRIAL {trial.number:04d} START[/]  {trial.params}")
        started = perf_counter()
        try:
            result = train_once(trial_config, trial=trial) if accepts_trial else train_once(trial_config)
            if not isinstance(result, dict):
                raise TypeError("train_once must return a metrics dictionary")
            value: Any = result
            for key in objective_key.split("."):
                if not isinstance(value, dict) or key not in value:
                    raise KeyError(f"train_once result lacks objective_key: {objective_key}")
                value = value[key]
            objective_value = float(value)
            if not np.isfinite(objective_value):
                raise ValueError(f"objective {objective_key} must be finite")
            duration = perf_counter() - started
            write_json(trial_dir / f"trial_{trial.number:04d}.json", {"trial": trial.number, "params": trial.params, "objective": objective_value, "duration_seconds": duration, "result": result})
            prior_values = [item.value for item in study.trials if item.state == optuna.trial.TrialState.COMPLETE and item.value is not None]
            current_best = (max([objective_value, *prior_values]) if direction == "maximize" else min([objective_value, *prior_values]))
            table = Table(title="[bold green]TRIAL COMPLETE[/]", border_style="bright_blue", header_style="bold bright_blue")
            for column in ("Trial", "Status", "Objective", "Best", "Seconds", "Parameters"):
                table.add_column(column)
            table.add_row(str(trial.number), "[bold green]COMPLETE[/]", f"[bold green]{objective_value:.6f}[/]", f"[bold gold1]{current_best:.6f}[/]", f"{duration:.1f}", str(trial.params))
            console.print(table)
            return objective_value
        except optuna.TrialPruned:
            console.print(f"[bold red]TRIAL {trial.number:04d} PRUNED[/]")
            raise
        except Exception as error:
            console.print(f"[bold red]TRIAL {trial.number:04d} FAILED:[/] {error}")
            raise

    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Optuna completed without a successful trial")
    best = study.best_trial
    best_config = deepcopy(config)
    for dotted_key, value in best.params.items():
        cursor: dict[str, Any] = best_config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    best_config["optuna"] = {**options, "enabled": False, "best_trial": best.number}
    OmegaConf.save(config=OmegaConf.create(best_config), f=output_dir / "best_params.yaml")
    summary: dict[str, Any] = {
        "study_name": study_name,
        "direction": direction,
        "objective_key": objective_key,
        "best_trial": best.number,
        "best_value": float(best.value),
        "best_params": best.params,
        "completed_trials": len(completed),
        "total_trials": len(study.trials),
        "storage": storage,
    }
    console.print(Panel(
        f"[bold gold1]BEST TRIAL[/]  [bold]{best.number}[/]\n"
        f"[bold gold1]{objective_key}:[/] [bold green]{best.value:.6f}[/]\n"
        f"[bold gold1]PARAMETERS:[/] {best.params}",
        title="[bold gold1]GRAY OPTUNA RESULT[/]",
        border_style="gold1",
    ))
    if bool(options.get("final_train", False)):
        console.print("[bold bright_blue]FINAL TRAINING WITH BEST PARAMETERS[/]")
        final_result = train_once(best_config)
        if not isinstance(final_result, dict):
            raise TypeError("final train_once result must be a metrics dictionary")
        summary["final_train"] = final_result
        console.print("[bold green]FINAL TRAINING COMPLETE[/]")
    write_json(output_dir / "study_summary.json", summary)
    return summary
