"""Hydra-configured Optuna study execution with Rich terminal reporting."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import inspect
from time import perf_counter
from typing import Any

import numpy as np
import optuna
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gray.core.config import artifact_dir
from gray.optuna.validation import resolve_storage, set_dotted_value, suggest_parameter, validate_options
from gray.utils.io import write_json


def run_optuna(config: dict[str, Any], train_once: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """Optimize one project's ``train_once`` callable from its single experiment YAML."""
    settings = validate_options(config)
    options = settings.options
    search_space = settings.search_space
    direction = settings.direction
    objective_key = settings.objective_key
    n_trials = settings.n_trials
    output_dir = artifact_dir(config, "optuna", create=True)
    trial_dir = output_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    study_name = settings.study_name
    storage = resolve_storage(config, output_dir, options.get("storage"))
    sampler = optuna.samplers.TPESampler(seed=settings.seed, multivariate=True) if settings.sampler_name == "tpe" else optuna.samplers.RandomSampler(seed=settings.seed)
    pruner = optuna.pruners.MedianPruner() if settings.pruner_name == "median" else optuna.pruners.NopPruner()
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
            value = suggest_parameter(trial, dotted_key, specification)
            set_dotted_value(trial_config, dotted_key, value)
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
        set_dotted_value(best_config, dotted_key, value)
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
