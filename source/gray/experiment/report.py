"""Unified training configuration and experiment lineage reporting."""
from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


_PACKAGES = (
    ("Gray", "gray"),
    ("NumPy", "numpy"),
    ("PyTorch", "torch"),
    ("TorchVision", "torchvision"),
    ("Hydra", "hydra-core"),
    ("OmegaConf", "omegaconf"),
    ("SimpleITK", "SimpleITK"),
    ("scikit-learn", "scikit-learn"),
    ("Matplotlib", "matplotlib"),
    ("PyYAML", "PyYAML"),
    ("Rich", "rich"),
    ("ClearML", "clearml"),
)
_SENSITIVE_NAMES = {
    "access_key",
    "api_key",
    "client_secret",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
}


def experiment_report(
    config: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    console: Console | None = None,
) -> None:
    """Print configuration, environment, and optional experiment lineage.

    Nested configuration keys are flattened with dot-separated paths. Values
    whose final key identifies a password, token, secret, or access key are
    redacted. Package versions are read from installed distribution metadata,
    so optional heavyweight libraries are not imported. When ``manifest`` is
    provided, the report also includes the experiment lineage and identities.

    Args:
        config: Resolved experiment configuration mapping to display.
        manifest: Optional mapping returned by
            :func:`gray.experiment.experiment_manifest`.
        console: Rich console receiving the report. ``None`` uses a new console
            connected to the current standard output.

    Returns:
        None.

    Raises:
        TypeError: If ``config``, ``manifest``, or a required manifest section
            is not a mapping.
        KeyError: If a required manifest lineage field is absent.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if manifest is not None and not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping or None")

    output = console if console is not None else Console()
    output.print(_config_table(config))
    output.print(_environment_table())
    if manifest is None:
        return

    _print_lineage(manifest, output)


def _print_lineage(manifest: Mapping[str, Any], console: Console) -> None:
    experiment = _section(manifest, "experiment")
    dataset = _section(manifest, "dataset")
    code = _section(manifest, "code")
    config = _section(manifest, "config")
    _section(manifest, "training")
    model = _section(manifest, "model")
    evaluation = manifest.get("evaluation")
    if evaluation is not None and not isinstance(evaluation, Mapping):
        raise TypeError("manifest.evaluation must be a mapping or None")

    tree = Tree(f"[bold cyan]Experiment[/] [white]{escape(str(experiment['id']))}[/]")
    inputs = tree.add("[bold]Inputs[/]")
    inputs.add(f"[green]Dataset[/] {escape(str(dataset['version']))}")
    commit = code.get("git_commit")
    inputs.add(f"[magenta]Git Code[/] {escape(str(commit)[:12]) if commit else 'unavailable'}")
    inputs.add(f"[yellow]Config[/] {escape(str(config['version']))}")
    run_id = experiment.get("run_id") or "local run"
    run = tree.add(f"[bold blue]Training[/] {escape(str(run_id))}")
    run.add(f"[bold green]Model[/] {escape(str(model['version']))} ({escape(str(model['status']))})")
    run.add(f"[bold yellow]Evaluation[/] {escape(_evaluation_status(evaluation))}")

    console.print(tree)
    console.print(_identity_table(manifest))


def _config_table(config: Mapping[str, Any]) -> Table:
    table = Table(title="Training Configuration", border_style="blue", header_style="bold cyan")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    rows = _flatten_config(config)
    if not rows:
        table.add_row(Text("-"), Text("<empty>", style="dim"))
        return table
    for name, value in rows:
        table.add_row(Text(name), Text(value))
    return table


def _environment_table() -> Table:
    table = Table(title="Environment Versions", border_style="green", header_style="bold green")
    table.add_column("Component", style="green", no_wrap=True)
    table.add_column("Version")
    table.add_row("Python", platform.python_version())
    table.add_row("Operating System", f"{platform.system()} {platform.release()}")
    table.add_row("Architecture", platform.machine() or "unknown")
    table.add_row("Executable", sys.executable)
    for display_name, distribution_name in _PACKAGES:
        table.add_row(display_name, _package_version(distribution_name))
    return table


def _flatten_config(config: Mapping[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if _is_sensitive(name):
            rows.append((name, "<redacted>"))
        elif isinstance(value, Mapping) and value:
            rows.extend(_flatten_config(value, name))
        else:
            rows.append((name, _format_value(value)))
    return rows


def _is_sensitive(path: str) -> bool:
    name = path.rsplit(".", 1)[-1].lower().replace("-", "_")
    return name in _SENSITIVE_NAMES or any(name.endswith(f"_{suffix}") for suffix in _SENSITIVE_NAMES)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return repr(value)
    if value is None:
        return "None"
    return str(value)


def _package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not installed"


def _section(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = manifest[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"manifest.{name} must be a mapping")
    return value


def _evaluation_status(evaluation: Mapping[str, Any] | None) -> str:
    if evaluation is None:
        return "pending"
    status = evaluation.get("status")
    return str(status) if status is not None else "complete"


def _identity_table(manifest: Mapping[str, Any]) -> Table:
    experiment = _section(manifest, "experiment")
    dataset = _section(manifest, "dataset")
    code = _section(manifest, "code")
    config = _section(manifest, "config")
    model = _section(manifest, "model")
    evaluation = manifest.get("evaluation")
    table = Table(title="Experiment Identity", border_style="blue", header_style="bold cyan")
    table.add_column("Object", style="cyan")
    table.add_column("Identity", overflow="fold")
    table.add_row("Experiment", str(experiment["id"]))
    table.add_row("Run", str(experiment.get("run_id") or "local run"))
    table.add_row("Dataset", str(dataset["version"]))
    table.add_row("Git Commit", str(code.get("git_commit") or "unavailable"))
    table.add_row("Config", f"{config['version']} / {config['sha256']}")
    table.add_row("Model", f"{model['version']} / {model['checkpoint_sha256']}")
    table.add_row("Evaluation", json.dumps(evaluation, ensure_ascii=False) if evaluation is not None else "pending")
    return table
