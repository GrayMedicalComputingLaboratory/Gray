"""Visual experiment lineage reporting."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree


def experiment_report(manifest: Mapping[str, Any], *, console: Console | None = None) -> None:
    """Print the lineage and identities of one experiment manifest.

    Args:
        manifest: Mapping returned by :func:`gray.core.experiment_manifest`.
        console: Rich console receiving the report. ``None`` uses standard output.

    Returns:
        None.

    Raises:
        TypeError: If ``manifest`` or a required section is not a mapping.
        KeyError: If a required lineage field is absent.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    experiment = _section(manifest, "experiment")
    dataset = _section(manifest, "dataset")
    code = _section(manifest, "code")
    config = _section(manifest, "config")
    training = _section(manifest, "training")
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

    output = console if console is not None else Console()
    output.print(tree)
    output.print(_identity_table(manifest))


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
