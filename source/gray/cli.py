"""CLI dispatcher for project-owned training, validation and analysis stages."""
from __future__ import annotations

import argparse
import importlib
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from gray.core.config import load_config
from gray.experiment import artifact_dir
from gray.utils.io import write_json


def _load_entrypoint(config: dict[str, Any], stage: str) -> Callable[[dict[str, Any]], Any]:
    """Import the project-owned callable configured for a workflow stage.

    Args:
        config: Experiment configuration containing
            ``project.entrypoints.<stage>``.
        stage: Workflow stage whose entry point should be loaded.

    Returns:
        The configured callable, which accepts the resolved configuration.

    Raises:
        ValueError: If the entry point is missing or not written as
            ``package.module:function``.
        ImportError: If the configured module cannot be imported.
        TypeError: If the referenced module attribute is not callable.
    """
    target = config.get("project", {}).get("entrypoints", {}).get(stage)
    if not isinstance(target, str) or ":" not in target:
        raise ValueError(
            f"config.project.entrypoints.{stage} must be 'package.module:function'; "
            "the Gray framework does not ship task-specific trainers"
        )
    module_name, function_name = target.rsplit(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"entrypoint is not callable: {target}")
    return function


def main(argv: list[str] | None = None) -> None:
    """Run a configured training, validation, or analysis stage.

    If the project entry point returns a dictionary, it is saved as
    ``summary.json`` in the stage artifact directory before the result is
    printed.

    Args:
        argv: Command-line arguments excluding the executable name. ``None``
            reads arguments from :data:`sys.argv`.

    Returns:
        None.

    Raises:
        SystemExit: If command-line arguments are invalid.
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration or configured entry point is invalid.
        ImportError: If the entry-point module cannot be imported.
        TypeError: If the entry-point attribute is not callable.
    """
    parser = argparse.ArgumentParser(prog="gray")
    parser.add_argument("command", choices=("train", "validate", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config, _ = load_config(args.config, args.override)
    stage = args.command
    entrypoint = _load_entrypoint(config, stage)
    result = entrypoint(config)
    if isinstance(result, dict):
        stage_dir = artifact_dir(config, stage, create=True)
        write_json(stage_dir / "summary.json", result)
    Console().print(Panel(Pretty(result), title="[bold green]GRAY COMPLETE[/]", border_style="bright_blue"))
if __name__ == "__main__": main()
