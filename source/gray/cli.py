"""CLI dispatcher for project-owned training, validation and analysis stages."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from gray.core.config import load_config
from gray.experiment import artifact_dir
from gray.utils.io import write_json


def _load_entrypoint(
    config: dict[str, Any], stage: str
) -> Callable[[dict[str, Any]], Any]:
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
    project = config.get("project")
    entrypoints = project.get("entrypoints") if isinstance(project, Mapping) else None
    target = entrypoints.get(stage) if isinstance(entrypoints, Mapping) else None
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


_WORKFLOW_COMMANDS = ("train", "validate", "analyze")
_SDK_COMMANDS = {
    "infer": "Run model inference",
    "info": "Show model information",
    "speed": "Measure model inference speed",
    "inspect": "Validate a model manifest and ONNX session",
    "benchmark": "Benchmark a model with custom run counts",
}
_COMMAND_GROUP = "gray.commands"


def main(argv: list[str] | None = None) -> int | None:
    """Run a Gray workflow or an installed SDK command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else ""
    if command in _WORKFLOW_COMMANDS:
        return _run_workflow(arguments)

    extensions = entry_points(group=_COMMAND_GROUP)
    parser = argparse.ArgumentParser(
        prog="gray", description="Gray workflows and SDK commands."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in _WORKFLOW_COMMANDS:
        commands.add_parser(name, help=f"Run the configured {name} stage")
    installed = {extension.name for extension in extensions}
    installed.difference_update(_WORKFLOW_COMMANDS)
    for name in sorted(installed):
        description = (
            "Analyze skin images (Skin AI SDK)"
            if name == "skin_inference"
            else _SDK_COMMANDS.get(name, "Run an installed SDK command")
        )
        commands.add_parser(name, help=description)

    matches = [extension for extension in extensions if extension.name == command]
    if len(matches) > 1:
        parser.error(f"multiple installed SDKs register {command!r}")
    if matches:
        return matches[0].load()(arguments[1:])
    if command in _SDK_COMMANDS:
        parser.error(
            "install the gray-sdk package in the same Python environment as gray"
        )
    if command == "skin_inference":
        parser.error(
            "install the gray-skin-ai SDK in the same Python environment as gray"
        )
    if command.startswith("-") and command not in ("-h", "--help"):
        return _run_workflow(arguments)
    parser.parse_args(arguments)
    return None


def _run_workflow(argv: list[str]) -> None:
    """Run a configured training, validation, or analysis stage.

    If the project entry point returns a dictionary, it is saved as
    ``summary.json`` in the stage artifact directory before the result is
    printed.

    Args:
        argv: Command-line arguments excluding the executable name.

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
    parser.add_argument("command", choices=_WORKFLOW_COMMANDS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.override)
    stage = args.command
    entrypoint = _load_entrypoint(config, stage)
    result = entrypoint(config)
    if isinstance(result, dict):
        stage_dir = artifact_dir(config, stage, create=True)
        write_json(stage_dir / "summary.json", result)
    Console().print(
        Panel(
            Pretty(result),
            title="[bold green]GRAY COMPLETE[/]",
            border_style="bright_blue",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
