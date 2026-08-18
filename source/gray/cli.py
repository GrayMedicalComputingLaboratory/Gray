"""CLI dispatcher for project-owned training, validation and analysis stages."""
from __future__ import annotations

import argparse
import importlib
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from gray.core.config import artifact_dir, load_config
from gray.utils.artifacts import write_json


def _load_entrypoint(config: dict[str, Any], stage: str) -> Callable[[dict[str, Any]], Any]:
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
