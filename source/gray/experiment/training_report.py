"""Rich training configuration and environment reporting."""
from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


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


def training_report(config: Mapping[str, Any], *, console: Console | None = None) -> None:
    """Print training configuration and runtime versions as Rich tables.

    Nested configuration keys are flattened with dot-separated paths. Values
    whose final key identifies a password, token, secret, or access key are
    redacted. Package versions are read from installed distribution metadata,
    so optional heavyweight libraries are not imported.

    Args:
        config: Resolved training configuration mapping to display.
        console: Rich console receiving the report. ``None`` uses a new console
            connected to the current standard output.

    Returns:
        None.

    Raises:
        TypeError: If ``config`` is not a mapping.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    output = console if console is not None else Console()
    output.print(_config_table(config))
    output.print(_environment_table())


def _config_table(config: Mapping[str, Any]) -> Table:
    table = Table(title="Training Configuration", border_style="blue", header_style="bold cyan")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    rows = list(_flatten_config(config))
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
