"""Internal experiment configuration normalization."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gray.core.config import resolve_path


_SENSITIVE_NAMES = {
    "access_key",
    "api_key",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "secret",
    "secret_key",
    "signing_key",
    "token",
}


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return configuration fields intended for persistence or reporting."""
    return {str(key): value for key, value in config.items() if not str(key).startswith("_")}


def redact_config(value: Any, key: str = "") -> Any:
    """Recursively replace credential-like configuration values."""
    normalized = key.lower().replace("-", "_")
    if normalized in _SENSITIVE_NAMES or any(normalized.endswith(f"_{name}") for name in _SENSITIVE_NAMES):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(child_key): redact_config(child_value, str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    return value


def resolve_config_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a path using the configuration directory when available."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if "_config_dir" in config:
        return resolve_path(config, path)
    return path.resolve()
