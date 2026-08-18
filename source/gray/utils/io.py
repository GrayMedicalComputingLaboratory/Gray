"""JSON input/output helpers for experiment artifacts."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    """Write one UTF-8 JSON artifact atomically, creating its parent directory."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    payload = json.dumps(dict(value), indent=2, ensure_ascii=False)
    try:
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one JSON object artifact and reject non-object roots."""
    source = Path(path).expanduser()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact root must be an object: {source}")
    return value

__all__ = ["read_json", "write_json"]
