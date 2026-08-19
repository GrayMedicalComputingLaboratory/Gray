"""JSON input/output helpers for experiment artifacts."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    """Write a mapping to a UTF-8 JSON file using an atomic replacement.

    The parent directory is created when necessary. Data is first written to a
    temporary file beside the destination and then moved into place so readers
    never observe a partially written artifact.

    Args:
        path: Destination file path. User-home markers such as ``~`` are
            expanded.
        value: Mapping to serialize as the root JSON object.

    Returns:
        None.

    Raises:
        OSError: If a directory or file cannot be created, written, or replaced.
        TypeError: If ``value`` contains an object that JSON cannot serialize.
    """
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
    """Read a UTF-8 JSON file whose root value is an object.

    Args:
        path: JSON file path. User-home markers such as ``~`` are expanded.

    Returns:
        The decoded JSON object as a dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        OSError: If the file cannot be read.
        UnicodeDecodeError: If the file is not valid UTF-8.
        ValueError: If the file is invalid JSON or its root is not an object.
    """
    source = Path(path).expanduser()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact root must be an object: {source}")
    return value
