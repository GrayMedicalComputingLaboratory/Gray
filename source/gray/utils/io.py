"""Small, explicit artifact serialization helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a UTF-8 JSON artifact, creating only its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
