"""Deterministic local experiment artifact paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gray.core.config import resolve_path


def artifact_dir(config: dict[str, Any], stage: str, create: bool = False) -> Path:
    """Build the artifact directory for one experiment stage.

    Args:
        config: Experiment configuration containing ``project.output_root`` and
            ``experiment_id``.
        stage: Single directory name, for example ``train`` or ``validate``.
        create: Create the directory and missing parents when ``True``.

    Returns:
        ``<output_root>/<experiment_id>/<stage>`` as an absolute path.

    Raises:
        ValueError: If ``stage`` is empty, absolute, nested, ``.`` or ``..``.
        KeyError: If required configuration fields are missing.
        OSError: If ``create`` is true and the directory cannot be created.
    """
    stage_path = Path(stage)
    if not stage or stage_path.is_absolute() or stage_path.name != stage or stage in {".", ".."}:
        raise ValueError(f"invalid artifact stage: {stage!r}")
    output_root = resolve_path(config, config["project"]["output_root"])
    path = output_root / config["experiment_id"] / stage
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
