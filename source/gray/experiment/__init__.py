"""ClearML task initialization and local artifact paths."""

from .artifacts import artifact_dir
from .clearml import get_dataset_version, init_task, register_dataset_version

__all__ = [
    "artifact_dir",
    "get_dataset_version",
    "init_task",
    "register_dataset_version",
]
