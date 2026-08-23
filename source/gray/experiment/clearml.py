"""Initialize a ClearML task from a resolved Gray configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gray.utils.logging import GrayLogger, forward_standard_logger


def init_task(config: Mapping[str, Any], *, logger: GrayLogger | None = None) -> Any:
    """Create and configure one native ClearML training task.

    ClearML owns task completion and failure detection through its process-exit
    and uncaught-exception hooks. Callers use the returned task directly for
    logs, metrics, artifacts, parameters, and models.

    Args:
        config: Mapping returned by :func:`gray.load_config`. It must contain
            ``experiment_id`` and ``tracking.project_name``. Optional
            ``tracking.tags`` and ``tracking.output_uri`` values are passed to
            ClearML.
        logger: Optional Gray logger receiving local ClearML SDK records.

    Returns:
        The initialized ``clearml.Task``.

    Raises:
        KeyError: If a required configuration field is missing.
        ModuleNotFoundError: If the optional ClearML dependency is absent.
        TypeError: If ClearML rejects an option type.
        ValueError: If ClearML rejects an option value.
    """
    if logger is not None:
        forward_standard_logger("clearml", logger)
    Task = _clearml_task_class()
    tracking = config["tracking"]
    task = Task.init(
        project_name=tracking["project_name"],
        task_name=config["experiment_id"],
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        output_uri=tracking.get("output_uri"),
        auto_connect_arg_parser=False,
        auto_connect_streams=True,
    )
    tags = tracking.get("tags")
    if tags:
        task.add_tags(tags)
    task.connect_configuration(
        configuration={
            key: value for key, value in config.items() if not str(key).startswith("_")
        },
        name="Resolved Config",
    )
    return task


def _clearml_task_class() -> Any:
    """Import the optional ClearML Task class.

    Returns:
        The ``clearml.Task`` class.

    Raises:
        ModuleNotFoundError: If ClearML is not installed.
    """
    try:
        from clearml import Task
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ClearML integration requires the optional dependency: pip install 'gray[clearml]'"
        ) from error
    return Task


def register_dataset_version(
    *,
    project: str,
    name: str,
    version: str,
    paths: Mapping[str, str | Path],
    tags: list[str] | None = None,
    parent_ids: list[str] | None = None,
    output_uri: str | None = None,
    description: str | None = None,
    publish: bool = True,
    logger: GrayLogger | None = None,
) -> str:
    """Create, upload, finalize, and optionally publish one ClearML Dataset.

    Args:
        project: ClearML dataset project name.
        name: Stable dataset family name.
        version: Immutable dataset version string.
        paths: Dataset-relative destination names mapped to local files or directories.
        tags: Optional searchable ClearML tags.
        parent_ids: Optional parent Dataset IDs for derived versions.
        output_uri: Optional ClearML file-server or object-storage URI.
        description: Optional dataset description.
        publish: Publish the completed dataset when true.
        logger: Optional Gray logger receiving local ClearML SDK records.

    Returns:
        The immutable ClearML Dataset ID.

    Raises:
        FileNotFoundError: If a declared local path does not exist.
        ValueError: If names, paths, or a ClearML lifecycle operation are invalid.
        ModuleNotFoundError: If the optional ClearML dependency is absent.
    """
    if not project or not name or not version:
        raise ValueError("dataset project, name, and version must be non-empty")
    if not paths:
        raise ValueError("dataset paths must not be empty")
    if logger is not None:
        forward_standard_logger("clearml", logger)
    Dataset = _clearml_dataset_class()
    dataset = Dataset.create(
        dataset_name=name,
        dataset_project=project,
        dataset_tags=tags,
        parent_datasets=parent_ids,
        dataset_version=version,
        output_uri=output_uri,
        description=description,
    )
    for dataset_path, local_value in paths.items():
        local_path = Path(local_value).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"dataset source path not found: {local_path}")
        added = dataset.add_files(
            path=str(local_path),
            dataset_path=dataset_path,
            recursive=local_path.is_dir(),
            verbose=False,
        )
        if added < 1:
            raise ValueError(
                f"ClearML Dataset added no files from {local_path} to {dataset_path!r}"
            )
    if dataset.upload(show_progress=False, verbose=False) is False:
        raise ValueError(f"ClearML Dataset upload failed: {project}/{name}:{version}")
    if not dataset.finalize(verbose=False, raise_on_error=True):
        raise ValueError(f"ClearML Dataset finalize failed: {project}/{name}:{version}")
    if publish and not dataset.publish(raise_on_error=True):
        raise ValueError(f"ClearML Dataset publish failed: {project}/{name}:{version}")
    return str(dataset.id)


def get_dataset_version(
    dataset_id: str,
    *,
    only_published: bool = True,
    logger: GrayLogger | None = None,
) -> Any:
    """Resolve one immutable ClearML Dataset by ID.

    Args:
        dataset_id: Non-empty ClearML Dataset ID.
        only_published: Require the Dataset to be published when true.
        logger: Optional Gray logger receiving local ClearML SDK records.

    Returns:
        The resolved native ``clearml.Dataset`` instance.

    Raises:
        ValueError: If ``dataset_id`` is empty.
        ModuleNotFoundError: If the optional ClearML dependency is absent.
    """
    if not dataset_id:
        raise ValueError("dataset_id must not be empty")
    if logger is not None:
        forward_standard_logger("clearml", logger)
    Dataset = _clearml_dataset_class()
    return Dataset.get(
        dataset_id=dataset_id,
        only_completed=True,
        only_published=only_published,
    )


def _clearml_dataset_class() -> Any:
    """Import the optional ClearML Dataset class."""
    try:
        from clearml import Dataset
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ClearML integration requires the optional dependency: pip install 'gray[clearml]'"
        ) from error
    return Dataset
