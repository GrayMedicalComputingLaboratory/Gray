"""ClearML-backed experiment lifecycle management."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from gray.utils.io import write_json
from gray.utils.logging import GrayLogger, get_logger

from .artifacts import artifact_dir
from ._config import public_config, redact_config, resolve_config_path
from .manifest import experiment_manifest
from .report import experiment_report


class Experiment:
    """Manage one Gray experiment run through a ClearML Task.

    Creating an instance starts a new ClearML Task, connects a redacted copy of
    the resolved configuration, and exposes the generated Task ID as ``run_id``.
    Use :meth:`run` for the standard managed training lifecycle. Advanced
    callers may invoke :meth:`complete` or :meth:`fail` directly.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        project_name: str | None = None,
        task_name: str | None = None,
        tags: Sequence[str] | None = None,
        output_uri: str | None = None,
        auto_connect_frameworks: bool = True,
    ) -> None:
        """Start a ClearML training Task for a resolved Gray configuration.

        Args:
            config: Resolved Gray configuration containing ``experiment_id``.
            project_name: ClearML project name. When omitted, reads
                ``config.tracking.project_name``.
            task_name: Human-readable ClearML Task name. Defaults to the Gray
                ``experiment_id``.
            tags: Optional ClearML tags. Defaults to ``config.tracking.tags``.
            output_uri: Optional ClearML model/artifact storage URI. Defaults to
                ``config.tracking.output_uri``.
            auto_connect_frameworks: Let ClearML automatically connect supported
                training frameworks and their model files.

        Raises:
            ModuleNotFoundError: If the optional ``clearml`` package is absent.
            TypeError: If the configuration or option types are invalid.
            ValueError: If required names are empty or tags are invalid.
            KeyError: If ``experiment_id`` is missing.
        """
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        self._config = dict(config)
        tracking = config.get("tracking", {})
        if not isinstance(tracking, Mapping):
            raise TypeError("config.tracking must be a mapping when provided")
        selected_project = project_name if project_name is not None else tracking.get("project_name")
        selected_task = task_name if task_name is not None else config["experiment_id"]
        selected_tags = tags if tags is not None else tracking.get("tags", ())
        selected_output_uri = output_uri if output_uri is not None else tracking.get("output_uri")
        _validate_name(selected_project, "project_name")
        _validate_name(selected_task, "task_name")
        normalized_tags = _normalize_tags(selected_tags)
        if selected_output_uri is not None and not isinstance(selected_output_uri, str):
            raise TypeError("output_uri must be a string or None")
        if not isinstance(auto_connect_frameworks, bool):
            raise TypeError("auto_connect_frameworks must be a bool")

        task_class = _clearml_task_class()
        self._task = task_class.init(
            project_name=selected_project.strip(),
            task_name=selected_task.strip(),
            task_type=task_class.TaskTypes.training,
            reuse_last_task_id=False,
            output_uri=selected_output_uri,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=auto_connect_frameworks,
            auto_connect_streams=False,
        )
        if normalized_tags:
            self._task.add_tags(normalized_tags)
        self._task.connect_configuration(
            configuration=redact_config(public_config(config)),
            name="Resolved Config",
        )
        self._loggers: list[GrayLogger] = []
        self._closed = False

    @property
    def run_id(self) -> str:
        """Return the unique ClearML Task ID for this training run."""
        return str(self._task.id)

    @property
    def task(self) -> Any:
        """Return the underlying ClearML Task for native reporting APIs."""
        return self._task

    def get_logger(self, name: str = "experiment", output_dir: str | Path | None = None) -> GrayLogger:
        """Create a Rich/file logger mirrored to this ClearML Task.

        Args:
            name: Logical local logger name.
            output_dir: Local ``run.log`` directory. Defaults to the experiment's
                ``logs`` artifact directory.

        Returns:
            A semantic logger writing to the terminal, local file, and ClearML.
        """
        self._ensure_open()
        destination = Path(output_dir).expanduser() if output_dir is not None else _default_log_path(self._config)
        logger = get_logger(name, destination, tracker=self._task.get_logger())
        self._loggers.append(logger)
        return logger

    def run(
        self,
        training: Callable[[dict[str, Any], GrayLogger], tuple[str | Path, Mapping[str, Any] | None]],
        *,
        logger_name: str = "train",
        manifest_path: str | Path | None = None,
        print_report: bool = True,
    ) -> dict[str, Any]:
        """Execute and finalize one managed training function.

        The training function receives the resolved configuration and a logger
        connected to the terminal, local file, and ClearML. Gray records the
        configuration source and experiment identities before training. A
        successful function must return ``(checkpoint, evaluation)``; Gray then
        persists and uploads the final manifest and checkpoint. Any exception
        marks the ClearML Task as failed and is re-raised unchanged.

        Args:
            training: Callable accepting ``(config, logger)`` and returning a
                checkpoint path plus an optional evaluation mapping.
            logger_name: Logical name used for the managed training logger.
            manifest_path: Optional JSON manifest destination. Relative paths
                resolve from the configuration directory.
            print_report: Print the final Rich experiment report when true.

        Returns:
            The structured experiment manifest uploaded to ClearML.

        Raises:
            TypeError: If ``training`` is not callable or returns an invalid
                result contract.
            BaseException: Re-raises any error produced during training or
                finalization after marking the Task as failed when possible.
        """
        self._ensure_open()
        try:
            if not callable(training):
                raise TypeError("training must be callable")
            logger = self.get_logger(logger_name)
            config_path = self._config.get("_config_path")
            if config_path is not None:
                logger.info("config=%s", config_path)
            logger.info("experiment_id=%s", self._config["experiment_id"])
            logger.info("clearml_run_id=%s", self.run_id)
            result = training(self._config, logger)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("training must return (checkpoint, evaluation)")
            checkpoint, evaluation = result
            return self.complete(
                checkpoint,
                evaluation=evaluation,
                manifest_path=manifest_path,
                print_report=print_report,
            )
        except BaseException as error:
            if not self._closed:
                try:
                    self.fail(str(error) or error.__class__.__name__)
                except BaseException as cleanup_error:
                    error.add_note(f"failed to close ClearML Task: {cleanup_error}")
            raise

    def complete(
        self,
        checkpoint: str | Path,
        *,
        evaluation: Mapping[str, Any] | None = None,
        manifest_path: str | Path | None = None,
        print_report: bool = True,
    ) -> dict[str, Any]:
        """Finalize the run, upload artifacts, and close the ClearML Task.

        Args:
            checkpoint: Trained model checkpoint to identify and upload.
            evaluation: Optional evaluation status and metrics.
            manifest_path: JSON destination. Defaults to
                ``<output_root>/<experiment_id>/experiment/manifest.json``.
            print_report: Print the Rich experiment lineage report when true.

        Returns:
            The structured experiment manifest uploaded to ClearML.

        Raises:
            RuntimeError: If this experiment has already been closed.
            OSError: If local artifacts cannot be read or written.
            TypeError: If manifest values are not JSON serializable.
        """
        self._ensure_open()
        checkpoint_path = resolve_config_path(self._config, checkpoint)
        manifest = experiment_manifest(
            self._config,
            checkpoint_path,
            evaluation=evaluation,
            run_id=self.run_id,
        )
        destination = (
            resolve_config_path(self._config, manifest_path)
            if manifest_path is not None
            else _default_manifest_path(self._config)
        )
        write_json(destination, manifest)
        self._task.upload_artifact("experiment_manifest", artifact_object=str(destination.resolve()))
        self._task.upload_artifact("checkpoint", artifact_object=str(checkpoint_path))
        if print_report:
            experiment_report(self._config, manifest=manifest)
        self._close_loggers()
        self._task.close()
        self._closed = True
        return manifest

    def fail(self, reason: str) -> None:
        """Mark the ClearML Task as failed and close this experiment.

        Args:
            reason: Non-empty human-readable failure reason.

        Returns:
            None.
        """
        self._ensure_open()
        _validate_name(reason, "reason")
        self._task.mark_failed(status_reason=reason)
        self._close_loggers()
        self._task.close()
        self._closed = True

    def close(self) -> None:
        """Close the ClearML Task without uploading final model artifacts."""
        if not self._closed:
            self._close_loggers()
            self._task.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("experiment is already closed")

    def _close_loggers(self) -> None:
        for logger in self._loggers:
            logger.close()
        self._loggers.clear()


def _clearml_task_class() -> Any:
    try:
        from clearml import Task
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ClearML integration requires the optional dependency: pip install 'gray[clearml]'"
        ) from error
    return Task


def _validate_name(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _normalize_tags(tags: Any) -> list[str]:
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise TypeError("tags must be a sequence of strings")
    normalized: list[str] = []
    for tag in tags:
        _validate_name(tag, "tag")
        normalized.append(tag.strip())
    return normalized
def _default_manifest_path(config: Mapping[str, Any]) -> Path:
    return artifact_dir(dict(config), "experiment", create=True) / "manifest.json"


def _default_log_path(config: Mapping[str, Any]) -> Path:
    return artifact_dir(dict(config), "logs", create=True)
