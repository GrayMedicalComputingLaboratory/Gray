"""Subprocess runner for the native nnU-Net v2 command line tools."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gray.nnunet.metrics import NnUNetLogReporter
from gray.nnunet.paths import model_dir, nnunet_environment
from gray.runtime.device import resolve_device
from gray.utils.hashing import sha256


class NnUNetRunner:
    """Build and execute official nnU-Net commands through a Gray logger."""

    def __init__(self, config: dict[str, Any], logger: Any, *, device: str | None = None) -> None:
        """Store resolved configuration and the required Gray logger."""
        if not callable(getattr(logger, "info", None)) or not callable(
            getattr(logger, "metric", None)
        ):
            raise TypeError("logger must provide GrayLogger-compatible info and metric methods")
        self.config = config
        self.logger = logger
        requested = device or config.get("runtime", {}).get(
            "device", config.get("device", "cpu")
        )
        self.device = resolve_device(requested)

    def plan_and_preprocess(self, *, verify_dataset_integrity: bool = True) -> None:
        """Run the official planning and preprocessing command."""
        command = self._command("nnUNetv2_plan_and_preprocess") + [
            "-d",
            str(self._settings()["dataset_id"]),
        ]
        if verify_dataset_integrity:
            command.append("--verify_dataset_integrity")
        self.run(command, create_results=False)

    def train(self, *, class_names: Sequence[str] = ()) -> None:
        """Run one configured native nnU-Net training fold."""
        settings = self.config["training"]
        command = self._command("nnUNetv2_train") + [
            str(self._settings()["dataset_id"]),
            str(settings["configuration"]),
            str(settings["split_index"]),
            "-tr", str(settings["trainer"]),
            "-p", str(settings["plans"]),
            "-device", self._nnunet_device(),
        ]
        if bool(settings.get("save_probabilities", True)):
            command.append("--npz")
        self.run(command, class_names=class_names, create_results=True)

    def predict(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        *,
        checkpoint: str | None = None,
        class_names: Sequence[str] = (),
    ) -> None:
        """Run native nnU-Net prediction on a prepared image directory."""
        settings = self.config["training"]
        inference = self.config.get("inference", {})
        preprocessing_workers = self._positive_worker_count(
            inference.get("preprocessing_workers", 1), "preprocessing_workers"
        )
        export_workers = self._positive_worker_count(
            inference.get("export_workers", 1), "export_workers"
        )
        command = self._command("nnUNetv2_predict") + [
            "-i", str(input_dir), "-o", str(output_dir),
            "-d", str(self._settings()["dataset_id"]),
            "-c", str(settings["configuration"]),
            "-f", str(settings["split_index"]),
            "-tr", str(settings["trainer"]), "-p", str(settings["plans"]),
            "-chk", str(checkpoint or inference["checkpoint"]),
            "-device", self._nnunet_device(),
            "-npp", str(preprocessing_workers),
            "-nps", str(export_workers),
        ]
        if bool(inference.get("save_probabilities", True)):
            command.append("--save_probabilities")
        Path(output_dir).mkdir(parents=True, exist_ok=False)
        self.run(command, class_names=class_names)

    def run(
        self,
        command: list[str],
        *,
        class_names: Sequence[str] = (),
        create_results: bool = False,
        cwd: Path | None = None,
    ) -> None:
        """Stream a command and raise with recent output on failure."""
        executable = shutil.which(command[0])
        if executable is None:
            raise RuntimeError(f"{command[0]} is unavailable; install the nnU-Net v2 dependency")
        command = [executable, *command[1:]]
        environment = nnunet_environment(
            self.config,
            visible_gpu=None if self._device() == "cpu" else self._gpu_index(),
            create_results=create_results,
        )
        reporter = NnUNetLogReporter(self.logger, list(class_names))
        recent: list[str] = []
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=environment,
            cwd=cwd or Path(self.config["_config_dir"]),
        )
        assert process.stdout is not None
        for line in process.stdout:
            message = line.rstrip()
            self.logger.info(message)
            reporter.consume(message)
            recent.append(message)
            del recent[:-20]
        reporter.close()
        if process.wait() != 0:
            raise RuntimeError(
                f"nnU-Net command failed: {' '.join(command)}\n" + "\n".join(recent)
            )

    def validate_checkpoint(self, *, expected_sha256: str | None = None) -> Path:
        """Resolve and verify the configured fold checkpoint."""
        inference = self.config["inference"]
        expected = expected_sha256 or inference.get("checkpoint_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("checkpoint SHA-256 must be a registered 64-character digest")
        path = model_dir(self.config) / f"fold_{int(self.config['training']['split_index'])}" / str(inference["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(f"configured checkpoint not found: {path}")
        actual = sha256(path)
        if actual != expected.lower():
            raise ValueError(f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}")
        return path

    def _command(self, executable: str) -> list[str]:
        return [executable]

    def _settings(self) -> dict[str, Any]:
        value = self.config.get("nnunet")
        return value if isinstance(value, dict) else self.config["preprocessing"]["nnunet"]

    def _device(self) -> str:
        return self.device

    def _nnunet_device(self) -> str:
        return "cpu" if self._device() == "cpu" else "cuda"

    def _gpu_index(self) -> str:
        value = self._device()
        return value.removeprefix("cuda:")

    @staticmethod
    def _positive_worker_count(value: Any, name: str) -> int:
        """Validate a native nnU-Net prediction worker count."""
        count = int(value)
        if count < 1:
            raise ValueError(f"inference.{name} must be a positive integer")
        return count
