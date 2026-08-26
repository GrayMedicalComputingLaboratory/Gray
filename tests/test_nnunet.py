"""Tests for the reusable native nnU-Net wrappers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from gray.nnunet import (
    NnUNetLogReporter,
    NnUNetRunner,
    dataset_dir,
    model_dir,
    nnunet_environment,
)


def _config(tmp_path: Path) -> dict:
    return {
        "_config_dir": str(tmp_path),
        "experiment_id": "exp_demo",
        "project": {"output_root": str(tmp_path / "outputs")},
        "preprocessing": {"nnunet": {"dataset_id": 502, "dataset_name": "HC18FetalHead"}},
        "training": {
            "configuration": "2d",
            "trainer": "nnUNetTrainer",
            "plans": "nnUNetPlans",
            "split_index": 0,
        },
        "inference": {
            "checkpoint": "checkpoint_best.pth",
            "checkpoint_sha256": "0" * 64,
            "preprocessing_workers": 1,
            "export_workers": 1,
        },
    }


def test_paths_and_environment_are_experiment_scoped(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert dataset_dir(config).name == "Dataset502_HC18FetalHead"
    assert model_dir(config).name == "nnUNetTrainer__nnUNetPlans__2d"
    environment = nnunet_environment(config, visible_gpu="0")
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["nnUNet_raw"].endswith("exp_demo\\nnunet_raw") or environment["nnUNet_raw"].endswith("exp_demo/nnunet_raw")


class _Logger:
    def __init__(self) -> None:
        self.metrics = []
        self.messages = []

    def metric(self, name, value, *, iteration, title):
        self.metrics.append((name, value, iteration, title))

    def info(self, message):
        self.messages.append(message)


def test_reporter_handles_numpy_scalar_text() -> None:
    logger = _Logger()
    reporter = NnUNetLogReporter(logger, ["fetal_head"])
    for line in (
        "Epoch 0",
        "train_loss -0.4",
        "val_loss -0.3",
        "Pseudo dice [np.float32(0.9373)]",
        "Epoch time: 28.4 s",
    ):
        reporter.consume(line)
    assert ("fetal_head", 0.9373, 0, "Pseudo Dice") in logger.metrics


def test_plan_command_contains_dataset_id(tmp_path: Path, monkeypatch) -> None:
    runner = NnUNetRunner(_config(tmp_path), _Logger(), device="cpu")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(runner, "run", fake_run)
    runner.plan_and_preprocess()
    assert captured["command"] == [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        "502",
        "--verify_dataset_integrity",
    ]
    assert captured["kwargs"] == {"create_results": False}


def test_predict_builds_native_worker_and_checkpoint_arguments(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config["inference"].update(
        {"preprocessing_workers": 2, "export_workers": 3}
    )
    runner = NnUNetRunner(config, _Logger(), device="cpu")
    captured = {}
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, **kwargs: captured.update(command=command, kwargs=kwargs),
    )
    output_dir = tmp_path / "predictions"
    runner.predict(tmp_path / "input", output_dir)
    command = captured["command"]
    assert command[command.index("-d") + 1] == "502"
    assert command[command.index("-chk") + 1] == "checkpoint_best.pth"
    assert command[command.index("-npp") + 1] == "2"
    assert command[command.index("-nps") + 1] == "3"
    assert output_dir.is_dir()


def test_predict_rejects_non_positive_worker_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["inference"]["preprocessing_workers"] = 0
    runner = NnUNetRunner(config, _Logger(), device="cpu")
    with pytest.raises(ValueError, match="positive integer"):
        runner.predict(tmp_path / "input", tmp_path / "output")


def test_checkpoint_hash_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = model_dir(config) / "fold_0" / "checkpoint_best.pth"
    checkpoint.parent.mkdir(parents=True)
    content = b"checkpoint"
    checkpoint.write_bytes(content)
    config["inference"]["checkpoint_sha256"] = hashlib.sha256(content).hexdigest()
    runner = NnUNetRunner(config, _Logger(), device="cpu")
    assert runner.validate_checkpoint() == checkpoint
    config["inference"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="mismatch"):
        runner.validate_checkpoint()


def test_run_forwards_output_and_retains_failure_context(
    tmp_path: Path, monkeypatch
) -> None:
    class _Process:
        stdout = iter(["first line\n", "failure detail\n"])

        @staticmethod
        def wait():
            return 2

    logger = _Logger()
    runner = NnUNetRunner(_config(tmp_path), logger, device="cpu")
    monkeypatch.setattr("gray.nnunet.runner.shutil.which", lambda value: value)
    monkeypatch.setattr(
        "gray.nnunet.runner.subprocess.Popen", lambda *args, **kwargs: _Process()
    )
    with pytest.raises(RuntimeError, match="failure detail"):
        runner.run(["nnUNetv2_train"])
    assert logger.messages == ["first line", "failure detail"]


def test_runner_requires_gray_logger_interface(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="GrayLogger-compatible"):
        NnUNetRunner(_config(tmp_path), object(), device="cpu")
