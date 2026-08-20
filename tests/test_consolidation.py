from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from rich.console import Console

from gray.callbacks import EarlyStopping
from gray.dicom import apply_window_level
from gray.experiment import artifact_dir, experiment_manifest
from gray.experiment import Experiment
from gray.experiment._config import redact_config
from gray.experiment.report import experiment_report
from gray.inference import Ensemble
from gray.preprocess import remove_background
from gray.utils.logging import get_logger
from gray.core.config import load_config


def test_recursive_config_redaction_is_shared_by_report() -> None:
    config = {
        "credentials": [{"api_key": "hidden"}],
        "service": {"signing_key": "also-hidden"},
    }
    assert redact_config(config) == {
        "credentials": "<redacted>",
        "service": {"signing_key": "<redacted>"},
    }
    stream = StringIO()
    experiment_report(config, console=Console(file=stream, force_terminal=False, width=120))
    output = stream.getvalue()
    assert "hidden" not in output
    assert "also-hidden" not in output
    assert "<redacted>" in output


def test_remove_background_matches_one_complete_color() -> None:
    image = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [255, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    result = remove_background(
        image,
        background_colors=[[0, 0, 0], [255, 255, 255]],
        spatial_axes=(0, 1),
    )
    assert result.shape == (1, 1, 3)
    np.testing.assert_array_equal(result[0, 0], [255, 0, 0])


def test_ensemble_rejects_complex_probabilities_and_handles_large_weights() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        Ensemble(np.array([0.2 + 0.1j, 0.8]))
    result = Ensemble(
        np.array([0.2, 0.8]),
        np.array([0.6, 0.4]),
        method="weighted",
        weights=[1e308, 1e308],
    )
    np.testing.assert_allclose(result, [0.4, 0.6])


def test_early_stopping_state_requires_matching_complete_configuration() -> None:
    state = EarlyStopping(patience=2, mode="max", min_delta=0.1).state_dict()
    with pytest.raises(ValueError, match="configuration"):
        EarlyStopping(patience=2, mode="max", min_delta=0.2).load_state_dict(state)
    state["stopped"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        EarlyStopping(patience=2, mode="max", min_delta=0.1).load_state_dict(state)


def test_manifest_config_identity_excludes_run_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.json"
    checkpoint.write_text("{}", encoding="utf-8")
    base = {
        "experiment_id": "exp",
        "data": {"data_version": "v1", "label_schema": "binary"},
        "model": {"model_version": "v1", "architecture": "demo"},
        "train": {"seed": 42},
    }
    first = experiment_manifest({**base, "run_id": "run-a"}, checkpoint)
    second = experiment_manifest({**base, "run_id": "run-b"}, checkpoint)
    assert first["config"]["sha256"] == second["config"]["sha256"]


def test_artifact_dir_rejects_escaping_experiment_id(tmp_path: Path) -> None:
    config = {
        "_config_dir": str(tmp_path),
        "experiment_id": "../outside",
        "project": {"output_root": "outputs"},
    }
    with pytest.raises(ValueError, match="experiment_id"):
        artifact_dir(config, "train")


def test_dicom_linear_window_handles_width_one() -> None:
    image = sitk.GetImageFromArray(np.array([[-1, 0, 1]], dtype=np.int16))
    result = sitk.GetArrayFromImage(apply_window_level(image, 1, 0))
    np.testing.assert_array_equal(result, [[0.0, 1.0, 1.0]])


def test_logger_close_releases_handlers_and_tracker_formatting(tmp_path: Path) -> None:
    class Tracker:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def report_text(self, message: str, **_: object) -> None:
            self.messages.append(message)

    tracker = Tracker()
    logger = get_logger("test", tmp_path, tracker=tracker)
    logger.info("%(name)s", {"name": "run"})
    assert tracker.messages == ["run"]
    logger.close()
    assert logger.logger.handlers == []


def test_load_config_returns_mapping_with_source_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("project:\n  output_root: outputs\n", encoding="utf-8")
    config = load_config(config_path)
    assert config["experiment_id"] == "demo"
    assert config["_config_path"] == str(config_path.resolve())
    assert config["_config_dir"] == str(tmp_path.resolve())


def test_experiment_run_uses_single_training_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = Experiment.__new__(Experiment)
    experiment._config = {"experiment_id": "demo"}
    experiment._task = SimpleNamespace(id="run-1")
    experiment._closed = False
    logger = SimpleNamespace(info=lambda *args: None)
    captured: dict[str, object] = {}
    monkeypatch.setattr(experiment, "get_logger", lambda name: logger)
    monkeypatch.setattr(experiment, "complete", lambda checkpoint, **kwargs: captured.update({"checkpoint": checkpoint, **kwargs}) or {"ok": True})
    monkeypatch.setattr(experiment, "fail", lambda reason: (_ for _ in ()).throw(AssertionError(reason)))
    result = experiment.run(lambda config, received_logger: ("model.bin", {"status": "passed"}))
    assert result == {"ok": True}
    assert captured["checkpoint"] == "model.bin"


def test_experiment_run_marks_training_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = Experiment.__new__(Experiment)
    experiment._config = {"experiment_id": "demo"}
    experiment._task = SimpleNamespace(id="run-1")
    experiment._closed = False
    monkeypatch.setattr(experiment, "get_logger", lambda name: SimpleNamespace(info=lambda *args: None))
    failures: list[str] = []
    monkeypatch.setattr(experiment, "fail", lambda reason: failures.append(reason))
    with pytest.raises(RuntimeError, match="boom"):
        experiment.run(lambda config, logger: (_ for _ in ()).throw(RuntimeError("boom")))
    assert failures == ["boom"]
