from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gray.callbacks import EarlyStopping
from gray.core.config import load_config
from gray.dicom import (
    DicomVolume,
    apply_monochrome,
    apply_window_level,
    get_spacing,
    read_series,
    resample_volume,
)
from gray.experiment import (
    artifact_dir,
    get_dataset_version,
    init_task,
    register_dataset_version,
)
from gray.experiment import clearml as clearml_module
from gray.inference import Ensemble
from gray.preprocess import remove_background
from gray.utils.logging import forward_standard_logger, get_logger
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


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


def test_artifact_dir_rejects_escaping_experiment_id(tmp_path: Path) -> None:
    config = {
        "_config_dir": str(tmp_path),
        "experiment_id": "../outside",
        "project": {"output_root": "outputs"},
    }
    with pytest.raises(ValueError, match="experiment_id"):
        artifact_dir(config, "train")


def test_dicom_linear_window_handles_width_one() -> None:
    image = DicomVolume(
        array=np.array([[-1, 0, 1]], dtype=np.int16),
        spacing=(1.0, 1.0),
        datasets=(),
        photometric_interpretation="MONOCHROME2",
    )
    result = apply_window_level(image, 1, 0)
    np.testing.assert_array_equal(result.array, [[0.0, 1.0, 1.0]])
    assert result.array.dtype == np.float32


def test_monochrome_inversion_avoids_integer_overflow() -> None:
    image = DicomVolume(
        array=np.array([[0, 65535]], dtype=np.uint16),
        spacing=(1.0, 1.0),
        datasets=(),
        photometric_interpretation="MONOCHROME1",
    )

    result = apply_monochrome(image)

    np.testing.assert_array_equal(result.array, [[65535, 0]])
    assert result.array.dtype == np.uint16


def test_pydicom_series_decoding_and_volume_operations(tmp_path: Path) -> None:
    series_uid = generate_uid()
    first = _write_dicom_slice(
        tmp_path / "first.dcm",
        0,
        series_uid,
        np.array([[0, 1], [2, 3]], dtype=np.int16),
    )
    second = _write_dicom_slice(
        tmp_path / "second.dcm",
        2,
        series_uid,
        np.array([[4, 5], [6, 7]], dtype=np.int16),
    )

    volume = read_series([first, second])

    assert volume.array.shape == (2, 2, 2)
    assert get_spacing(volume) == (2.0, 0.5, 0.75)
    np.testing.assert_array_equal(volume.array[0], [[-1000, -998], [-996, -994]])
    normalized = apply_monochrome(volume)
    assert normalized.photometric_interpretation == "MONOCHROME2"
    np.testing.assert_array_equal(
        normalized.array, volume.array.max() + volume.array.min() - volume.array
    )
    resampled = resample_volume(normalized, (1.0, 0.5, 0.75), interpolator="nearest")
    assert resampled.array.shape == (4, 2, 2)
    assert resampled.array.dtype == normalized.array.dtype


def _write_dicom_slice(
    path: Path,
    z_position: float,
    series_uid: str,
    pixels: np.ndarray,
) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesInstanceUID = series_uid
    dataset.PatientName = "Test^Patient"
    dataset.StudyDate = datetime.now().strftime("%Y%m%d")
    dataset.Modality = "CT"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.PixelSpacing = [0.5, 0.75]
    dataset.ImagePositionPatient = [0.0, 0.0, z_position]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PhotometricInterpretation = "MONOCHROME1"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -1000
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path


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


def test_forward_standard_logger_uses_gray_handlers_and_cleans_up(
    tmp_path: Path,
) -> None:
    external = logging.getLogger("clearml.gray-test")
    original_handler = logging.NullHandler()
    external.addHandler(original_handler)
    logger = get_logger("clearml-forward", tmp_path)

    forward_standard_logger("clearml.gray-test", logger)
    external.warning("ClearML warning %s", "forwarded")

    assert original_handler not in external.handlers
    assert len(external.handlers) == 1
    assert "ClearML warning forwarded" in (tmp_path / "run.log").read_text(
        encoding="utf-8"
    )
    logger.close()
    assert external.handlers == []


def test_init_task_maps_gray_config_to_clearml(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    task = SimpleNamespace(
        add_tags=lambda tags: captured.update(tags=tags),
        connect_configuration=lambda **options: captured.update(configuration=options),
    )

    class Task:
        TaskTypes = SimpleNamespace(training="training")

        @staticmethod
        def init(**options: object) -> object:
            captured["init"] = options
            return task

    monkeypatch.setattr(clearml_module, "_clearml_task_class", lambda: Task)
    config = {
        "_config_path": "private.yaml",
        "experiment_id": "demo",
        "tracking": {
            "project_name": "Gray/Test",
            "tags": ["unit"],
            "output_uri": "file:///models",
        },
    }

    result = init_task(config)

    assert result is task
    assert captured["init"] == {
        "project_name": "Gray/Test",
        "task_name": "demo",
        "task_type": "training",
        "reuse_last_task_id": False,
        "output_uri": "file:///models",
        "auto_connect_arg_parser": False,
        "auto_connect_streams": True,
    }
    assert captured["tags"] == ["unit"]
    assert captured["configuration"] == {
        "configuration": {
            "experiment_id": "demo",
            "tracking": config["tracking"],
        },
        "name": "Resolved Config",
    }


def test_register_dataset_version_runs_complete_clearml_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.txt").write_text("sample", encoding="utf-8")
    calls: dict[str, object] = {"added": []}

    class FakeDataset:
        id = "dataset-id"

        def add_files(self, **options: object) -> int:
            calls["added"].append(options)  # type: ignore[union-attr]
            return 1

        def upload(self, **options: object) -> bool:
            calls["upload"] = options
            return True

        def finalize(self, **options: object) -> bool:
            calls["finalize"] = options
            return True

        def publish(self, **options: object) -> bool:
            calls["publish"] = options
            return True

    class Dataset:
        @staticmethod
        def create(**options: object) -> FakeDataset:
            calls["create"] = options
            return FakeDataset()

    monkeypatch.setattr(clearml_module, "_clearml_dataset_class", lambda: Dataset)

    dataset_id = register_dataset_version(
        project="Gray/Data",
        name="demo",
        version="1.0.0",
        paths={"raw": source},
        tags=["unit"],
        parent_ids=["parent-id"],
        output_uri="file:///datasets",
        description="test dataset",
    )

    assert dataset_id == "dataset-id"
    assert calls["create"] == {
        "dataset_name": "demo",
        "dataset_project": "Gray/Data",
        "dataset_tags": ["unit"],
        "parent_datasets": ["parent-id"],
        "dataset_version": "1.0.0",
        "output_uri": "file:///datasets",
        "description": "test dataset",
    }
    assert calls["added"] == [
        {
            "path": str(source.resolve()),
            "dataset_path": "raw",
            "recursive": True,
            "verbose": False,
        }
    ]
    assert calls["upload"] == {"show_progress": False, "verbose": False}
    assert calls["finalize"] == {"verbose": False, "raise_on_error": True}
    assert calls["publish"] == {"raise_on_error": True}


def test_get_dataset_version_requires_published_completed_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    class Dataset:
        @staticmethod
        def get(**options: object) -> object:
            captured.update(options)
            return expected

    monkeypatch.setattr(clearml_module, "_clearml_dataset_class", lambda: Dataset)

    result = get_dataset_version("dataset-id")

    assert result is expected
    assert captured == {
        "dataset_id": "dataset-id",
        "only_completed": True,
        "only_published": True,
    }
