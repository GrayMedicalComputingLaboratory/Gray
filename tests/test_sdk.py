"""Strict-contract tests and real optional ONNX Runtime integration tests."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml", reason="SDK YAML optional dependency missing")

from gray.sdk import (
    BackendError,
    GrayRuntime,
    ImageInput,
    InferenceError,
    InputError,
    ManifestError,
    ModelManifest,
    RuntimeClosedError,
    load_image,
    load_model,
)
from gray.sdk.cli import main

DEMO = Path(__file__).resolve().parents[1] / "examples" / "sdk"
REAL = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="Real inference requires optional onnxruntime",
)


@pytest.fixture
def contract() -> dict:
    return yaml.safe_load((DEMO / "model" / "manifest.yaml").read_text())


def write_manifest(tmp_path: Path, contract: dict) -> Path:
    contract = copy.deepcopy(contract)
    contract["model"]["path"] = str(DEMO / "model" / "model.onnx")
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    return path


def test_manifest_load_and_relative_path() -> None:
    manifest = ModelManifest.load(DEMO / "model" / "manifest.yaml")
    assert manifest.model_path == DEMO / "model" / "model.onnx"
    assert manifest.input["shape"] == [1, 1, 8, 8]


def test_dimension_names_are_case_insensitive(tmp_path: Path, contract: dict) -> None:
    contract["input"]["dim"] = "nchw"
    manifest = ModelManifest.load(write_manifest(tmp_path, contract))
    assert manifest.input["dim"] == "NCHW"
    image = ImageInput(np.zeros((8, 8), dtype=np.float32), dim="hw")
    assert image.dim == "HW"


def test_chw_and_hwc_are_supported() -> None:
    chw = ImageInput(np.zeros((1, 8, 8), dtype=np.float32), dim="CHW")
    hwc = ImageInput(np.zeros((8, 8, 1), dtype=np.float32), dim="HWC")
    assert chw.dim == "CHW"
    assert hwc.dim == "HWC"


def test_load_png_jpg_as_explicit_rgb_or_grayscale(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image")
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    gray_path = tmp_path / "gray.png"
    pil.fromarray(gray).save(gray_path)
    loaded_gray = load_image(gray_path)
    assert loaded_gray.dim == "HW"
    np.testing.assert_array_equal(loaded_gray.data, gray)

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    rgb_path = tmp_path / "color.jpg"
    pil.fromarray(rgb).save(rgb_path, quality=100, subsampling=0)
    loaded_rgb = load_image(rgb_path)
    assert loaded_rgb.dim == "HWC"
    assert loaded_rgb.data.shape == (8, 8, 3)
    loaded_chw = load_image(rgb_path, dim="chw")
    assert loaded_chw.dim == "CHW"
    assert loaded_chw.data.shape == (3, 8, 8)


@REAL
def test_cli_infers_png_with_declared_conversion(tmp_path: Path, contract: dict) -> None:
    pil = pytest.importorskip("PIL.Image")
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image_path = tmp_path / "input.png"
    pil.fromarray(pixels).save(image_path)
    contract["input"]["preprocessing"] = {
        "cast_dtype": "float32",
        "normalization": {"type": "scale", "scale": 1 / 255},
    }
    manifest = write_manifest(tmp_path, contract)
    output = tmp_path / "result"
    assert main(
        ["infer", "--manifest", str(manifest), "--input", str(image_path), "--output", str(output)]
    ) == 0
    prediction = np.load(output / "outputs.npz")["output"]
    np.testing.assert_allclose(prediction[0, 0], pixels / 255, rtol=1e-6)

    model = load_model(manifest)
    try:
        result = model.predict(image_path)
        np.testing.assert_allclose(result.outputs["output"][0, 0], pixels / 255, rtol=1e-6)
    finally:
        model.close()

    batch_contract = yaml.safe_load(
        (DEMO / "model" / "manifest_batch.yaml").read_text(encoding="utf-8")
    )
    batch_contract["input"]["preprocessing"] = contract["input"]["preprocessing"]
    batch_contract["model"]["path"] = str(
        (DEMO / "model" / "model_batch.onnx").resolve()
    )
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    batch_manifest = batch_dir / "manifest.yaml"
    batch_manifest.write_text(yaml.safe_dump(batch_contract), encoding="utf-8")
    batch_model = load_model(batch_manifest)
    try:
        batch_result = batch_model.predict_batch([image_path, image_path])
        assert batch_result.metadata["batch_size"] == 2
        np.testing.assert_allclose(
            batch_result.outputs["output"][:, 0],
            np.stack([pixels, pixels]) / 255,
            rtol=1e-6,
        )
    finally:
        batch_model.close()


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.pop("input"),
        lambda c: c.update(schema_version="2.0"),
        lambda c: c.update(unknown=True),
        lambda c: c["input"].update(shape=[1, -1]),
        lambda c: c["input"].update(dtype="object"),
        lambda c: c["input"].update(dim="RGB"),
        lambda c: c["input"].update(preprocessing={"resize": {"enabled": True}}),
        lambda c: c["outputs"].append(c["outputs"][0]),
        lambda c: c["runtime"].update(providers=[]),
        lambda c: c["input"].update(
            preprocessing={"normalization": {"type": "scale", "scale": float("nan")}}
        ),
    ],
)
def test_manifest_rejects_invalid(tmp_path: Path, contract: dict, change) -> None:
    change(contract)
    with pytest.raises(ManifestError):
        ModelManifest.load(write_manifest(tmp_path, contract))


def test_missing_model(tmp_path: Path, contract: dict) -> None:
    path = write_manifest(tmp_path, contract)
    content = yaml.safe_load(path.read_text())
    content["model"]["path"] = "missing.onnx"
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ManifestError, match="existing"):
        ModelManifest.load(path)


@pytest.mark.parametrize(
    "data,dim,kwargs",
    [
        (np.zeros((8, 8)), "NCHW", {}),
        (np.zeros((8, 8)), "XX", {}),
        (np.array([[np.nan]]), "HW", {}),
        (np.array([[object()]]), "HW", {}),
        (np.zeros((8, 8)), "HW", {"dtype": "float32"}),
        (np.zeros((8, 8)), "HW", {"spacing": (0, 1)}),
        (np.zeros((8, 8)), "HW", {"direction": ((1, 1), (0, 1))}),
    ],
)
def test_invalid_image(data, dim, kwargs) -> None:
    with pytest.raises(InputError):
        ImageInput(data, dim, **kwargs)


@REAL
def test_real_cpu_and_close() -> None:
    array = np.arange(64, dtype=np.float32).reshape(8, 8)
    with GrayRuntime.from_manifest(DEMO / "model" / "manifest.yaml") as runtime:
        result = runtime.predict(ImageInput(array, "HW", spacing=(1, 1)))
        np.testing.assert_array_equal(result.outputs["output"], array[None, None])
        assert result.task_type == "tensor" and result.latency_ms > 0
        assert result.metadata["providers"] == ["CPUExecutionProvider"]
        assert len(result.metadata["model_sha256"]) == 64
        assert result.metadata["input_shape"] == [8, 8]
        for value in (
            array.astype(np.uint8)[None, None],
            np.zeros((1, 1, 7, 8), np.float32),
        ):
            with pytest.raises(InputError):
                runtime.predict(value)
        with pytest.raises(InputError):
            runtime.predict(ImageInput(np.zeros((8, 8, 8), np.float32), "DHW"))
    runtime.close()
    with pytest.raises(RuntimeClosedError):
        runtime.predict(array)
    with pytest.raises(RuntimeClosedError):
        runtime.__enter__()


@REAL
def test_real_volume() -> None:
    volume = np.ones((4, 8, 8), np.float32)
    with GrayRuntime.from_manifest(DEMO / "model" / "manifest_3d.yaml") as runtime:
        result = runtime.predict(ImageInput(volume, "DHW", spacing=(2, 1, 1)))
    np.testing.assert_array_equal(result.outputs["output"], volume[None, None])


@REAL
def test_real_chw_and_hwc_inputs() -> None:
    chw = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    hwc = chw.transpose(1, 2, 0)
    with GrayRuntime.from_manifest(DEMO / "model" / "manifest.yaml") as runtime:
        from_chw = runtime.predict(ImageInput(chw, "CHW"))
        from_hwc = runtime.predict(ImageInput(hwc, "HWC"))
    np.testing.assert_array_equal(from_chw.outputs["output"], chw[None])
    np.testing.assert_array_equal(from_hwc.outputs["output"], chw[None])


@REAL
def test_real_dynamic_batch(tmp_path: Path) -> None:
    manifest = DEMO / "model" / "manifest_batch.yaml"
    batch = np.load(DEMO / "input" / "sample_batch.npy", allow_pickle=False)
    with GrayRuntime.from_manifest(manifest) as runtime:
        arrays = [item[0] for item in batch]
        channel_arrays = [item for item in batch]
        result = runtime.predict_batch(arrays, dim="hw")
        np.testing.assert_array_equal(result.outputs["output"], batch)
        assert result.metadata["batch_size"] == 3
        assert result.metadata["source_dim"] == "HW"
        result = runtime.predict_batch(batch, dim="nchw")
        np.testing.assert_array_equal(result.outputs["output"], batch)
        assert result.metadata["batch_size"] == 3
        assert result.metadata["batch_dim"] == "N"
        result = runtime.predict(batch, dim="bchw")
        np.testing.assert_array_equal(result.outputs["output"], batch)
        with pytest.raises(InputError, match="at least one"):
            runtime.predict_batch([])
        with pytest.raises(InputError, match="dim containing B or N"):
            runtime.predict_batch(batch)

    source = ModelManifest.load(manifest)
    b_manifest = tmp_path / "manifest_b.yaml"
    content = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    content["input"]["dim"] = "BCHW"
    content["model"]["path"] = str(source.model_path)
    b_manifest.write_text(yaml.safe_dump(content), encoding="utf-8")
    with GrayRuntime.from_manifest(b_manifest) as runtime:
        result = runtime.predict_batch(channel_arrays, dim="CHW")
        np.testing.assert_array_equal(result.outputs["output"], batch)
        assert result.metadata["batch_dim"] == "B"
        result = runtime.predict_batch(batch, dim="bchw")
        np.testing.assert_array_equal(result.outputs["output"], batch)
        result = runtime.predict_batch(batch, dim="nchw")
        np.testing.assert_array_equal(result.outputs["output"], batch)

    b_output = tmp_path / "batch-b-results"
    assert (
        main(
            [
                "infer",
                "--model",
                str(b_manifest),
                "--input",
                str(DEMO / "input" / "sample_batch.npy"),
                "--dim",
                "bchw",
                "--output",
                str(b_output),
            ]
        )
        == 0
    )
    with np.load(b_output / "outputs.npz") as outputs:
        np.testing.assert_array_equal(outputs["output"], batch)

    output_path = tmp_path / "batch-results"
    assert (
        main(
            [
                "infer",
                "--model",
                str(manifest),
                "--input",
                str(DEMO / "input" / "sample_batch.npy"),
                "--dim",
                "nchw",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    with np.load(output_path / "outputs.npz") as outputs:
        np.testing.assert_array_equal(outputs["output"], batch)


@REAL
def test_real_dynamic_volume_batch() -> None:
    manifest = DEMO / "model" / "manifest_batch_3d.yaml"
    batch = np.load(DEMO / "input" / "sample_batch_3d.npy", allow_pickle=False)
    with GrayRuntime.from_manifest(manifest) as runtime:
        volumes = [volume[0] for volume in batch]
        result = runtime.predict_batch(volumes, dim="DHW")
        np.testing.assert_array_equal(result.outputs["output"], batch)
        assert result.metadata["batch_size"] == 2
        single_channel_batch = batch[:, 0]
        result = runtime.predict_batch(single_channel_batch, dim="ndhw")
        np.testing.assert_array_equal(result.outputs["output"], batch)


@REAL
def test_batch_requires_model_batch_axis() -> None:
    from types import SimpleNamespace

    runtime = object.__new__(GrayRuntime)
    runtime.manifest = SimpleNamespace(input={"dim": "CHW"})
    with pytest.raises(InputError, match="include B or N"):
        runtime.predict_batch([np.zeros((1, 8, 8), np.float32)], dim="CHW")


@REAL
def test_cast_normalization(tmp_path: Path, contract: dict) -> None:
    contract["input"]["preprocessing"] = {
        "cast_dtype": "float32",
        "normalization": {"type": "scale", "scale": 0.5, "offset": -1},
    }
    with GrayRuntime.from_manifest(write_manifest(tmp_path, contract)) as runtime:
        result = runtime.predict(ImageInput(np.full((8, 8), 8, np.uint8), "HW"))
    np.testing.assert_array_equal(
        result.outputs["output"], np.full((1, 1, 8, 8), 3, np.float32)
    )


def test_unknown_backend_and_hash(tmp_path: Path, contract: dict) -> None:
    contract["runtime"]["backend"] = "tensorrt"
    with pytest.raises(BackendError):
        GrayRuntime.from_manifest(write_manifest(tmp_path, contract))
    contract["model"]["sha256"] = "0" * 64
    with pytest.raises(ManifestError, match="SHA-256"):
        GrayRuntime.from_manifest(write_manifest(tmp_path, contract))


@REAL
def test_provider_unavailable(monkeypatch) -> None:
    import onnxruntime

    monkeypatch.setattr(
        onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    with pytest.raises(BackendError, match="unavailable"):
        GrayRuntime.from_manifest(DEMO / "model" / "manifest.yaml", device="cuda")


@REAL
def test_execution_failure(monkeypatch) -> None:
    with GrayRuntime.from_manifest(DEMO / "model" / "manifest.yaml") as runtime:

        def fail(*args):
            raise RuntimeError("native error")

        monkeypatch.setattr(runtime._backend.session, "run", fail)
        with pytest.raises(InferenceError) as error:
            runtime.predict(np.zeros((1, 1, 8, 8), np.float32))
        assert isinstance(error.value.__cause__, RuntimeError)


@REAL
@pytest.mark.parametrize(
    "section,key,value",
    [
        ("input", "name", "wrong"),
        ("input", "shape", [1, 1, 9, 8]),
        ("input", "dtype", "float64"),
    ],
)
def test_model_contract_mismatch(tmp_path, contract, section, key, value) -> None:
    contract[section][key] = value
    with pytest.raises(BackendError):
        GrayRuntime.from_manifest(write_manifest(tmp_path, contract))


@REAL
def test_cli(tmp_path: Path, capsys) -> None:
    manifest = str(DEMO / "model" / "manifest.yaml")
    common = ["--manifest", manifest, "--input", str(DEMO / "input" / "sample.npy")]
    assert main(["inspect", manifest]) == 0
    output = tmp_path / "predictions"
    assert main(["infer", *common, "--output", str(output)]) == 0
    with np.load(output / "outputs.npz") as tensors:
        np.testing.assert_array_equal(
            tensors["output"], np.load(DEMO / "input" / "sample.npy")
        )
    assert (output / "metadata.json").is_file()
    assert main(["infer", *common, "--output", str(output)]) == 1
    assert main(["benchmark", *common, "--warmup", "0", "--iterations", "2"]) == 0
    assert main(["inspect", str(tmp_path / "missing")]) == 1
    for argv in (
        [],
        ["infer"],
        ["benchmark", *common, "--iterations", "0"],
        ["benchmark", *common, "--warmup", "-1"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
    assert "gray-sdk:" in capsys.readouterr().err


def test_optional_backend_not_imported() -> None:
    code = "import sys; import gray.sdk; assert 'onnxruntime' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_onnxruntime(monkeypatch) -> None:
    import builtins

    original = builtins.__import__

    def without_engine(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError("optional engine absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_engine)
    with pytest.raises(BackendError, match="Install"):
        GrayRuntime.from_manifest(DEMO / "model" / "manifest.yaml")


@REAL
def test_cuda_initialization_cannot_silently_fallback(monkeypatch) -> None:
    import onnxruntime

    class CpuFallback:
        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: CpuFallback())
    with pytest.raises(BackendError, match="not activated"):
        GrayRuntime.from_manifest(DEMO / "model" / "manifest.yaml", device="cuda:0")


@REAL
def test_context_closes_after_exception() -> None:
    runtime = GrayRuntime.from_manifest(DEMO / "model" / "manifest.yaml")
    with pytest.raises(ValueError, match="caller error"), runtime:
        raise ValueError("caller error")
    with pytest.raises(RuntimeClosedError):
        runtime.predict(np.zeros((1, 1, 8, 8), np.float32))


@REAL
def test_invalid_model_initialization(tmp_path, contract) -> None:
    path = write_manifest(tmp_path, contract)
    bad = tmp_path / "invalid.onnx"
    bad.write_bytes(b"not an ONNX model")
    content = yaml.safe_load(path.read_text())
    content["model"]["path"] = str(bad)
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(BackendError, match="initialization"):
        GrayRuntime.from_manifest(path)


@REAL
def test_output_contract_mismatch(tmp_path, contract) -> None:
    contract["outputs"][0]["name"] = "missing_output"
    with pytest.raises(BackendError, match="output names"):
        GrayRuntime.from_manifest(write_manifest(tmp_path, contract))


@REAL
def test_provider_conflict(tmp_path, contract) -> None:
    contract["runtime"]["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(BackendError, match="conflict"):
        GrayRuntime.from_manifest(write_manifest(tmp_path, contract), device="cpu")


def test_license_hook_runs_before_backend() -> None:
    def deny(manifest):
        assert manifest.model["name"] == "synthetic-identity-2d"
        raise PermissionError("local policy denied")

    with pytest.raises(PermissionError, match="local policy"):
        GrayRuntime.from_manifest(
            DEMO / "model" / "manifest.yaml", license_validator=deny
        )


@REAL
def test_cli_subprocess_exit_codes(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gray.sdk.cli",
            "inspect",
            str(DEMO / "model" / "manifest.yaml"),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    result = subprocess.run(
        [sys.executable, "-m", "gray.sdk.cli", "inspect", str(tmp_path / "missing")],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1 and b"ManifestError" in result.stderr


@REAL
def test_load_model_and_array_layout() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    with load_model(DEMO / "model") as model:
        result = model.predict(image, dim="HW")
        np.testing.assert_array_equal(result.outputs["output"], image[None, None])
        with pytest.raises(InputError):
            model.predict(image, dim="DHW")
        with pytest.raises(InputError):
            model.predict(ImageInput(image, "HW"), dim="HW")
    with load_model(DEMO / "model" / "manifest_3d.yaml") as model:
        result = model.predict(np.ones((4, 8, 8), np.float32), dim="DHW")
        assert result.outputs["output"].shape == (1, 1, 4, 8, 8)


@REAL
def test_simple_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    model = ["--model", str(DEMO / "model")]
    data = ["--input", str(DEMO / "input" / "sample.npy")]
    assert main(["info", *model]) == 0
    assert "Model: synthetic-identity-2d" in capsys.readouterr().out
    # Repeated runs must save independently rather than overwrite predictions.
    for _ in range(2):
        assert main(["infer", *model, *data]) == 0
        assert "Result saved to:" in capsys.readouterr().out
    folders = list((tmp_path / "outputs").iterdir())
    assert len(folders) == 2
    for folder in folders:
        with np.load(folder / "outputs.npz") as result:
            np.testing.assert_array_equal(
                result["output"], np.load(DEMO / "input" / "sample.npy")
            )
    assert main(["infer", *model, *data, "--output", str(tmp_path / "custom")]) == 0
    assert main(["infer", *model, *data, "--dim", "nchw"]) == 0
    assert main(["speed", *model, *data, "--runs", "2"]) == 0
    assert "Average time:" in capsys.readouterr().out
    for count in ("0", "-1"):
        with pytest.raises(SystemExit) as exc:
            main(["speed", *model, *data, "--runs", count])
        assert exc.value.code == 2


def test_model_folder_requires_settings(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        load_model(tmp_path)
