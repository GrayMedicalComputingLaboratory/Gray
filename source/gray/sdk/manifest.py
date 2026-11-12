"""Strict versioned YAML model contracts, independent of Hydra configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ManifestError
from .types import INPUT_DIMS


def _mapping(
    value: Any, required: set[str], optional: set[str], label: str
) -> dict[str, Any]:
    """Validate one manifest mapping against its exact allowed key set."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestError(f"{label} must be a mapping with string keys")
    if required - value.keys() or value.keys() - required - optional:
        raise ManifestError(
            f"{label}: missing {sorted(required - value.keys())}; "
            f"unsupported {sorted(value.keys() - required - optional)}"
        )
    return value


def _text(value: Any, label: str) -> None:
    """Require a human-readable nonempty string field."""
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a nonempty string")


@dataclass(frozen=True)
class ModelManifest:
    """Validated model contract. Load via :meth:`load`, not direct construction."""

    source: Path
    model_path: Path
    model: dict[str, Any]
    runtime: dict[str, Any]
    input: dict[str, Any]
    outputs: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, path: str | Path) -> ModelManifest:
        """Read schema 1.0 YAML; reject unknown fields and missing model files.

        Relative model paths resolve against the manifest directory. Input dim
        names are normalized to uppercase before returning the parsed contract.
        """
        try:
            import yaml
        except ImportError as error:
            raise ManifestError("Install gray[sdk] to load YAML manifests") from error
        try:
            source = Path(path).expanduser().resolve(strict=True)
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise ManifestError("Cannot read manifest YAML") from error
        root = _mapping(
            raw,
            {"schema_version", "model", "runtime", "input", "outputs"},
            set(),
            "manifest",
        )
        if root["schema_version"] != "1.0":
            raise ManifestError("Unsupported schema_version; expected string '1.0'")
        model = _mapping(
            root["model"],
            {"name", "version", "task_type", "format", "path"},
            {"sha256"},
            "model",
        )
        for name in ("name", "version", "task_type", "path"):
            _text(model[name], f"model.{name}")
        if model["format"] != "onnx":
            raise ManifestError("Only model.format onnx is supported")
        if "sha256" in model and (
            not isinstance(model["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", model["sha256"])
        ):
            raise ManifestError("sha256 must be 64 lowercase hexadecimal characters")
        model_path = (source.parent / model["path"]).resolve()
        if not model_path.is_file():
            raise ManifestError("Model path must reference an existing regular file")
        runtime = _mapping(root["runtime"], {"backend"}, {"providers"}, "runtime")
        _text(runtime["backend"], "runtime.backend")
        if "providers" in runtime:
            providers = runtime["providers"]
            if (
                not isinstance(providers, list)
                or not providers
                or any(
                    p not in ("CPUExecutionProvider", "CUDAExecutionProvider")
                    for p in providers
                )
                or len(set(providers)) != len(providers)
            ):
                raise ManifestError("providers must be unique CPU/CUDA provider names")
        spec = _mapping(
            root["input"],
            {"name", "dim", "dtype", "shape"},
            {"preprocessing"},
            "input",
        )
        _text(spec["name"], "input.name")
        if isinstance(spec["dim"], str):
            spec["dim"] = spec["dim"].upper()
        if not isinstance(spec["dim"], str) or spec["dim"] not in INPUT_DIMS:
            raise ManifestError("Unsupported input dim")
        if spec["dtype"] not in (
            "float32",
            "float64",
            "float16",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint16",
        ):
            raise ManifestError("Unsupported input dtype")
        shape = spec["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != len(spec["dim"])
            or any(v is not None and (type(v) is not int or v <= 0) for v in shape)
        ):
            raise ManifestError(
                "input.shape must match dim; use positive integers or null"
            )
        pre = _mapping(
            spec.get("preprocessing", {}),
            set(),
            {"cast_dtype", "normalization"},
            "preprocessing",
        )
        if "cast_dtype" in pre and pre["cast_dtype"] != spec["dtype"]:
            raise ManifestError("cast_dtype must equal model input dtype")
        if "normalization" in pre:
            norm = _mapping(
                pre["normalization"], {"type", "scale"}, {"offset"}, "normalization"
            )
            if norm["type"] != "scale" or not spec["dtype"].startswith("float"):
                raise ManifestError("scale normalization requires floating model input")
            for key in ("scale", "offset"):
                value = norm.get(key, 0)
                if type(value) not in (int, float) or not np.isfinite(value):
                    raise ManifestError(
                        "normalization scale/offset must be finite numbers"
                    )
        outputs = root["outputs"]
        if not isinstance(outputs, list) or not outputs:
            raise ManifestError("outputs must be a nonempty list")
        names = []
        for output in outputs:
            _mapping(output, {"name", "type"}, set(), "output")
            _text(output["name"], "output.name")
            _text(output["type"], "output.type")
            names.append(output["name"])
        if len(set(names)) != len(names):
            raise ManifestError("Duplicate output names")
        return cls(source, model_path, model, runtime, spec, tuple(outputs))
