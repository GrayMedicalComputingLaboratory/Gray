"""ONNX adapter boundary. No training runtime imports or network operations."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .exceptions import BackendError, InferenceError
from .manifest import ModelManifest


class Backend(Protocol):
    """Minimal extension contract for future inference engines."""

    version: str
    providers: list[str]

    def run(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Execute one prepared tensor."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...


class OnnxBackend:
    """One-input ONNX Runtime session with explicit provider selection."""

    def __init__(self, manifest: ModelManifest, device: str) -> None:
        """Create a checked session for the explicitly selected device."""
        if not isinstance(device, str):
            raise BackendError("device must be cpu, cuda, or cuda:N")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise BackendError(
                "Install gray[sdk] (CPU) or gray[sdk-cuda] (CUDA)"
            ) from error
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
            options = [{}]
        elif device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit()):
            index = 0 if device == "cuda" else int(device[5:])
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            options = [{"device_id": str(index)}, {}]
        else:
            raise BackendError("device must be cpu, cuda, or cuda:N")
        configured = manifest.runtime.get("providers")
        if configured is not None and configured != providers:
            raise BackendError("Manifest providers conflict with requested device")
        if any(p not in ort.get_available_providers() for p in providers):
            raise BackendError("Requested execution provider unavailable")
        try:
            self.session = ort.InferenceSession(
                str(manifest.model_path), providers=providers, provider_options=options
            )
            self.session.disable_fallback()
            self.providers = self.session.get_providers()
            if self.providers != providers:
                raise BackendError("Requested execution provider was not activated")
            self.version = ort.__version__
            inputs = self.session.get_inputs()
            spec = manifest.input
            if len(inputs) != 1 or inputs[0].name != spec["name"]:
                raise BackendError(
                    "ONNX input names do not match single-input manifest"
                )
            node = inputs[0]
            types = {"float32": "float", "float64": "double", "float16": "float16"}
            if node.type != f"tensor({types.get(spec['dtype'], spec['dtype'])})":
                raise BackendError("ONNX input dtype differs from manifest")
            if len(node.shape) != len(spec["shape"]) or any(
                isinstance(a, int) and a != b for a, b in zip(node.shape, spec["shape"])
            ):
                raise BackendError("ONNX input shape differs from manifest")
            expected = [item["name"] for item in manifest.outputs]
            if [item.name for item in self.session.get_outputs()] != expected:
                raise BackendError("ONNX output names/order differ from manifest")
            self.input_name = node.name
            self.output_names = expected
        except BackendError:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise BackendError("ONNX session initialization failed") from error

    def run(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Run a tensor and translate engine errors without exposing input data."""
        try:
            tensors = self.session.run(self.output_names, {self.input_name: data})
            if any(not isinstance(tensor, np.ndarray) for tensor in tensors):
                raise InferenceError("Only tensor outputs are supported")
            return dict(zip(self.output_names, tensors, strict=True))
        except Exception as error:
            raise InferenceError("ONNX inference failed") from error

    def close(self) -> None:
        """Drop the native session reference; safe to call repeatedly."""
        self.session = None
