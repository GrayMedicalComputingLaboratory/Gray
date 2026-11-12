"""Offline model lifecycle and standardized inference provenance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

from gray import __version__
from gray.utils.hashing import sha256

from .backend import Backend, OnnxBackend
from .exceptions import BackendError, InputError, ManifestError, RuntimeClosedError
from .image import load_image
from .manifest import ModelManifest
from .preprocess import prepare
from .types import BATCH_AXES, ImageInput, InferenceResult


class GrayRuntime:
    """One model/session. close and predict are serialized per runtime.

    license_validator is an optional local policy hook, called before loading
    the engine. No licensing, activation, networking or image logging is built in.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        *,
        device: str = "cpu",
        license_validator: Callable[[ModelManifest], None] | None = None,
    ) -> None:
        """Verify model identity and initialize the selected inference backend."""
        self._lock = RLock()
        self.manifest = manifest
        self.device = device
        if license_validator is not None:
            license_validator(manifest)
        try:
            self.model_sha256 = sha256(manifest.model_path)
        except OSError as error:
            raise ManifestError("Cannot read model for SHA-256 verification") from error
        if manifest.model.get("sha256", self.model_sha256) != self.model_sha256:
            raise ManifestError("Model SHA-256 mismatch")
        if manifest.runtime["backend"] != "onnxruntime":
            raise BackendError("Unsupported backend; expected onnxruntime")
        self._backend: Backend | None = OnnxBackend(manifest, device)

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        license_validator: Callable[[ModelManifest], None] | None = None,
    ) -> GrayRuntime:
        """Validate YAML, verify checksum, and load the selected local engine."""
        return cls(
            ModelManifest.load(path), device=device, license_validator=license_validator
        )

    def predict(
        self,
        image: ImageInput | np.ndarray | str | Path,
        *,
        dim: str | None = None,
    ) -> InferenceResult:
        """Return named tensors from an image path, array, or ImageInput.

        Image paths are decoded by :func:`load_image`. For a bare array, dim
        declares its axes (for example ``HW``); when omitted the array must
        already use the model's configured dim. ImageInput carries its own dim
        and spatial metadata. latency_ms includes validation, preprocessing
        and backend execution. Spatial metadata is copied as provenance.
        """
        with self._lock:
            if self._backend is None:
                raise RuntimeClosedError("Runtime is closed")
            start = perf_counter()
            if isinstance(image, (str, Path)):
                image = load_image(image, dim=dim)
                dim = None
            elif isinstance(image, np.ndarray):
                image = ImageInput(
                    image,
                    dim=dim if dim is not None else self.manifest.input["dim"],
                )
            elif dim is not None:
                raise InputError("Use dim on an array, or set it inside ImageInput")
            if not isinstance(image, ImageInput):
                raise InputError("predict requires ImageInput or ndarray")
            # Revalidate in case the caller modified the array after construction.
            image.__post_init__()
            data = prepare(image, self.manifest.input)
            return self._execute(data, image, start)

    def _execute(
        self, data: np.ndarray, image: ImageInput, start: float
    ) -> InferenceResult:
        """Run an already prepared tensor and attach standard provenance."""
        if self._backend is None:
            raise RuntimeClosedError("Runtime is closed")
        outputs = self._backend.run(data)
        metadata = {
            "model_name": self.manifest.model["name"],
            "model_version": self.manifest.model["version"],
            "model_sha256": self.model_sha256,
            "sdk_version": __version__,
            "backend": self.manifest.runtime["backend"],
            "backend_version": self._backend.version,
            "providers": list(self._backend.providers),
            "device": self.device,
            "input_shape": list(image.data.shape),
            "input_dim": image.dim,
            "tensor_shape": list(data.shape),
            "spacing": image.spacing,
            "direction": image.direction,
            "window_applied": image.window_applied,
            "preprocessing": self.manifest.input.get("preprocessing", {}),
            "output_types": {o["name"]: o["type"] for o in self.manifest.outputs},
        }
        return InferenceResult(
            self.manifest.model["task_type"],
            outputs,
            metadata,
            (perf_counter() - start) * 1000,
        )

    def predict_batch(
        self,
        images: Sequence[ImageInput | np.ndarray | str | Path] | np.ndarray,
        *,
        dim: str | None = None,
    ) -> InferenceResult:
        """Infer a batch of images with one ONNX session call.

        A batched ndarray must include B or N and declare it in ``dim`` (for
        example BCHW or NCHW). For a sequence, each item is one image/volume and
        ``dim`` describes each item's axes (for example CHW or DHW). Sequence
        items must have matching transformed shapes and dtype. The model input
        Model input must declare B or N; a fixed ONNX batch accepts exactly
        that many items, while a dynamic batch uses ``null`` at that position
        in manifest input.shape.
        """
        start = perf_counter()
        target = self.manifest.input["dim"]
        if isinstance(images, np.ndarray):
            if not isinstance(dim, str) or not BATCH_AXES.intersection(dim.upper()):
                raise InputError(
                    "A batched ndarray requires dim containing B or N, such as BCHW"
                )
            result = self.predict(images, dim=dim)
            batch_token = next(axis for axis in "BN" if axis in dim.upper())
            count = images.shape[dim.upper().index(batch_token)]
            result.metadata.update(batch_size=count, batch_dim=batch_token)
            return replace(result, latency_ms=(perf_counter() - start) * 1000)
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            raise InputError(
                "predict_batch requires a sequence of images or a batched ndarray"
            )
        if not images:
            raise InputError("predict_batch requires at least one image")
        batch_token = next((axis for axis in "BN" if axis in target), None)
        if batch_token is None:
            raise InputError(
                "The model input dim must include B or N for batch inference"
            )

        if any(isinstance(item, (str, Path)) for item in images):
            images = [
                load_image(item, dim=dim)
                if isinstance(item, (str, Path))
                else item
                for item in images
            ]
        first = images[0]
        if dim is not None and not isinstance(dim, str):
            raise InputError("dim must be a string such as CHW or DHW")
        source_dim = (
            dim.upper()
            if isinstance(dim, str)
            else (first.dim if isinstance(first, ImageInput) else None)
        )
        if source_dim is None:
            raise InputError("Set dim for ndarray batch items, such as CHW or DHW")
        if BATCH_AXES.intersection(source_dim):
            raise InputError(
                "Sequence items are individual images; dim must not contain B or N"
            )

        prepared: list[np.ndarray] = []
        image_inputs: list[ImageInput] = []
        for item in images:
            if isinstance(item, (str, Path)):
                item = load_image(item, dim=source_dim)
            if isinstance(item, np.ndarray):
                item = ImageInput(item, dim=source_dim)
            if not isinstance(item, ImageInput):
                raise InputError("Batch items must be ImageInput or NumPy ndarray")
            item.__post_init__()
            if item.dim != source_dim:
                raise InputError("All batch items must use the same dim")
            image_inputs.append(item)
            prepared.append(prepare(item, self.manifest.input, allow_batch_size=True))

        batch_axis = target.index(batch_token)
        expected_batch = self.manifest.input["shape"][batch_axis]
        if expected_batch is not None and expected_batch != len(prepared):
            raise InputError(
                f"Model requires a batch of exactly {expected_batch} items"
            )
        try:
            data = np.ascontiguousarray(np.concatenate(prepared, axis=batch_axis))
        except ValueError as error:
            raise InputError(
                "Batch items must have matching shapes and dtypes"
            ) from error
        batch_image = ImageInput(data, dim=target)
        with self._lock:
            result = self._execute(data, batch_image, start)
        result.metadata.update(
            batch_size=len(prepared),
            batch_dim=batch_token,
            source_dim=source_dim,
            item_spacing=[image.spacing for image in image_inputs],
            item_direction=[image.direction for image in image_inputs],
            item_window_applied=[image.window_applied for image in image_inputs],
        )
        return replace(result, latency_ms=(perf_counter() - start) * 1000)

    def close(self) -> None:
        """Release the session. Repeated calls are harmless."""
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None

    def __enter__(self) -> GrayRuntime:  # noqa: PYI034 -- Python 3.10 without typing_extensions
        """Enter an open runtime context."""
        if self._backend is None:
            raise RuntimeClosedError("Runtime is closed")
        return self

    def __exit__(self, *args: object) -> None:
        """Release resources without suppressing exceptions."""
        self.close()


def load_model(path: str | Path, *, device: str = "cpu") -> GrayRuntime:
    """Load a model folder containing manifest.yaml, or a specific YAML file.

    Returns the same GrayRuntime used by the advanced API. Use it as a context
    manager or call close() when finished. No input properties are guessed.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "manifest.yaml"
    return GrayRuntime.from_manifest(source, device=device)
