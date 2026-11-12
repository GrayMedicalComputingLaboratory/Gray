"""Run a local synthetic 2D/3D example from any working directory."""

from pathlib import Path

import numpy as np
from gray.sdk import ImageInput, load_model

root = Path(__file__).resolve().parent
for suffix, dim, spacing in (
    ("", "HW", (0.5, 0.5)),
    ("_3d", "DHW", (2.0, 0.5, 0.5)),
):
    sample = np.load(root / "input" / f"sample{suffix}.npy", allow_pickle=False)
    model = load_model(root / "model" / f"manifest{suffix}.yaml")
    try:
        result = model.predict(
            ImageInput(sample[0, 0], dim=dim, spacing=spacing, window_applied=False)
        )
    finally:
        model.close()
    np.testing.assert_array_equal(result.outputs["output"], sample)
    print(
        result.task_type,
        result.outputs["output"].shape,
        result.latency_ms,
        result.metadata,
    )

for suffix, dim in (("", "NCHW"), ("_3d", "NCDHW")):
    batch = np.load(root / "input" / f"sample_batch{suffix}.npy", allow_pickle=False)
    model = load_model(root / "model" / f"manifest_batch{suffix}.yaml")
    try:
        result = model.predict_batch(batch, dim=dim)
    finally:
        model.close()
    np.testing.assert_array_equal(result.outputs["output"], batch)
    print(f"batch {dim}", result.outputs["output"].shape, result.metadata["batch_size"])
