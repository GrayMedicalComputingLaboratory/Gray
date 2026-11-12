"""Generate tiny identity ONNX fixtures and synthetic arrays; no patient data."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def generate(root: Path) -> None:
    """Write deterministic 2D/3D identity model bundles (requires onnx)."""
    (root / "model").mkdir(parents=True, exist_ok=True)
    (root / "input").mkdir(parents=True, exist_ok=True)
    for suffix, shape in (("", [1, 1, 8, 8]), ("_3d", [1, 1, 4, 8, 8])):
        graph = helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])],
            "synthetic-identity",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
        )
        onnx.checker.check_model(model)
        onnx.save(model, root / "model" / f"model{suffix}.onnx")
        np.save(
            root / "input" / f"sample{suffix}.npy",
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        )

    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "synthetic-dynamic-batch-identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 1, 8, 8])],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, ["batch", 1, 8, 8]
            )
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.checker.check_model(model)
    onnx.save(model, root / "model" / "model_batch.onnx")
    batch = np.arange(3 * 64, dtype=np.float32).reshape(3, 1, 8, 8)
    np.save(root / "input" / "sample_batch.npy", batch)

    batch_3d_graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "synthetic-dynamic-3d-batch-identity",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["batch", 1, 4, 8, 8]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, ["batch", 1, 4, 8, 8]
            )
        ],
    )
    batch_3d_model = helper.make_model(
        batch_3d_graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(batch_3d_model)
    onnx.save(batch_3d_model, root / "model" / "model_batch_3d.onnx")
    batch_3d = np.arange(2 * 4 * 8 * 8, dtype=np.float32).reshape(2, 1, 4, 8, 8)
    np.save(root / "input" / "sample_batch_3d.npy", batch_3d)


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
