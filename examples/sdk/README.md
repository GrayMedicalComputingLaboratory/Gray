# SDK 简单示例

这里的模型和数组都是合成测试数据，没有真实患者信息，也不用于诊断。
模型只会原样返回输入，用来检查 SDK 是否正常工作。

在仓库根目录安装：

```bash
python -m pip install './source/gray[sdk]'
```

运行模型，结果会自动保存到 `outputs/` 下的新文件夹：

```bash
gray-sdk infer --model examples/sdk/model --input examples/sdk/input/sample.npy
```

`--input` 也可直接传入 PNG、JPG、BMP、TIFF 或 WebP 图片。

`--model` 指模型文件夹，`--input` 指输入文件。再记住两个命令即可：

```bash
# 查看模型信息
gray-sdk info --model examples/sdk/model

# 测试运行速度
gray-sdk speed --model examples/sdk/model --input examples/sdk/input/sample.npy
```

测速会自动预热和重复运行，不需要手动填写参数。要指定次数，可以加 `--runs 20`。
结果路径和耗时会显示在终端。

Python 示例：

```python
import numpy as np
from gray.sdk import load_model

model = load_model("examples/sdk/model")
result = model.predict("images/sample.png")
print(result.outputs)
model.close()
```

`dim` 表示数组各维的含义，大小写都接受。`HW` 与 `hw` 等价；带通道图像支持
`CHW`（通道在前）和 `HWC`（通道在后）；二维灰度图用 `HW`，三维图像用 `DHW`（深度、高、宽），
对应模型配置是 `examples/sdk/model/manifest_3d.yaml`。
`NCHW`/`BCHW`（4维数组）表示多个二维图片；`NCDHW`/`BCDHW`（5维数组）表示多个三维体数据。
`NCHW` 中可用多个通道表示 2.5D 输入，例如相邻切片作为通道；纯三维体数据用 `NDHW` 或 `NCDHW`。
批量推理 Python 示例：

```python
import numpy as np
from gray.sdk import load_model

batch = np.load("examples/sdk/input/sample_batch.npy")
model = load_model("examples/sdk/model/manifest_batch.yaml")
result = model.predict_batch(batch, dim="nchw")
print(result.metadata["batch_size"], result.outputs)
model.close()
```

三维批量示例使用 `examples/sdk/model/manifest_batch_3d.yaml` 和
`examples/sdk/input/sample_batch_3d.npy`。`NDHW` 表示批量、深度、高、宽；
`NCDHW` 额外显式包含通道。
运行 `python examples/sdk/run_inference.py` 会验证二维和三维结果。

开发者需要交付安装包时，运行 `python build_sdk.py`，文件会生成在 `dist/sdk/`。
首次打包需要安装工具：`python -m pip install build`。
普通使用方不需要打包或重新生成示例模型。

重新生成测试模型（仅开发者需要 ONNX）：

```bash
python -m pip install onnx
python examples/sdk/generate_demo.py
```

完整的医学影像输入、GPU 和配置参考见 [SDK 文档](../../docs/SDK.md)。
