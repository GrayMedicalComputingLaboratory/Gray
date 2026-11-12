# Gray SDK 使用说明

## 先记住这三个命令

`--model` 是模型所在文件夹，`--input` 支持 `.npy`、PNG、JPG、BMP、TIFF 和 WebP。以下命令在仓库根目录执行。

```bash
# 查看模型
gray-sdk info --model examples/sdk/model

# 运行模型，结果自动保存到 outputs/ 下的新文件夹
gray-sdk infer --model examples/sdk/model --input examples/sdk/input/sample.npy

# 也可以直接传入常见图片文件
gray-sdk infer --model examples/sdk/model --input images/sample.png

# 测速，不用填写预热次数和重复次数
gray-sdk speed --model examples/sdk/model --input examples/sdk/input/sample.npy
```

测速会先自动预热，再运行 100 次，显示平均耗时，单位是毫秒。
需要更改次数时加 `--runs 20`。需要指定结果文件夹时加 `--output outputs/my-result`，
该文件夹必须尚不存在。默认使用 CPU；使用显卡时加 `--device cuda:0`。
模型文件夹中需要有 `manifest.yaml`，也就是模型说明文件，记录模型文件名、
输入尺寸和数据格式；模型提供方准备好后，使用方只需指定整个文件夹。
输入 `.npy` 是保存 NumPy 数组的文件，示例已经提供，可直接使用。

## 在 Python 中使用

### 读取图片并推理

`load_model` 负责加载模型，`predict` 可直接读取常见图片并执行推理。下面的例子用
`try/finally` 确保原生推理会话被关闭：

```python
from gray.sdk import load_model

model = load_model("examples/sdk/model")
try:
    result = model.predict("images/sample.png")
    print(result.task_type)
    print(result.outputs["output"].shape)
    print(result.latency_ms)
finally:
    model.close()
```

`model.predict(path, dim=None)` 支持 PNG、JPG/JPEG、BMP、TIFF 和 WebP。
如果需要在推理前检查解码数组或添加空间元数据，也可单独调用
`load_image(path, dim=None)`，它返回 `ImageInput`。灰度图片的数组为 `HW`，
彩色图片解码为 RGB，默认数组维度为 `HWC`。彩色模型若要求通道在前，可使用
`model.predict(path, dim="CHW")` 或 `load_image(path, dim="CHW")`。
函数会应用 EXIF 方向；调色板、CMYK 和带 alpha 的图片转为 RGB，alpha 会被丢弃。
灰度数据尽量保留文件位深。图片读取不会自动裁剪、缩放、重采样、窗宽窗位处理
或归一化。读取器适用于 2D 图片，不负责 DICOM 序列或 3D 医学影像；DICOM 请使用
后文的 `gray.dicom` API。安装 `gray[sdk]` 会一并安装 Pillow。

图片解码与模型预处理是不同步骤。若图片为 `uint8` 而模型输入为 `float32`，
请在 Manifest 里明确声明转换，例如：

```yaml
input:
  name: input
  dim: NCHW
  dtype: float32
  shape: [1, 1, 512, 512]
  preprocessing:
    cast_dtype: float32
    normalization:
      type: scale
      scale: 0.00392156862745098
```

SDK 先将单张 `HW` 图片插入模型所需的 batch 和 channel 轴，再按 Manifest 里的
`cast_dtype`、`scale` 和可选 `offset` 执行处理。未配置时不会自动把像素除以 255。
输入像素的含义、颜色空间和模型训练时的数据准备方法由模型提供方定义。

### NumPy 数组与 dim

`predict` 也接收 `numpy.ndarray`。传裸数组时，`dim` 必须描述数组当前轴的含义；
省略 `dim` 时 SDK 假定数组已经使用 Manifest 声明的维度顺序。

```python
import numpy as np
from gray.sdk import load_model

pixels = np.load("image.npy", allow_pickle=False)  # shape: (H, W)
model = load_model("examples/sdk/model/manifest.yaml")
try:
    result = model.predict(pixels, dim="HW")
    prediction = result.outputs["output"]
finally:
    model.close()
```

`dim` 名称不区分大小写。`HW` 是二维灰度图，`HWC`/`CHW` 是二维通道图，
`DHW` 是一个 3D 体数据。SDK 可重排已有轴，并插入或移除大小为 1 的 `N`/`B`
批次轴和 `C` 通道轴；不会猜测轴、把 `D` 当通道、交换 RGB/BGR、改变空间尺寸
或进行插值。所有非批次空间轴和通道数必须符合模型输入约定。

| `dim` | 含义 | 数组形状示例 |
| --- | --- | --- |
| `HW` | 一张灰度图 | `(H, W)` |
| `HWC` / `CHW` | 一张多通道 2D 图片 | `(H, W, C)` / `(C, H, W)` |
| `DHW` | 一个单通道 3D 体数据 | `(D, H, W)` |
| `NCHW` / `BCHW` | 一批 2D 图片 | `(N, C, H, W)` / `(B, C, H, W)` |
| `NDHW` / `BDHW` | 一批单通道 3D 体数据 | `(N, D, H, W)` / `(B, D, H, W)` |
| `NCDHW` / `BCDHW` | 一批多通道 3D 体数据 | `(N, C, D, H, W)` / `(B, C, D, H, W)` |

`B` 和 `N` 都表示 batch 轴，SDK 会映射到模型 Manifest 使用的批次轴。`NCHW`
里的 `C` 也可表示 2.5D 模型输入的相邻切片；真正的 3D 空间维度应写作 `DHW`。

### 批量推理

已有批量 NumPy 数组时，数组本身必须包含 `B` 或 `N`，并通过 `dim` 明确指出：

```python
import numpy as np
from gray.sdk import load_model

batch = np.load("examples/sdk/input/sample_batch.npy", allow_pickle=False)
model = load_model("examples/sdk/model/manifest_batch.yaml")
try:
    result = model.predict_batch(batch, dim="NCHW")
    print(result.metadata["batch_size"])
    print(result.outputs["output"].shape)
finally:
    model.close()
```

也可以传入图片路径、`ImageInput` 或数组组成的列表；路径会由 SDK 解码，`dim` 可
省略并从图片读取结果确定。裸数组列表需通过 `dim` 描述每一项，例如 `HW`、`HWC`
或 `DHW`。所有项需要有相同的形状、类型和 `dim`。模型输入必须声明 `B`/`N`；
静态 batch 只接受固定数量，动态 batch 则需在 Manifest 的对应 `input.shape` 项写
`null`。一次 `predict_batch` 使用一次模型调用。SDK v1 不提供自动目录扫描和通用图像
批处理器，应用可自行选择文件并调用此接口。

### 返回值和资源

`predict` 和 `predict_batch` 返回 `InferenceResult`：

| 属性 | 含义 |
| --- | --- |
| `task_type` | Manifest 中模型作者填写的任务标记 |
| `outputs` | `{输出名称: NumPy 数组}`，名称和语义遵循模型合同 |
| `metadata` | 模型哈希、SDK/后端版本、设备、输入形状、预处理和空间来源信息 |
| `latency_ms` | 输入校验、预处理和 ONNX 推理耗时，单位毫秒 |

SDK 不会对分类结果执行 softmax/argmax，不会将分割 logits 转成掩膜，也不会解析检测
框或作临床解释。调用方按模型定义处理 `outputs`。模型用完后调用 `model.close()`；
重复调用安全，关闭后再推理会抛出 `RuntimeClosedError`。

有 spacing、direction 或窗宽窗位来源信息时，可用 `ImageInput(data, dim, spacing=...,`
`direction=..., window_applied=...)` 明确携带。它不执行方向重排或物理空间重采样，
只校验并保留提供的来源元数据。

## 生成安装包

开发者在仓库根目录运行：

```bash
python build_sdk.py
```

生成的 `.whl` 文件就在 `dist/sdk/` 文件夹，它是交给其他项目安装的 Python 安装包。
已经安装 SDK 的使用方不需要再打包。如果提示缺少打包工具，先执行一次
`python -m pip install build`。下面保留详细的开发接口和配置参考。

Gray SDK loads local ONNX models without PyTorch, ClearML, a web service or
network activation. Python >=3.10 follows the existing package contract.
The existing training framework APIs and `gray` command remain supported.
This is a technical integration foundation, not a clinically validated device.

## Install and run

From this repository root (use a virtual environment):

```bash
python -m pip install './source/gray[sdk]'
python examples/sdk/run_inference.py
gray-sdk info --model examples/sdk/model
gray-sdk infer --model examples/sdk/model --input examples/sdk/input/sample.npy
gray-sdk speed --model examples/sdk/model --input examples/sdk/input/sample.npy
```

The SDK ships in the existing `gray` distribution. `[sdk]` adds PyYAML, CPU
ONNX Runtime and Pillow for PNG/JPG and other 2D image decoding. Existing base dependencies (Hydra, NumPy, pydicom, rich, SciPy)
are retained for compatibility. Neither PyTorch nor ONNX model generation is a
runtime requirement. The checked-in demo assets are generated synthetic data.
For development use `python -m pip install -e './source/gray[sdk-dev]'`.

## Public Python API

```python
import numpy as np
from gray.sdk import GrayRuntime, ImageInput

image = np.zeros((8, 8), dtype=np.float32)
runtime = GrayRuntime.from_manifest('examples/sdk/model/manifest.yaml', device='cpu')
result = runtime.predict(ImageInput(image, dim='HW', spacing=(0.5, 0.5),
                                    window_applied=False))
print(result.task_type, result.outputs['output'].shape, result.metadata)
print(result.latency_ms)
runtime.close()

volume = np.zeros((4, 8, 8), dtype=np.float32)
runtime = GrayRuntime.from_manifest('examples/sdk/model/manifest_3d.yaml')
result = runtime.predict(ImageInput(volume, dim='DHW', spacing=(2.0, 0.5, 0.5)))
runtime.close()
```

`load_model(path, device='cpu')` accepts a folder containing `manifest.yaml` or
an explicit YAML filename, and returns a GrayRuntime. For bare arrays,
`model.predict(array, dim='HW')` declares source axes without an ImageInput.
`GrayRuntime.from_manifest(path, device='cpu', license_validator=None)` validates
and loads the model. `predict(ImageInput | ndarray | str | Path)` returns
`InferenceResult`. Image paths are decoded internally as PNG/JPEG/BMP/TIFF/WebP.
`predict_batch` accepts an already batched ndarray or a sequence of paths,
ImageInput instances or arrays.
`close()` is idempotent; predicting after close raises `RuntimeClosedError`.
Call `close()` when finished to release native session resources. The runtime
also supports context managers, though explicit `close()` is sufficient.
Calls are serialized within one runtime. Create separate runtimes for concurrent
sessions. `ModelManifest.load(path)` validates YAML without loading ONNX Runtime.
`license_validator(manifest)` is an optional local hook called before engine load;
its exceptions propagate. No authorization service is implemented.

`InferenceResult(task_type, outputs, metadata, latency_ms)` preserves named NumPy
tensors. Classification scores, segmentation logits and detection arrays retain
the meaning declared by the model author. SDK v1 performs no implicit softmax,
argmax, thresholds, coordinate decoding or clinical interpretation. Postprocess
these tensors in the consuming project. Metadata includes model version/hash,
SDK/backend version, device/providers, source/tensor sizes, preprocessing and
supplied spatial geometry. Latency includes input validation, preprocessing and
session execution; benchmark reports mean, p50 and p95 after warmup, in ms.

## Medical image contract

| dim | Meaning | Example shape |
| --- | --- | --- |
| `HW` | One 2D grayscale image | `(H, W)` |
| `CHW` / `HWC` | One 2D image with channels first / last | `(C, H, W)` / `(H, W, C)` |
| `DHW` | One 3D volume; depth is its own axis | `(D, H, W)` |
| `NDHW` / `BDHW` | A batch of single-channel 3D volumes | `(batch, D, H, W)` |
| `NCHW` / `BCHW` | A batch of 2D channel images | `(batch, C, H, W)` |
| `NCDHW` / `BCDHW` | A batch of 3D channel volumes | `(batch, C, D, H, W)` |

D is depth, H row, W column, C channel, B/N batch. Arrays must
be nonempty, finite and real numeric. `dtype` optionally asserts the source
dtype; the actual source dtype is always recorded in `ImageInput.dtype`.
Bare ndarray without a `dim` argument **already uses the model's configured dim**;
otherwise supply `predict(array, dim='HW')`. Dimensions are never guessed.
Axis permutation and insertion/removal of singleton N/C are allowed. D cannot
be interpreted as C. Color conversion is not automatic.

Both `CHW` and `HWC` channel orders are supported. NumPy color conversion is not
automatic; `load_image` explicitly decodes color files as RGB. B and N are interchangeable
batch-axis names; the SDK maps either spelling to the batch axis declared by
the model. `BCHW` and `NCHW` both
describe multiple 2D images; `BCDHW` and `NCDHW` describe multiple 3D volumes.
`NCHW` with several channels can represent a 2.5D model that treats adjacent
slices as channels. `NDHW` represents batched single-channel volumes, and
`NCDHW` represents batched multi-channel volumes. Channel meaning is model-defined.
Choose the form matching the model's exported input. A single image may use
`ImageInput(image, dim="CHW")`; the SDK reorders dimensions to match the model.
For a list of individual images, use `model.predict_batch(images, dim="CHW")`.
All images must have matching shape and dtype. For an already assembled batch,
use `model.predict_batch(batch, dim="NCHW")` or `dim="BCHW"`. The model input
must include B or N.
A fixed-size model requires exactly the configured batch size. To choose batch
size at runtime, the ONNX model must have a dynamic batch dimension and the
manifest must set that position in `input.shape` to `null`. Outputs keep their
batch dimension, and
`result.metadata["batch_size"]` reports the number of images processed. One batch
uses one ONNX session call.

Spacing uses spatial axis order `(H,W)` or `(D,H,W)` in millimeters. Direction,
when provided, is an orthonormal spatial square direction-cosine matrix with
columns corresponding to those axes; the caller owns the physical coordinate
system. No origin mapping or physical resampling is performed. Geometry is
provenance for the source, not an assertion that every model output has the same
geometry. `window_applied` is True/False/None (unknown).

Use `gray.dicom.read_series(ordered_paths)` for already selected/sorted DICOM
files; it applies modality LUT/rescale and returns `(D,H,W)`. Pair it with
`get_spacing`. Windowing, MONOCHROME1 handling and resampling use the existing
explicit DICOM functions. Series discovery, selection, patient data and output
geometry restoration belong to the application. If resizing/resampling is needed,
do it explicitly before SDK invocation and supply updated geometry. Use nearest
interpolation for masks/labels and explicitly selected linear interpolation for
continuous intensities. SDK performs **no interpolation** and rejects resize fields.

## Manifest schema 1.0

See `examples/sdk/model/manifest.yaml`. YAML uses `safe_load`; unknown fields,
missing required fields and unsupported versions fail. Quote `schema_version`
as `"1.0"`. Model paths resolve relative to the manifest, independent of cwd;
absolute paths are allowed. Deploy trusted manifests and trusted models.

| Field | Contract |
| --- | --- |
| schema_version | Required string `1.0` |
| model | Required nonempty name, version, task_type, path; format `onnx` |
| model.sha256 | Optional lowercase 64-digit SHA-256, verified before loading |
| runtime.backend | Required `onnxruntime`; other names fail during initialization |
| runtime.providers | Optional exact requested order: CPU alone or CUDA then CPU |
| input | Required name, dim (case-insensitive), dtype, shape |
| input.shape | One positive int per dim; null for ONNX symbolic dimensions (use null for a variable batch size) |
| input.dtype | float16/32/64, uint8/16, int8/16/32/64 |
| input.preprocessing | Optional cast_dtype and normalization only |
| outputs | Nonempty list of unique name/type pairs, matching all ONNX outputs in order |

ONNX model input must be one tensor and agree with the manifest name, dtype and
rank. Static ONNX dimensions must match static manifest values; symbolic model
dimensions may be constrained by the manifest. Runtime checks concrete arrays.
Outputs must be tensors. Task type and output type are author-defined strings.
No general multi-input, ONNX sequence/map output or task decoding is supported.

No default normalization/cast occurs. Explicit preprocessing example:

```yaml
preprocessing:  # nested under input
  cast_dtype: float32  # must equal input.dtype
  normalization:
    type: scale
    scale: 0.00392156862745098
    offset: 0.0
```

This computes `cast(input) * scale + offset`, rejecting nonfinite results.
Manifest dictionaries are validated at load; treat them as read-only after load.

## CPU, CUDA and backend extensions

CPU uses only CPUExecutionProvider. In a separate GPU environment install
`python -m pip install './source/gray[sdk-cuda]'`, then call
`GrayRuntime.from_manifest(path, device='cuda:0')` or CLI `--device cuda:0`.
Install CPU and GPU ONNX Runtime in separate environments (they share a Python
namespace). Compatible NVIDIA driver, CUDA and cuDNN must be installed according
to the selected ONNX Runtime version. No torch installation is needed.

CUDA requests require an available and activated CUDAExecutionProvider; missing
CUDA initialization fails rather than falling back to a CPU-only session. CUDA
sessions allow CPU placement of operators unsupported by CUDA; this is not a
promise that every operator runs on GPU. If providers are specified in YAML they
must match the selected device. Omit providers for a portable CPU/GPU manifest.
`sdk/backend.py:Backend` defines the run/close and version/providers boundary for
future engines; TensorRT is not implemented.

## CLI and errors

`infer`, `info` and `speed` are the recommended commands. They use `--model` for
a model folder or settings file. `infer` defaults to a fresh `outputs/sdk-...`
directory; `speed` automatically warms up 10 times and measures 100 runs (change
with `--runs`). These commands print readable summaries. The older `inspect`,
`infer --manifest`, and `benchmark --manifest` commands retain JSON output and
their existing options for integrations.

`gray-sdk --help` and each subcommand's `--help` describe parameters. `inspect`
checks the real model session, not only YAML. CLI accepts numeric `.npy` files
with pickle disabled and PNG/JPG/BMP/TIFF/WebP images. `--dim` supplies array
dimensions; image files are single 2D images, while batches use `.npy` with N,
for example `--dim NCHW`. `infer` creates a **new**
output directory with `outputs.npz` and `metadata.json`; it refuses overwrite.
It saves predictions only, never the input array. Output archive keys follow model
output names. Benchmark writes no image/output files. Exit codes: 0 success,
1 validation/IO/engine failure, 2 invalid CLI arguments.

All SDK operational exceptions derive from `GraySDKError`: `ManifestError`,
`InputError`, `BackendError`, `InferenceError`, `RuntimeClosedError`. Engine
exceptions are chained for controlled debugging. CLI prints safe SDK messages;
do not log chained engine errors without reviewing your application's privacy
policy. The SDK does not emit per-image logs and never writes image content,
patient information or inputs to a log. Predictions may still contain
sensitive derived information; the application owns storage and access controls.

Troubleshooting: install extras for missing engines/YAML; use the exact manifest
dim and float32 data or explicit cast for dtype errors; align ONNX tensor names
and shapes for contract errors; check relative model file placement and SHA-256
for bundle errors; select CPU when CUDA libraries/hardware are unavailable.

## Build, offline Linux delivery and verification

```bash
python -m pip install build
python build_sdk.py
# On a machine matching the target Linux architecture/Python:
python -m pip download --dest wheelhouse './dist/sdk/gray-0.1.0-py3-none-any.whl[sdk]'
# Copy wheelhouse, model bundle, example and docs to offline target:
python -m pip install --no-index --find-links wheelhouse 'gray[sdk]==0.1.0'
python examples/sdk/run_inference.py
python -m pytest tests -q
ruff check source/gray/sdk tests/test_sdk.py examples/sdk
```

The Gray wheel is Python-only; ONNX Runtime and NumPy/SciPy wheels depend on
platform/Python. Resolve and retain an exact dependency lock per delivery target,
hash wheelhouse files and test in the target Linux image. Do not reuse Windows
native wheels on Linux. Models/demo assets are delivered beside the wheel, not
embedded in the package. No external model download or network call is required
for inference. Test dependencies ONNX/pytest/ruff are not runtime requirements.
`tests/test_sdk.py` explicitly skips real-engine tests if onnxruntime is absent.
Linux CI builds/installs a wheel and runs real CPU tests on Python 3.10 and 3.12.

See `SDK_VALIDATION.md` for results and remaining commercial delivery work.
