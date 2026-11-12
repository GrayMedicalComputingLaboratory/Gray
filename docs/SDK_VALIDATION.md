# SDK v1 validation and delivery record

## 简化入口复验（2026-11-12）

图片输入复验：`predict("image.png")` 和 `predict_batch(["a.png", "b.png"])`
由 SDK 内部解码，CLI `infer` 也可直接读取图片。支持灰度 HW 与 RGB HWC/CHW，
处理 EXIF 方向；像素不会自动缩放或归一化。Pillow 加入 `gray[sdk]` 安装选项。
PNG/JPG 真实 CPU 推理和路径列表批量推理均通过。

API 整理复验：SDK 函数与类方法均补充了 docstring，AST 检查没有遗漏。
`write_json` 复用 `gray.utils.io` 的原子写入；模型 SHA-256 复用
`gray.utils.hashing.sha256`。现有 DICOM 窗宽窗位、重采样和解码与 SDK 的
模型输入轴整理职责不同，保持分离。Python 示例显式调用 `model.close()`，
不使用 context manager 写法；该 API 为兼容性仍保留 context manager 支持。

批次支持复验：支持序列和 NumPy 批数组输入；Manifest 和 `dim` 支持 `B` 与
`N` 两种批次轴名称。已添加动态批次 ONNX 合成示例和 `predict_batch` API。
`NCHW`/`BCHW` 表示批量 2D，`NCDHW`/`BCDHW` 表示批量 3D；通道可表示 2.5D
模型的相邻切片。真实 ONNX Runtime 测试覆盖 CHW/HWC 转换、动态批量 2D/3D、
N/B 两种写法、CLI 输出以及无批次轴的清晰错误。

后续按用户命名要求，将数组维度参数统一由 `layout` 改为 `dim`：
`ImageInput(..., dim="HW")`、`model.predict(array, dim="HW")`、CLI `--dim HW`，
Manifest 改用 `input.dim`，结果元数据改用 `input_dim`。旧的 `layout` 参数或
Manifest 字段不再接受，避免两种名称并存。

新增面向使用者的入口：`load_model`、`predict(array, dim=...)`，以及
`gray-sdk infer/info/speed --model ...`。只需指定模型文件夹和输入文件；
结果自动保存，测速自动预热和重复运行。旧 API 和命令仍通过兼容性测试。

- 图片 API 加入后的完整测试 **75 passed**，包括 SDK 回归和批次测试，覆盖二维/三维简单 API、
  动态二维和三维批次、重复运行独立保存、输出路径、测速及模型错误。
  Windows 主机限制 pytest 缓存目录，测试本身全部通过；无代码告警。
- ruff 检查和格式检查通过，共 13 个 Python 文件。
- `python build_sdk.py` 已真实执行，安装包生成在
  `dist/sdk/gray-0.1.0-py3-none-any.whl`。旧 `dist/` 根目录安装包是前一版产物，
  请使用 `dist/sdk/` 中的新文件。
- 新安装包已从本地缓存离线安装到第三个独立环境，离开源码目录运行
  `info`、`infer`、`speed` 均成功；`load_model`、`predict(array, dim='HW')` 和
  `predict_batch(array, dim='NCHW')`
  也已通过真实 CPU 推理验证。CLI 默认运行 100 次，无需填写预热参数。
- Linux/CUDA 仍未实机运行，下文限制继续适用。

以下是初版交付的详细验证记录。

Validation date: 2026-11-12. Host: Windows, CPython 3.12.14 x64.
Installed engines: ONNX Runtime 1.31.1, ONNX 1.23.2, NumPy 2.5.4.
Available providers: AzureExecutionProvider and CPUExecutionProvider; tests used
CPUExecutionProvider explicitly. No PyTorch is installed in the wheel environment.

## Executed successfully

- `python -m pytest tests -q --basetemp .pytest-tmp-sdk-delivery`:
  **64 passed in 3.03 seconds**, no skips/warnings. Includes 26 existing tests
  and 38 new SDK tests. Basetemp is inside the writable workspace because the
  host's default pytest temporary directory is restricted by the sandbox.
- SDK tests cover manifest fields/version/paths, image rank/dim/dtype/geometry,
  explicit cast/normalization, real CPU 2D/3D ONNX tensor inference, model contract
  mismatches, failed inference, session lifecycle, missing backend dependency,
  SHA-256 mismatch, provider conflict/unavailability/fallback rejection, local
  license hook and CLI errors, output files and process exit codes.
- `ruff check source/gray/sdk tests/test_sdk.py examples/sdk` and
  `ruff format --check source/gray/sdk tests/test_sdk.py examples/sdk`: passed.
  No pre-existing source files were reformatted.
- `python examples/sdk/run_inference.py`: exact array equality verified for both
  checked-in synthetic identity models, via real ONNX Runtime (not a mock).
- `python -m build --wheel --no-isolation source/gray --outdir dist`: built
  `gray-0.1.0-py3-none-any.whl`, including gray.sdk and both command entry points.
- Installed that wheel into a separate virtual environment with runtime extras.
  Changed cwd away from source, verified `gray.__file__` resides in site-packages,
  and ran both real 2D/3D examples plus `gray-sdk inspect`. Neither `onnx` (model
  generator) nor `torch` was installed in this environment.
- Reinstalled the wheel using `uv --offline` and the local dependency cache;
  dependency resolution and installation completed without network access.
- Existing `gray --help` still exposes train/validate/analyze and original options.
- CPU CLI benchmark on the 8x8 identity fixture, warmup 10, iterations 100:
  mean 0.028216 ms, p50 0.027900 ms, p95 0.031210 ms. This measures validation,
  preprocessing and engine execution; it does not predict clinical-model speed.

## Not executed on this host

- Actual CUDA execution: CUDAExecutionProvider is unavailable. Failure paths
  are tested deterministically, including engine initialization returning only
  CPU providers. This is not evidence of successful GPU execution.
- Linux installation/runtime and Python 3.10 execution: the host is Windows
  with Python 3.12. A Linux GitHub Actions workflow is included for Python 3.10
  and 3.12, building/installing the wheel and running real CPU inference. It has
  not been pushed or run remotely in this task.
- Fresh Linux wheelhouse offline install: local cached Windows offline install
  passed; target Linux wheels must still be resolved and tested on matching
  architecture/Python. The normal isolated build command is documented; local
  build used the already-installed setuptools to avoid build-time downloads.

## Scope and commercial delivery gaps

SDK v1 supports one ONNX tensor input and named tensor outputs, with explicit
dim conversion and optional affine intensity preparation. Color conversion,
resizing, physical reorientation, multi-input models, tiled inference, task
decoders and spatial output restoration remain application responsibilities.
The manifest checksum covers the ONNX file itself; external-data sidecars need
separate bundle checksums and customer packaging validation.

Remaining delivery work: test on each target Linux/CPU/GPU image, pin exact
dependencies and produce SBOM/vulnerability records, benchmark customer models,
define model cards and clinical validation, approve logging/storage/audit policies,
sign releases and validate rollback/support procedures. C/C++ ABI, TensorRT and
device authorization/license-server systems are outside this Python phase.

## Git environment and review artifacts

The supplied `Gray-main` directory is not itself a Git checkout. The discovered
Git root is its parent company repository, whose deny-by-default ignore rule
ignores all of `Gray-main`; no Gray baseline source files are tracked there.
The requested `feature/inference-sdk-v1` branch was created on that detected root.
The existing modified company standards document was preserved. No force-add,
commit, remote push or merge was performed, and no ignore rule outside the
workspace was modified. Consequently parent `git diff` cannot represent the SDK
source changes. A task-specific text diff, binary assets in a complete source
bundle, and a SHA-256 changed-file inventory are provided under
`outputs/sdk-review/` for migration into the actual Gray repository.
