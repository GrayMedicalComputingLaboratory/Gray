<p align="center">
  <img src="https://raw.githubusercontent.com/GrayMedicalComputingLaboratory/Gray/main/docs/assets/gray.png" alt="Gray Medical Computing Laboratory" width="360">
</p>

<p align="center">
  <a href="#中文说明">中文 README</a> ·
  <a href="#english">English README</a>
</p>

<a id="中文说明"></a>

# Gray 中文说明

Gray 是 Gray Medical Computing Laboratory 的轻量、可复用训练研究基础框架。
它提供配置身份、可复现性、产物路径、生命周期调度和扩展契约；模型、数据集、指标、增强、损失和推理仍由具体项目负责。

## 文档

- [架构说明](https://github.com/GrayMedicalComputingLaboratory/Gray/blob/main/docs/ARCHITECTURE.md)：框架模块的职责和所有权。
- [API 参考](https://graymedicalcomputinglaboratory.github.io/Gray/api.html)：可渲染的接口文档，包含参数、返回值、异常和源码链接。
- [API 源文件](https://github.com/GrayMedicalComputingLaboratory/Gray/blob/main/docs/api.html)：仓库中的静态文档源文件。

## 安装

```bash
python -m pip install --upgrade \
  "gray @ git+ssh://git@github.com/GrayMedicalComputingLaboratory/Gray.git@main#subdirectory=source/gray"
```

本地开发安装：

```bash
python -m pip install -e /path/to/Gray/source/gray
```

启用 ClearML 实验管理：

```bash
python -m pip install -e "/path/to/Gray/source/gray[train]"
```

最小训练示例见 [`examples/main.py`](https://github.com/GrayMedicalComputingLaboratory/Gray/blob/main/examples/main.py)，完成 `clearml-init` 后运行：

```bash
python examples/main.py
```

常用训练入口只需要加载配置并提交训练函数：

```python
from gray import load_config
from gray.experiment import init_task

config = load_config("configs/exp_001.yaml")
task = init_task(config)
result = train(config, task)
```

`train(config, task)` 接收原生 ClearML `Task`。日志、训练曲线和 debug sample 统一通过绑定 tracker 的 `GrayLogger` 上报；参数、产物和模型仍使用 `task.connect()`、`task.upload_artifact()` 和 `task.update_output_model()`。正常进程退出与未捕获异常由 ClearML 自动标记为完成或失败。

## 项目集成

每个项目声明自己的生命周期入口；Gray 读取解析后的配置并调用对应函数。

```yaml
# configs/exp_001.yaml
# 文件名而不是 YAML 字段定义 experiment_id。
hydra:
  job:
    chdir: false
project:
  output_root: ../../outputs
  entrypoints:
    train: my_project.training:train
    validate: my_project.validation:validate
    analyze: my_project.analysis:analyze
runtime:
  seed: 42
  device: cuda:0
```

```python
from gray.runtime import seed_everything

def train(config: dict, trial=None) -> dict:
    seed_everything(config["runtime"]["seed"])
    # 在项目中构建 dataset、model、optimizer 和训练循环。
    return {"experiment_id": config["experiment_id"], "status": "complete"}
```

运行生命周期阶段：

```bash
python -m gray.cli train --config configs/exp_001.yaml
python -m gray.cli validate --config configs/exp_001.yaml
python -m gray.cli analyze --config configs/exp_001.yaml
```

Hydra 只解析一份自包含 YAML，不支持 `defaults`、配置组或自动切换工作目录。
阶段函数返回字典时，Gray 写入 `<output_root>/<experiment_id>/<stage>/summary.json`。

## 测试时增强

`tta` 返回命名的 2D/3D 推理变体；概率或 logits 如何聚合仍由项目决定。
3D TTA 不反转或旋转 Z/depth 轴，只对每个 slice 的 H/W 平面执行相同变换。

```python
from gray.inference import tta

variants = tta(
    volume,
    dim="3d",
    horizontal_flip=True,
    vertical_flip=False,
    rotate90_angles=[0, 90, 180, 270],
)
logits = [model(variant) for variant in variants.values()]
```

## DICOM 处理

`gray.dicom` 使用 pydicom 解码 DICOM，并用 NumPy/SciPy 完成像素与体数据处理。项目负责 Series 发现、选择和文件排序；Gray 显式应用 Modality LUT 或 Rescale Slope/Intercept，并负责极性、窗宽窗位和 spacing-aware 重采样。

```python
from gray.dicom import apply_monochrome, apply_window_level
from gray.dicom import get_spacing, read_series, resample_volume

volume = read_series(ordered_dicom_files)  # array: (z, y, x)
volume = apply_monochrome(volume)
volume = apply_window_level(volume, window_width=400, window_center=40)
volume = resample_volume(volume, target_spacing=(1.0, 1.0, 1.0))
spacing = get_spacing(volume)              # (slice, row, column)
pixels = volume.array
```

mask 和 label 使用 `interpolator="nearest"`。压缩 DICOM 是否可解码取决于 pydicom 环境中安装的像素解码插件。

## 边界

Gray 不规定通用图像 CSV、模型基类、loss、Web UI、推理服务、3D loader 或患者 metadata 策略。只有至少两个独立项目需要相同且稳定的行为时，才应将代码提升到 Gray。

<a id="english"></a>

# English README

Gray is a small, reusable training-research foundation for Gray Medical Computing Laboratory.
It provides configuration identity, reproducibility, artifact paths, lifecycle dispatching and extension contracts. Models, datasets, metrics, augmentation, losses and inference remain owned by each project.

## Documentation

- [Architecture](https://github.com/GrayMedicalComputingLaboratory/Gray/blob/main/docs/ARCHITECTURE.md): responsibilities and ownership of every framework module.
- [API Reference](https://graymedicalcomputinglaboratory.github.io/Gray/api.html): rendered API documentation with parameters, returns, exceptions and source links.
- [API source](https://github.com/GrayMedicalComputingLaboratory/Gray/blob/main/docs/api.html): the static documentation source stored in this repository.

## Installation

```bash
python -m pip install --upgrade \
  "gray @ git+ssh://git@github.com/GrayMedicalComputingLaboratory/Gray.git@main#subdirectory=source/gray"
```

For local development:

```bash
python -m pip install -e /path/to/Gray/source/gray
```

Enable ClearML experiment management:

```bash
python -m pip install -e "/path/to/Gray/source/gray[train]"
```

See [`examples/main.py`](https://github.com/GrayMedicalComputingLaboratory/Gray/blob/main/examples/main.py) for a minimal training run. After
running `clearml-init`, execute it with `python examples/main.py`.

The standard managed entry point is:

```python
from gray import load_config
from gray.experiment import init_task

config = load_config("configs/exp_001.yaml")
task = init_task(config)
result = train(config, task)
```

`train(config, task)` receives the native ClearML `Task`. Logs, curves, and debug samples are reported through a tracker-bound `GrayLogger`; parameters, artifacts, and models continue to use `task.connect()`, `task.upload_artifact()`, and `task.update_output_model()`. ClearML marks normal process exit as completed and uncaught exceptions as failed.

## Project Integration

Each project declares its lifecycle entrypoints. Gray calls them with the resolved configuration dictionary.

```python
from gray.runtime import seed_everything

def train(config: dict, trial=None) -> dict:
    seed_everything(config["runtime"]["seed"])
    # Build the dataset, model, optimizer and training loop in the project.
    return {"experiment_id": config["experiment_id"], "status": "complete"}
```

Run lifecycle stages with:

```bash
python -m gray.cli train --config configs/exp_001.yaml
python -m gray.cli validate --config configs/exp_001.yaml
python -m gray.cli analyze --config configs/exp_001.yaml
```

Hydra accepts one self-contained YAML and does not support `defaults`, configuration groups or automatic working-directory changes. A dictionary returned by a stage is written to `<output_root>/<experiment_id>/<stage>/summary.json`.

## Test-Time Augmentation

`tta` returns named 2D/3D inference variants; prediction aggregation remains project-owned. For 3D TTA, the Z/depth axis is never changed and the same in-plane transform is applied to every slice.

```python
from gray.inference import tta

variants = tta(volume, dim="3d", horizontal_flip=True,
               rotate90_angles=[0, 90, 180, 270])
logits = [model(variant) for variant in variants.values()]
```

## DICOM Processing

The `gray.dicom` module uses pydicom for decoding and NumPy/SciPy for volume processing. The project discovers, selects, and orders the Series. Gray explicitly applies the Modality LUT or Rescale Slope/Intercept, then handles polarity, windowing, and spacing-aware resampling. Arrays use `(z, y, x)` order and spacing uses `(slice, row, column)` order. Use `interpolator="nearest"` for masks and labels. Compressed DICOM transfer syntaxes require a compatible pydicom pixel decoder plugin.

## Boundaries

Gray intentionally does not prescribe a universal image CSV, model base class, loss, Web UI, inference service, 3D loader or patient-metadata policy. Promote code into Gray only after at least two independent projects need the same stable behavior.
