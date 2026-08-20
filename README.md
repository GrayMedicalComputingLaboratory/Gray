<p align="center">
  <img src="docs/assets/gray.png" alt="Gray Medical Computing Laboratory" width="360">
</p>

<p align="center">
  <a href="#中文说明">中文 README</a> ·
  <a href="#english">English README</a>
</p>

<a id="中文说明"></a>

# Gray 中文说明

Gray 是 Gray Medical Computing Laboratory 的轻量、可复用训练研究基础框架。
它提供配置身份、可复现性、指标、产物路径、生命周期调度和扩展契约；模型、数据集、增强、损失和推理仍由具体项目负责。

## 文档

- [架构说明](docs/ARCHITECTURE.md)：框架模块的职责和所有权。
- [API 参考](https://graymedicalcomputinglaboratory.github.io/Gray/api.html)：可渲染的接口文档，包含参数、返回值、异常和源码链接。
- [API 源文件](docs/api.html)：仓库中的静态文档源文件。

## 安装

```bash
python -m pip install --upgrade "git+ssh://git@github.com/GrayMedicalComputingLaboratory/Gray.git@main"
```

本地开发安装：

```bash
python -m pip install -e /path/to/gray
```

启用 ClearML 实验管理：

```bash
python -m pip install "gray[clearml]"
```

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
from gray.utils import seed_everything

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
from gray.utils import tta

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

可选的 `gray.dicom` 模块只使用 SimpleITK。项目负责 Series 选择和文件排序，Gray 负责像素强度与空间处理。

```python
from gray.dicom import apply_monochrome, apply_rescale, apply_window_level
from gray.dicom import get_spacing, read_series, resample_volume

image = read_series(ordered_dicom_files)
image = apply_rescale(image)
image = apply_monochrome(image)
image = apply_window_level(image, window_width=400, window_center=40)
image = resample_volume(image, target_spacing=(1.0, 1.0, 1.0))
spacing = get_spacing(image)
```

mask 和 label 使用 `interpolator="nearest"`；已完成物理值转换时不要重复调用 `apply_rescale`。

## 指标与临床评估

支持 Accuracy、balanced accuracy、Precision、Recall、F1、混淆矩阵、Specificity、Sensitivity、PPV、NPV、ROC-AUC、PR-AUC、log loss、Brier score、校准曲线、bootstrap 置信区间和阈值报告。

```python
from gray.metrics import clinical_binary_metrics

report = clinical_binary_metrics(
    targets=["Non-BCC", "BCC", "BCC"],
    predictions=["Non-BCC", "BCC", "Non-BCC"],
    probabilities=[0.08, 0.91, 0.42],
    positive_label="BCC",
    n_bootstrap=2_000,
)
```

临床报告要求观测到两个类别；单类别 fold 应使用通用 classification metrics。

## 绘图

`gray.plot` 提供混淆矩阵、指标与置信区间、ROC-AUC、PR 曲线、校准曲线和阈值分析。所有绘图函数返回 Matplotlib `Figure`，可通过 `save_path` 保存。

```python
from gray.plot import plot_confusion_matrix, plot_metrics, plot_roc_auc

plot_confusion_matrix(targets, predictions, labels=["Non-BCC", "BCC"],
                      normalize=True, save_path="figures/confusion_matrix.png")
plot_roc_auc(targets, probabilities, positive_label="BCC", ci=True,
             n_bootstrap=2_000, save_path="figures/roc_auc.png")
plot_metrics({"f1_macro": 0.82, "roc_auc": 0.89},
             ci={"roc_auc": {"lower": 0.81, "upper": 0.95}},
             save_path="figures/metrics.png")
```

## 边界

Gray 不规定通用图像 CSV、模型基类、loss、Web UI、推理服务、3D loader 或患者 metadata 策略。只有至少两个独立项目需要相同且稳定的行为时，才应将代码提升到 Gray。

<a id="english"></a>

# English README

Gray is a small, reusable training-research foundation for Gray Medical Computing Laboratory.
It provides configuration identity, reproducibility, metrics, artifact paths, lifecycle dispatching and extension contracts. Models, datasets, augmentation, losses and inference remain owned by each project.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): responsibilities and ownership of every framework module.
- [API Reference](https://graymedicalcomputinglaboratory.github.io/Gray/api.html): rendered API documentation with parameters, returns, exceptions and source links.
- [API source](docs/api.html): the static documentation source stored in this repository.

## Installation

```bash
python -m pip install --upgrade "git+ssh://git@github.com/GrayMedicalComputingLaboratory/Gray.git@main"
```

For local development:

```bash
python -m pip install -e /path/to/gray
```

Enable ClearML experiment management:

```bash
python -m pip install "gray[clearml]"
```

## Project Integration

Each project declares its lifecycle entrypoints. Gray calls them with the resolved configuration dictionary.

```python
from gray.utils import seed_everything

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
from gray.utils import tta

variants = tta(volume, dim="3d", horizontal_flip=True,
               rotate90_angles=[0, 90, 180, 270])
logits = [model(variant) for variant in variants.values()]
```

## DICOM Processing

The optional `gray.dicom` module uses SimpleITK only. The project selects and orders the Series; Gray handles pixel intensity and spatial processing. Use `interpolator="nearest"` for masks and labels, and do not apply rescaling twice.

## Metrics and Clinical Assessment

Gray supports accuracy, balanced accuracy, precision, recall, F1, confusion matrices, specificity, sensitivity, PPV, NPV, ROC-AUC, PR-AUC, log loss, Brier score, calibration curves, bootstrap confidence intervals and threshold reports.

`clinical_binary_metrics` requires both observed classes. Use the general classification metrics for a single-class fold diagnostic.

## Plotting

`gray.plot` provides confusion matrices, metric bars with confidence intervals, ROC-AUC, precision-recall, calibration and threshold plots. Every plotting function returns a Matplotlib `Figure` and can save through `save_path`.

## Boundaries

Gray intentionally does not prescribe a universal image CSV, model base class, loss, Web UI, inference service, 3D loader or patient-metadata policy. Promote code into Gray only after at least two independent projects need the same stable behavior.
