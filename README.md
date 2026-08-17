# Gray

Gray 是 Gray Medical Computing Laboratory 的轻量、可复用训练研究基础框架。
它提供配置身份、可复现性、指标、产物路径、生命周期调度和扩展契约；模型、数据集、增强、损失和推理仍由具体项目负责。

Gray is a small, reusable training-research foundation for Gray Medical Computing Laboratory.
It provides configuration identity, reproducibility, metrics, artifact paths, lifecycle dispatching and extension contracts.
Models, datasets, augmentation, losses and inference remain owned by each project.

## Documentation / 文档

- [Architecture / 架构说明](docs/ARCHITECTURE.md)：框架模块的职责和所有权。 / Responsibilities and ownership of each framework module.
- [API Reference / API 参考](https://graymedicalcomputinglaboratory.github.io/Gray/api.html)：可渲染的接口文档，包含参数、返回值、异常和源码链接。 / Rendered API documentation with parameters, returns, exceptions and source links.
- [API source / API 源文件](docs/api.html)：仓库中的静态文档源文件。 / Static documentation source stored in this repository.

## Install / 安装

```powershell
python -m pip install --upgrade "git+ssh://git@github.com/GrayMedicalComputingLaboratory/Gray.git@main"
```

本地开发安装： / For active local framework development:

```bash
python -m pip install -e /path/to/gray
```

## Project Integration / 项目集成

每个项目声明自己的生命周期入口；Gray 读取解析后的配置并调用对应函数。
Each project declares its own lifecycle entrypoints; Gray calls them with the resolved configuration dictionary.

```yaml
# configs/exp_001.yaml
# 文件名而不是 YAML 字段定义 experiment_id。 / The filename defines experiment_id.
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
    # Build the dataset, model, optimizer and training loop in the project.
    return {"experiment_id": config["experiment_id"], "status": "complete"}
```

运行生命周期阶段： / Run lifecycle stages:

```bash
python -m gray.cli train --config configs/exp_001.yaml
python -m gray.cli validate --config configs/exp_001.yaml
python -m gray.cli analyze --config configs/exp_001.yaml
```

Hydra 只解析一份自包含 YAML，不支持 `defaults`、配置组或自动切换工作目录。
When a stage returns a dictionary, Gray writes it to `<output_root>/<experiment_id>/<stage>/summary.json`.
Hydra accepts one self-contained YAML and does not change the working directory.

## Test-Time Augmentation / 测试时增强

`tta` 返回命名的 2D/3D 推理变体；概率或 logits 如何聚合仍由项目决定。
`tta` returns named 2D/3D inference variants; prediction aggregation remains project-owned.

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

3D TTA 不反转或旋转 Z/depth 轴，只对每个 slice 的 H/W 平面执行相同变换。
For 3D TTA, the Z/depth axis is never changed; the same in-plane transform is applied to every slice.

## Optuna Search / Optuna 搜索

在同一实验 YAML 中添加 `optuna` 配置。`enabled: true` 时 `gray train` 自动搜索，`gray tune` 是显式别名。
Add an `optuna` section to the same experiment YAML. `gray train` searches when enabled; `gray tune` is the explicit alias.

```yaml
optuna:
  enabled: true
  study_name: exp_001_search
  direction: maximize
  objective_key: valid.f1_macro
  n_trials: 30
  seed: 42
  sampler: tpe
  pruner: median
  final_train: true
  search_space:
    train.lr:
      type: float
      low: 0.000001
      high: 0.0003
      log: true
```

训练入口必须返回 dotted objective key 对应的字典值；Gray 保存 SQLite study、trial 快照、最佳参数和 summary。
The train entrypoint must return the dotted objective value; Gray saves the SQLite study, trial snapshots, best parameters and summary.

## DICOM Processing / DICOM 处理

可选的 `gray.dicom` 模块只使用 SimpleITK。项目负责 Series 选择和文件排序，Gray 负责像素强度与空间处理。
The optional `gray.dicom` module uses SimpleITK only. The project selects and orders the Series; Gray handles pixel and spatial processing.

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
Use `interpolator="nearest"` for masks and labels; do not apply rescaling twice.

## Metrics / 指标

每项指标都是独立模块；组合函数只负责生成统一报告。
Each metric is an independent module; the combined function only assembles a unified report.

- Accuracy / balanced accuracy：准确率与均衡准确率
- Precision / recall / F1：支持 macro 和 weighted
- Confusion matrix / classification report：混淆矩阵与分类报告
- Specificity / sensitivity / PPV / NPV：临床二分类指标
- ROC-AUC / PR-AUC / log loss / Brier score：概率与排序指标

## Clinical Binary Assessment / 临床二分类评估

`clinical_binary_metrics` 提供 Sensitivity、Specificity、PPV、NPV、ROC-AUC、PR-AUC、校准曲线、ECE、Brier、bootstrap 置信区间和阈值报告。
`clinical_binary_metrics` provides sensitivity, specificity, PPV, NPV, ROC-AUC, PR-AUC, calibration data, ECE, Brier score, bootstrap confidence intervals and threshold analysis.

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
Clinical reports require both observed classes; use general classification metrics for a single-class fold.

## Boundaries / 边界

Gray 不规定通用图像 CSV、模型基类、loss、Web UI、推理服务、3D loader 或患者 metadata 策略。
Gray intentionally does not prescribe a universal image CSV, model base class, loss, Web UI, inference service, 3D loader or patient-metadata policy.

只有至少两个独立项目需要相同且稳定的行为时，才应将代码提升到 Gray。
Promote code into Gray only after at least two independent projects need the same stable behavior.
