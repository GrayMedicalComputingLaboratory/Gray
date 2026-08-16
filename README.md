# Gray

Gray is a small, reusable training-research foundation for Gray Medical
Computing Laboratory computer-vision and medical-AI projects. It provides
configuration identity, reproducibility, metrics, artifact paths, stage
dispatching, and extension contracts. Models, datasets, augmentation, loss,
and inference remain owned by the project that needs them.

## Install

```powershell
python -m pip install --upgrade "git+ssh://git@github.com/GrayMedicalComputingLaboratory/Gray.git@main"
```

For active local framework work:

```powershell
python -m pip install -e /path/to/gray
```

## Project Integration

Each project declares its own stage entrypoints. Gray imports and calls the
configured function with the resolved configuration dictionary.

```yaml
# configs/exp_001.yaml
experiment_id: exp_001
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
# my_project/training.py
from gray.utils import seed_everything

def train(config: dict) -> dict:
    seed_everything(config["runtime"]["seed"])
    # Build this project's dataset, model, optimizer and training loop.
    return {"experiment_id": config["experiment_id"], "status": "complete"}
```

For PyTorch DataLoaders, pass `gray.utils.seed_worker.seed_worker` as
`worker_init_fn` and `torch_generator(seed)` as `generator`.

## Test-Time Augmentation

`tta` returns selected, named inference variants. Prediction aggregation stays
project-owned so the framework does not impose a logit or probability policy.

```python
from gray.utils import tta

# image: [C, H, W], volume: [C, D, H, W] or [B, C, D, H, W]
variants = tta(
    volume,
    dim="3d",
    horizontal_flip=True,
    vertical_flip=False,
    rotate90_angles=[0, 90, 180, 270],
)
logits = [model(variant) for variant in variants.values()]
```

For `dim="3d"`, TTA never reverses or rotates the depth/Z axis. It applies the
same transform to the final `H/W` plane of every slice.

Run each lifecycle stage through Gray:

```powershell
python -m gray.cli train --config configs/exp_001.yaml
python -m gray.cli validate --config configs/exp_001.yaml
python -m gray.cli analyze --config configs/exp_001.yaml
```

If your Python Scripts directory is on `PATH`, `gray train ...` is equivalent.
When a stage function returns a dictionary, Gray writes it to
`<output_root>/<experiment_id>/<stage>/summary.json`.

## Metrics

Each metric is an independent module, for example
`gray.metrics.roc_auc.roc_auc` and `gray.metrics.f1.f1`. The optional
`gray.metrics.classification_metrics` function only combines those calls into
one report. Binary and multiclass reports support:

- Accuracy and balanced accuracy
- Precision, recall and F1 (macro and weighted)
- Confusion matrix and per-class report
- Specificity (macro and per class)
- ROC-AUC, PR-AUC / average precision, log loss and Brier score when scores are supplied

```python
from gray.metrics import classification_metrics

metrics = classification_metrics(
    targets=["Non-BCC", "BCC", "BCC"],
    predictions=["Non-BCC", "BCC", "Non-BCC"],
    scores=[0.10, 0.91, 0.42],
    labels=["Non-BCC", "BCC"],
)
```

## Clinical Binary Assessment

Use `clinical_binary_metrics` for a two-class clinical report. It provides
Sensitivity, Specificity, PPV, NPV, ROC-AUC, PR-AUC, reliability-curve data,
ECE, Brier score, percentile bootstrap confidence intervals, and a threshold
report with Youden-J and F1 operating points.

```python
from gray.metrics import clinical_binary_metrics

report = clinical_binary_metrics(
    targets=["Non-BCC", "BCC", "BCC"],
    predictions=["Non-BCC", "BCC", "Non-BCC"],
    probabilities=[0.08, 0.91, 0.42],  # P(BCC)
    positive_label="BCC",
    n_bootstrap=2_000,
)
```

`calibration["points"]` is plotting-ready curve data. Clinical reports require
both observed classes; use the general classification metrics for a single-class
fold diagnostic.

## Boundaries

Gray intentionally does not prescribe a universal image CSV schema, generic
inference service, Web UI, 3D medical loader, loss function, model base class,
or augmentation pipeline. Promote code into Gray only after two independent
projects need the same stable behavior.

See [Architecture](docs/ARCHITECTURE.md) for the responsibility and ownership
of every framework module.
