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
python -m pip install -e D:\Desktop\Frameworks\Gray
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
from gray.utils.runtime import seed_everything

def train(config: dict) -> dict:
    seed_everything(config["runtime"]["seed"])
    # Build this project's dataset, model, optimizer and training loop.
    return {"experiment_id": config["experiment_id"], "status": "complete"}
```

For PyTorch DataLoaders, pass `gray.utils.runtime.seed_worker` as
`worker_init_fn` and `torch_generator(seed)` as `generator`.

Run each lifecycle stage through Gray:

```powershell
python -m gray.cli train --config configs\exp_001.yaml
python -m gray.cli validate --config configs\exp_001.yaml
python -m gray.cli analyze --config configs\exp_001.yaml
```

If your Python Scripts directory is on `PATH`, `gray train ...` is equivalent.
When a stage function returns a dictionary, Gray writes it to
`<output_root>/<experiment_id>/<stage>/summary.json`.

## Metrics

`gray.metrics.classification_metrics` supports binary and multiclass reports:

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

## Boundaries

Gray intentionally does not prescribe a universal image CSV schema, generic
inference service, Web UI, 3D medical loader, loss function, model base class,
or augmentation pipeline. Promote code into Gray only after two independent
projects need the same stable behavior.
