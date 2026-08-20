# Gray Architecture

Gray is a shared foundation, not a repository for project-specific algorithms.
Every source file has one owner and one responsibility.

| Location | Responsibility | Must not contain |
| --- | --- | --- |
| `source/gray/cli.py` | Parse a command, load configuration, dispatch one configured project stage, save its returned summary. | Model, dataset, loss, metric formulas, training loops. |
| `source/gray/core/config.py` | One-file Hydra composition, configuration identity and portable path resolution. | Artifact policy, config-group composition or training logic. |
| `source/gray/callbacks/early_stopping.py` | Framework-independent metric-based early stopping state. | Model calls, checkpoint writing or training-loop ownership. |
| `source/gray/inference/ensemble.py` | One-call equal or weighted averaging of classification probabilities. | Model loading, forward passes, logits conversion or threshold policy. |
| `source/gray/inference/tta.py` | Selected in-plane inference-time flip and Rotate90 variants. | Model calls, prediction aggregation or training augmentation. |
| `source/gray/preprocess/remove_background.py` | Crop margins matching one or more background colors from selected NumPy image axes. | Dataset policy, image decoding or clinical intensity normalization. |
| `source/gray/experiment/artifacts.py` | Deterministic local artifact directories by experiment and stage. | File schemas, artifact uploads or training logic. |
| `source/gray/experiment/manifest.py` | Structured Experiment-Dataset-Code-Config-Training-Model-Evaluation lineage manifests. | Artifact persistence, training logic or remote tracking SDK calls. |
| `source/gray/experiment/clearml.py` | ClearML Task lifecycle, configuration linkage and final experiment artifact upload. | Training loops, metric definitions, model promotion or deployment. |
| `source/gray/runtime/device.py` | Explicit CPU/CUDA device validation. | Seeding, model execution or CUDA memory policy. |
| `source/gray/runtime/reproducibility.py` | Process, DataLoader worker and PyTorch generator reproducibility controls. | Data loading policy, logging or artifact IO. |
| `source/gray/utils/logging.py` | Semantic terminal/file logging with an optional remote logger bridge. | Experiment lifecycle, metric aggregation or tracking SDK initialization. |
| `source/gray/utils/io.py` | Small UTF-8 JSON artifact writes. | Project result schemas or data loading. |
| `source/gray/utils/hashing.py` | Streaming file SHA-256 calculation. | Model identity, artifact policy or experiment metadata. |
| `source/gray/dicom/<operation>.py` | SimpleITK-backed pixel and volume processing: read, rescale, window/level, polarity and resampling. | Patient metadata, Series discovery/selection, UID policy or clinical labels. |

## Project Boundary

Each consuming project owns its `datasets`, `models`, `training`, `evaluation`,
`preprocess`, `inference`, and `web` packages. It exposes only explicit stage
functions through its experiment configuration:

```yaml
project:
  entrypoints:
    train: my_project.training:train
    validate: my_project.validation:validate
    analyze: my_project.analysis:analyze
```

Gray calls the configured function with the resolved configuration. A stage
returns a dictionary; Gray writes that dictionary to the corresponding
artifact directory. The project remains responsible for the contents and
clinical meaning of the result.

Public imports follow their domain ownership: experiment lifecycle and artifacts
come from `gray.experiment`, inference transforms from `gray.inference`, and
device/reproducibility controls from `gray.runtime`.

## Promotion Rule

Do not add a helper to Gray because one project happens to need it. Keep it in
the project until a second independent project uses the same behavior without
task-specific conditions. Then add a small documented API, unit tests, and a
clear owner here.
