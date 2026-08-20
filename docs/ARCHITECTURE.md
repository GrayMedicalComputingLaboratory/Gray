# Gray Architecture

Gray is a shared foundation, not a repository for project-specific algorithms.
Every source file has one owner and one responsibility.

| Location | Responsibility | Must not contain |
| --- | --- | --- |
| `source/gray/cli.py` | Parse a command, load configuration, dispatch one configured project stage, save its returned summary. | Model, dataset, loss, metric formulas, training loops. |
| `source/gray/core/config.py` | One-file Hydra composition, configuration identity, validation and deterministic artifact paths. | Config-group composition, training logic or model metadata. |
| `source/gray/core/device.py` | Explicit CPU/CUDA device validation. | Seeding, model execution or CUDA memory policy. |
| `source/gray/core/provenance.py` | Checkpoint checksum and model identity manifest. | Artifact writing policy or training logic. |
| `source/gray/callbacks/early_stopping.py` | Framework-independent metric-based early stopping state. | Model calls, checkpoint writing or training-loop ownership. |
| `source/gray/inference/ensemble.py` | One-call equal or weighted averaging of classification probabilities. | Model loading, forward passes, logits conversion or threshold policy. |
| `source/gray/preprocess/remove_background.py` | Crop margins matching one or more background colors from selected NumPy image axes. | Dataset policy, image decoding or clinical intensity normalization. |
| `source/gray/utils/seed_everything.py` | Process-wide Python, NumPy and PyTorch random-state control. | DataLoader worker initialization, logging or IO. |
| `source/gray/utils/seed_worker.py` | One PyTorch DataLoader worker's random-state control. | Global process seeding or IO. |
| `source/gray/utils/torch_generator.py` | Seeded PyTorch `Generator` construction. | DataLoader configuration or global seeding. |
| `source/gray/utils/tta.py` | Selected in-plane inference-time flip and Rotate90 variants. | Model calls, prediction aggregation or training augmentation. |
| `source/gray/utils/logging.py` | Semantic console/file logging, including numeric metric log records. | Experiment tracking, metric aggregation or remote reporting. |
| `source/gray/utils/io.py` | Small UTF-8 JSON artifact writes. | Project result schemas or data loading. |
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

The package-level `gray.utils` imports are convenience aliases only. The
implementation remains one capability per module, so both
`from gray.utils import seed_everything` and
`from gray.utils.seed_everything import seed_everything` are valid.

## Promotion Rule

Do not add a helper to Gray because one project happens to need it. Keep it in
the project until a second independent project uses the same behavior without
task-specific conditions. Then add a small documented API, unit tests, and a
clear owner here.
