# Gray Architecture

Gray is a shared foundation, not a repository for project-specific algorithms.
Every source file has one owner and one responsibility.

| Location | Responsibility | Must not contain |
| --- | --- | --- |
| `source/gray/cli.py` | Parse a command, load configuration, dispatch one configured project stage, save its returned summary. | Model, dataset, loss, metric formulas, training loops. |
| `source/gray/core/config.py` | Configuration identity, validation and deterministic artifact paths. | Training logic or model metadata. |
| `source/gray/core/device.py` | Explicit CPU/CUDA device validation. | Seeding, model execution or CUDA memory policy. |
| `source/gray/core/interfaces.py` | Minimal stable contracts for project code. | Task-specific implementations. |
| `source/gray/core/provenance.py` | Checkpoint checksum and model identity manifest. | Artifact writing policy or training logic. |
| `source/gray/metrics/` | Stateless, reusable metric calculations. | File reads, model calls, threshold selection policy. |
| `source/gray/utils/reproducibility.py` | Python, NumPy and PyTorch random-state control. | Logging or artifact serialization. |
| `source/gray/utils/logging.py` | Console/file logger construction. | Metric reporting semantics. |
| `source/gray/utils/io.py` | Small UTF-8 JSON artifact writes. | Project result schemas or data loading. |

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

## Promotion Rule

Do not add a helper to Gray because one project happens to need it. Keep it in
the project until a second independent project uses the same behavior without
task-specific conditions. Then add a small documented API, unit tests, and a
clear owner here.
