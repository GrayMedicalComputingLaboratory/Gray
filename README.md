# Gray

Gray is the Gray Medical Computing Laboratory computer-vision framework template. It extracts stable engineering patterns from `skin_ai` and `RCM` without changing their model mathematics or moving their code.

It provides a training-research lifecycle: preparation interfaces, training, validation, scoring, analysis, experiment traceability, and extension boundaries for classification, detection, segmentation, and medical CV plugins.

## Layout

```text
source/gray/ reusable package: core, tasks, datasets, trainers, evaluators, metrics, visualization
source/configs/<experiment_id>/<experiment_id>.yaml  portable, identity-bearing configurations
examples/     small task examples
scripts/      setup and smoke-test commands
tests/        unit tests
docs/         future plugin documentation
```

Read [ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md) before extracting code from a project.
Read [Commercial Baseline](docs/COMMERCIAL_BASELINE.md) before adapting it for a customer or regulated workflow.

## Install

```powershell
cd D:\Desktop\Gray\Gray
python -m pip install -e .
python scripts/create_example_images.py
```

## Quick Start

```powershell
gray train --config source/configs/example_classification/example_classification.yaml
gray validate --config source/configs/example_classification/example_classification.yaml
gray analyze --config source/configs/example_classification/example_classification.yaml
```

The example writes model, validation and analysis artifacts below `outputs/example_classification/`. The centroid baseline proves configuration identity, paths, artifacts and evaluation. Replace it in real work; it is not a medical model.

## Add a Project

1. Copy a config and implement a project-local dataset with samples and metadata.
2. Add a model under the project or task plugin. Keep task-specific loss and preprocessing beside it.
3. Reuse `gray.core` configuration and lifecycle contracts.
4. Add an evaluator, OOF scorer when applicable, and tests for the new task.
5. Promote code to Gray only after another independent project needs the same stable behavior.

## Extend Tasks

Detection and segmentation are intentionally not empty placeholder packages. Add them when a concrete implementation needs unifying. The `gray.core.interfaces` contracts are deliberately small so task code retains valid forward, loss, and preprocessing semantics.

## Tests

```powershell
python -m compileall -q source tests
python -m pytest -q
python scripts/smoke_test.py
python scripts/smoke_test.py
```

## Limits

Gray intentionally does not design generic inference, Web, serving, DICOM/NIfTI/WSI IO, nnU-Net, YOLO, DINO, MLOps orchestration, or a universal loss/model base class. A project may extract its simple inference code locally; production deployment follows the deployment standard separately.
