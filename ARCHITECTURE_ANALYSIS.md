# GrayCV Architecture Analysis

## Sources Reviewed

| Area | `skin_ai` | `RCM` | GrayCV decision |
|---|---|---|---|
| Configuration | YAML deployment/task profiles | YAML experiment config and path helpers | Core config loader/path policy |
| Logging and artifacts | Logger, checkpoints, runtime output scopes | Logger, config snapshots, experiment paths | Core utility interfaces only |
| Dataset/dataloader | Image CSV, transforms, workers | NPY stacks, slice masks, sampling | Generic dataset contract; project datasets stay local |
| Training | Classification, segmentation, detection engines | DINO/MIL training loop | Base trainer contract; concrete loops remain task/project-side |
| Metrics/scoring | Classification/segmentation metrics, OOF score modules | Classification metrics, OOF aggregation | Metrics package; baseline classification metric now |
| Inference/web | Predictor SDK, ONNX, report, FastAPI UI | Stack service and attention visualizer | Explicitly project-owned; not designed in GrayCV |
| Models | timm, nnU-Net and YOLO integrations | DINO + MIL / 3D stack model | No model copied into core |

## What Is Common

Both projects separate offline training from inference, use YAML configuration, persist artifacts, contain a dataset/dataloader layer, calculate task metrics, and launch workflows through scripts or CLIs. These lifecycle concerns define GrayCV Core.

## What Must Remain Project-Specific

`skin_ai`: nnU-Net conventions, masks, detection, ONNX export, clinical measurements/reports, FastAPI workstation, and model ensembles.

`RCM`: NPY stack preprocessing, depth masks, DINO feature cache, MIL/BiLSTM attention, fold/OOF policy, and stack-attention visualization.

Neither project is changed or copied. These are candidates for future plugins only when another project needs the same stable contract.

## GrayCV v0.1 Architecture

```text
config -> dataset -> task model -> trainer -> checkpoint
                                  |              |
                                  v              v
                             evaluator <- inference -> visualization/report
```

`graycv.core` contains small contracts and configuration. `tasks/classification` contains a runnable training baseline proving the lifecycle. Future detection, segmentation, 3D, DICOM, WSI, or medical packages should appear only with a concrete implementation to move. Inference, Web, and serving stay in the concrete project.

## Migration Map

| GrayCV module | Design source | Implementation |
|---|---|---|
| `core.config` | RCM config identity and Skin AI YAML profiles | Rewritten, dependency-light |
| `utils.runtime` | Both projects' seed/logger/artifact utilities | Rewritten, minimal |
| `datasets.image_csv` | Skin AI classification dataset | New generic baseline, not copied |
| trainer/evaluator/metrics | Both projects' lifecycle separation | New baseline contracts |
| `visualization` | Skin AI artifacts and RCM attention-output boundary | New task-owned interface |

## Deliberately Not Abstracted

No task `if` dispatcher, generic medical preprocessor, unified loss class, forced model signature, or MLOps platform. GrayCV is an extension point, not a replacement for either existing project.
