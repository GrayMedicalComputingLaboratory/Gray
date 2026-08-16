# Commercial Baseline

Gray is a commercial framework baseline, not a production medical-device release.

## Enforced framework contracts

- Source code is under `source/`; tests are under `tests/`; input data is never written by framework code.
- Config filename defines `experiment_id`. The config directory and filename must agree.
- All generated artifacts use `outputs/<experiment_id>/<stage>/` and stages are created only on write.
- Device values are `cpu`, `cuda`, `cuda:N`, or non-negative GPU indexes. Invalid CUDA requests fail explicitly.
- Each checkpoint has a `model_manifest.json` with model, architecture, data/schema identity and SHA-256.
- Every command records a file log under the experiment `logs` stage.

## Required task-plugin responsibilities

Classification, detection, segmentation and medical plugins must provide their own input validation, loss, augmentation, evaluator, model metadata, batch progress, and testing. A cross-validation task must produce OOF predictions, fold summaries, a confusion matrix and diagnostic threshold analysis; a single-run task must state why OOF does not apply. Inference and serving are intentionally project-owned, not Gray Core.

## Before regulated deployment

Add approved data-version records, access control, audit trail, model card, software bill of materials, vulnerability scanning, protected Git/CI, production health checks, deployment rollback, monitoring, clinical evaluation and human review controls. These controls cannot be claimed from a code template alone.
