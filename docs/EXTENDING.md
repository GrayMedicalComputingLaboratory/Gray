# Extending GrayCV

Keep an implementation in its project until two independent projects share the same stable behavior.

## Add a classification model

Implement `predict()` and `save()` from `graycv.core.interfaces.BaseModel`, then make the project trainer construct it. Loss functions, augmentations, and preprocessing stay with the project unless they are reused unchanged.

## Add detection or segmentation

Create `graycv/tasks/<task>/` only alongside a working dataset, trainer, evaluator, metric, and visualization implementation. Do not add an empty plugin package in advance.

## Add medical imaging

DICOM/NIfTI/WSI/3D adapters should live in a dedicated plugin. Do not place RCM stack masks, skin lesion masks, or patient-level folds in `graycv.core`.
