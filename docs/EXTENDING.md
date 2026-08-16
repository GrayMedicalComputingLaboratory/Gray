# Extending Gray

Keep an implementation in its project until two independent projects share the same stable behavior.

## Add project-specific code

Implement models, datasets, trainers and evaluators inside the project package. Loss functions, augmentations, preprocessing and data schemas stay with the project unless they are reused unchanged by two independent projects.

## Add detection or segmentation

Do not create empty framework task packages. Promote a tested cross-project abstraction to Gray only after two independent projects share the same stable contract.

## Add medical imaging

DICOM/NIfTI/WSI/3D adapters should live in a dedicated plugin. Do not place task-specific masks, patient-level split policies, or domain preprocessing rules in `gray.core`.
