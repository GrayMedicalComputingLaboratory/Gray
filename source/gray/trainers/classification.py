from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gray.core.config import artifact_dir
from gray.core.interfaces import BaseTrainer
from gray.core.runtime import model_manifest
from gray.datasets import ImageCsvDataset
from gray.tasks.classification import CentroidClassifier
from gray.utils.runtime import write_json


class ClassificationTrainer(BaseTrainer):
    def __init__(self, train_manifest: Path, config: dict[str, Any]) -> None:
        self.dataset, self.config = ImageCsvDataset(train_manifest), config

    def fit(self) -> dict[str, Any]:
        features = np.stack([self.dataset.image_features(i) for i in range(len(self.dataset))])
        labels = [record.label for record in self.dataset.records]
        if any(label is None for label in labels): raise ValueError("labels are required")
        model = CentroidClassifier.fit(features, [str(label) for label in labels])
        output_dir = artifact_dir(self.config, "models", create=True)
        checkpoint = output_dir / f"{self.config['experiment_id']}_baseline.npz"
        model.save(checkpoint)
        result = {"samples": len(labels), "classes": model.labels, "checkpoint": str(checkpoint), "experiment_id": self.config["experiment_id"]}
        write_json(output_dir / "train_summary.json", result)
        write_json(output_dir / "model_manifest.json", model_manifest(self.config, checkpoint))
        return result
