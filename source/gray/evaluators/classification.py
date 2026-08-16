from __future__ import annotations

from pathlib import Path

from gray.core.interfaces import BaseEvaluator
from gray.datasets import ImageCsvDataset
from gray.metrics import classification_metrics
from gray.tasks.classification import CentroidClassifier
from gray.utils.runtime import write_json


class ClassificationEvaluator(BaseEvaluator):
    def __init__(self, manifest: Path, checkpoint: Path, output_dir: Path) -> None:
        self.dataset, self.model, self.output_dir = ImageCsvDataset(manifest), CentroidClassifier.load(checkpoint), output_dir

    def evaluate(self) -> dict[str, float]:
        targets = [str(record.label) for record in self.dataset.records]
        predictions = [str(self.model.predict(self.dataset.image_features(i))["label"]) for i in range(len(self.dataset))]
        result = classification_metrics(targets, predictions)
        write_json(self.output_dir / "metrics.json", result)
        return result
