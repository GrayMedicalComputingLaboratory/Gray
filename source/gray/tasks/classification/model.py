"""A deterministic baseline classifier used only to prove the framework lifecycle."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gray.core.interfaces import BaseModel


class CentroidClassifier(BaseModel):
    def __init__(self, labels: list[str], centroids: np.ndarray) -> None:
        self.labels, self.centroids = labels, np.asarray(centroids, dtype=np.float32)

    @classmethod
    def fit(cls, features: np.ndarray, labels: list[str]) -> "CentroidClassifier":
        names = sorted(set(labels))
        return cls(names, np.stack([features[np.array(labels) == name].mean(axis=0) for name in names]))

    def predict(self, sample: np.ndarray) -> dict[str, object]:
        distances = ((self.centroids - sample) ** 2).sum(axis=1)
        scores = np.exp(-(distances - distances.min()))
        probabilities = scores / scores.sum()
        idx = int(probabilities.argmax())
        return {"label": self.labels[idx], "confidence": float(probabilities[idx]), "probabilities": dict(zip(self.labels, map(float, probabilities)))}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, labels=np.asarray(self.labels), centroids=self.centroids)

    @classmethod
    def load(cls, path: Path) -> "CentroidClassifier":
        data = np.load(path, allow_pickle=False)
        return cls(data["labels"].tolist(), data["centroids"])
