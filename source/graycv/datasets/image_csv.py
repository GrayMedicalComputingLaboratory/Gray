"""A portable CSV image dataset for classification examples and small projects."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    image_path: Path
    label: str | None


class ImageCsvDataset:
    """Read ``sample_id,image_path,label`` records without task-specific assumptions."""
    def __init__(self, manifest: Path, require_label: bool = True) -> None:
        self.manifest = manifest.resolve()
        with self.manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"sample_id", "image_path"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError("manifest requires sample_id,image_path columns")
        if require_label and "label" not in rows[0]:
            raise ValueError("training/validation manifest requires label column")
        self.records = [ImageRecord(str(row["sample_id"]), (self.manifest.parent / row["image_path"]).resolve(), row.get("label") or None) for row in rows]

    def __len__(self) -> int: return len(self.records)

    def image_features(self, index: int) -> np.ndarray:
        with Image.open(self.records[index].image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return pixels.reshape(-1, 3).mean(axis=0)
