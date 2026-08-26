"""Parser for native nnU-Net epoch metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NnUNetLogReporter:
    """Convert nnU-Net text output into GrayLogger metric calls."""

    logger: Any
    class_names: list[str] = field(default_factory=list)
    epoch: int | None = None
    values: dict[str, float] = field(default_factory=dict)
    pseudo_dice: list[float] = field(default_factory=list)

    def consume(self, line: str) -> None:
        """Consume one output line and flush at the end of each epoch."""
        stripped = line.strip()
        epoch_match = re.search(
            r"(?:^|:\s*)Epoch\s+(\d+)\s*$", stripped, flags=re.IGNORECASE
        )
        if epoch_match:
            self._flush()
            self.epoch = int(epoch_match.group(1))
            return
        if self.epoch is None:
            return
        patterns = {
            "learning_rate": r"Current learning rate:\s*([-+0-9.eE]+)",
            "train_loss": r"train_loss\s+([-+0-9.eE]+)",
            "val_loss": r"val_loss\s+([-+0-9.eE]+)",
            "epoch_time": r"Epoch time:\s*([-+0-9.eE]+)(?:\s*s)?",
        }
        for name, pattern in patterns.items():
            match = re.search(pattern, stripped, flags=re.IGNORECASE)
            if match:
                self.values[name] = float(match.group(1))
                if name == "epoch_time":
                    self._flush()
                return
        dice_match = re.search(
            r"Pseudo dice\s*\[([^]]*)\]", stripped, flags=re.IGNORECASE
        )
        if dice_match:
            self.pseudo_dice = _parse_numeric_values(dice_match.group(1))

    def close(self) -> None:
        """Flush metrics buffered for the final epoch."""
        self._flush()

    def _flush(self) -> None:
        if self.epoch is None:
            return
        mapping = {
            "learning_rate": ("Learning Rate", "learning_rate"),
            "train_loss": ("Loss", "train"),
            "val_loss": ("Loss", "validation"),
            "epoch_time": ("Epoch Time", "seconds"),
        }
        for name, value in self.values.items():
            title, series = mapping[name]
            self.logger.metric(series, value, iteration=self.epoch, title=title)
        for index, value in enumerate(self.pseudo_dice):
            series = (
                self.class_names[index]
                if index < len(self.class_names)
                else f"class_{index + 1}"
            )
            self.logger.metric(series, value, iteration=self.epoch, title="Pseudo Dice")
        self.epoch = None
        self.values.clear()
        self.pseudo_dice.clear()


def _parse_numeric_values(value: str) -> list[float]:
    """Parse plain and NumPy scalar representations from nnU-Net output."""
    normalized = re.sub(r"(?:np\.)?(?:float|double)\d*\s*\(", "(", value)
    values: list[float] = []
    for token in normalized.split(","):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", token)
        if match:
            values.append(float(match.group(0)))
    return values
