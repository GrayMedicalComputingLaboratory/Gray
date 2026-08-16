"""Small extension interfaces; task code is free to add domain-specific APIs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def predict(self, sample: Any) -> Any: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...


class BaseTrainer(ABC):
    @abstractmethod
    def fit(self) -> dict[str, Any]: ...


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self) -> dict[str, Any]: ...
