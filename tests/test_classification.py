import unittest
import numpy as np

from gray.metrics import classification_metrics
from gray.tasks.classification import CentroidClassifier


class ClassificationTests(unittest.TestCase):
    def test_metrics(self) -> None:
        values = classification_metrics(["a", "b"], ["a", "b"])
        self.assertEqual(values["accuracy"], 1.0)

    def test_centroid_roundtrip(self) -> None:
        model = CentroidClassifier.fit(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32), ["red", "blue"])
        self.assertEqual(model.predict(np.array([0.9, 0, 0], dtype=np.float32))["label"], "red")
