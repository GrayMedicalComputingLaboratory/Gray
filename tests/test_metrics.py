from __future__ import annotations

import unittest

from gray.metrics import classification_metrics


class ClassificationMetricsTest(unittest.TestCase):
    def test_binary_metrics_with_scores(self) -> None:
        result = classification_metrics(
            targets=["negative", "positive", "positive", "negative"],
            predictions=["negative", "positive", "negative", "negative"],
            scores=[0.05, 0.95, 0.40, 0.10],
            labels=["negative", "positive"],
        )
        self.assertEqual(result["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertIsNotNone(result["roc_auc"])
        self.assertIsNotNone(result["pr_auc"])

    def test_single_class_scores_return_none_for_auc(self) -> None:
        result = classification_metrics(
            targets=["positive", "positive"],
            predictions=["positive", "positive"],
            scores=[0.8, 0.9],
            labels=["negative", "positive"],
        )
        self.assertIsNone(result["roc_auc"])


if __name__ == "__main__":
    unittest.main()
