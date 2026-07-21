"""CPU tests for offline Intent evaluation metrics."""

import unittest

import numpy as np

from examples.calvin.eval_files.eval_intent_checkpoint import (
    classification_metrics,
    decode_classes,
)


class IntentEvaluationTest(unittest.TestCase):
    def test_joint_class_decode(self):
        np.testing.assert_array_equal(
            decode_classes(np.asarray([0, 62, 124])),
            np.asarray([[0, 0, 0], [2, 2, 2], [4, 4, 4]]),
        )

    def test_perfect_predictions(self):
        targets = np.asarray([0, 1, 62, 124])
        probabilities = np.full((4, 125), 1e-6, dtype=np.float64)
        probabilities[np.arange(4), targets] = 1.0
        probabilities /= probabilities.sum(axis=-1, keepdims=True)

        metrics = classification_metrics(probabilities, targets)

        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["top5_accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy_supported_classes"], 1.0)
        self.assertEqual(metrics["macro_f1_supported_classes"], 1.0)
        self.assertEqual(metrics["mean_bin_manhattan_distance"], 0.0)
        self.assertEqual(metrics["per_axis_accuracy"], {"x": 1.0, "y": 1.0, "z": 1.0})


if __name__ == "__main__":
    unittest.main()
