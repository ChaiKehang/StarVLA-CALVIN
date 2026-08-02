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
        self.assertEqual(metrics["exact_top1_accuracy"], 1.0)
        self.assertEqual(metrics["exact_top5_accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy_supported_classes"], 1.0)
        self.assertEqual(metrics["macro_f1_supported_classes"], 1.0)
        self.assertEqual(metrics["mean_bin_manhattan_distance"], 0.0)
        self.assertEqual(metrics["top1_mean_manhattan_distance"], 0.0)
        self.assertEqual(metrics["top1_mean_chebyshev_distance"], 0.0)
        self.assertEqual(metrics["top1_within_manhattan_1_accuracy"], 1.0)
        self.assertEqual(metrics["top1_within_chebyshev_1_accuracy"], 1.0)
        self.assertEqual(metrics["top5_min_manhattan_distance"], 0.0)
        self.assertEqual(metrics["top5_min_chebyshev_distance"], 0.0)
        self.assertEqual(metrics["top5_within_manhattan_1_accuracy"], 1.0)
        self.assertEqual(metrics["top5_within_chebyshev_1_accuracy"], 1.0)
        self.assertEqual(metrics["per_axis_accuracy"], {"x": 1.0, "y": 1.0, "z": 1.0})

    def test_spatial_distances_do_not_use_flat_class_id_distance(self):
        # Target 64=(2,2,4). Top-1 65=(2,3,0) differs by only one class ID
        # but is spatially far: Manhattan=5, Chebyshev=4. Three other Top-5
        # candidates are true radius-1 spatial neighbors.
        targets = np.asarray([64])
        probabilities = np.zeros((1, 125), dtype=np.float64)
        for class_id, probability in zip(
            (65, 63, 39, 89, 60), (0.40, 0.25, 0.15, 0.12, 0.08)
        ):
            probabilities[0, class_id] = probability

        metrics = classification_metrics(probabilities, targets)

        self.assertEqual(metrics["exact_top1_accuracy"], 0.0)
        self.assertEqual(metrics["exact_top5_accuracy"], 0.0)
        self.assertEqual(metrics["top1_mean_manhattan_distance"], 5.0)
        self.assertEqual(metrics["top1_mean_chebyshev_distance"], 4.0)
        self.assertEqual(metrics["top1_within_manhattan_1_accuracy"], 0.0)
        self.assertEqual(metrics["top1_within_chebyshev_1_accuracy"], 0.0)
        self.assertEqual(metrics["top5_min_manhattan_distance"], 1.0)
        self.assertEqual(metrics["top5_min_chebyshev_distance"], 1.0)
        self.assertEqual(metrics["top5_within_manhattan_1_accuracy"], 1.0)
        self.assertEqual(metrics["top5_within_chebyshev_1_accuracy"], 1.0)
        self.assertAlmostEqual(
            metrics["top5_near_fraction_chebyshev_1"], 3.0 / 5.0
        )
        self.assertAlmostEqual(metrics["expected_manhattan_distance"], 2.84)
        self.assertAlmostEqual(
            metrics["probability_mass_within_chebyshev_1"], 0.52
        )


if __name__ == "__main__":
    unittest.main()
