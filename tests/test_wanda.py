from __future__ import annotations

import unittest

import numpy as np

from edge_opt.activation import ChannelStatistics
from edge_opt.errors import ConfigurationError
from edge_opt.wanda import WandaPruner, wanda_mask, wanda_scores


class WandaMetricTests(unittest.TestCase):
    def test_metric_is_weight_magnitude_times_input_l2_norm(self) -> None:
        weights = np.asarray([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        scores = wanda_scores(weights, np.asarray([10.0, 2.0, 0.5]))
        np.testing.assert_array_equal(scores, [[10.0, 4.0, 1.5], [40.0, 10.0, 3.0]])

    def test_activation_salience_changes_selection_from_magnitude(self) -> None:
        weights = np.asarray([[1.0, 2.0, 3.0, 4.0]])
        mask = wanda_mask(weights, np.asarray([100.0, 1.0, 1.0, 1.0]), 0.5)
        # Magnitude pruning would remove column 0; Wanda retains its salient activation channel.
        np.testing.assert_array_equal(mask, [[True, False, False, True]])

    def test_equal_scores_are_pruned_by_stable_column_order(self) -> None:
        mask = wanda_mask(np.ones((2, 4)), np.ones(4), 0.5)
        np.testing.assert_array_equal(
            mask,
            [[False, False, True, True], [False, False, True, True]],
        )

    def test_metric_validates_linear_shape_and_channel_coverage(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "2-D linear"):
            wanda_scores(np.ones((2, 3, 4)), np.ones(4))
        with self.assertRaisesRegex(ConfigurationError, "activation channels"):
            wanda_scores(np.ones((2, 4)), np.ones(3))


class WandaPrunerTests(unittest.TestCase):
    def test_prunes_each_output_row_and_preserves_source_by_default(self) -> None:
        weights = {
            "encoder": np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        }
        stats = {
            "encoder": ChannelStatistics(
                l2_norm=np.asarray([8.0, 1.0, 1.0, 8.0]),
                absmax=np.asarray([4.0, 1.0, 1.0, 4.0]),
                values_per_channel=4,
                batches=1,
                channel_axis=-1,
            )
        }
        pruned, result = WandaPruner(0.5).prune(weights, stats)
        self.assertEqual(result.pruned_parameters, 4)
        self.assertEqual(result.actual_sparsity, 0.5)
        self.assertEqual(result.layer_sparsity["encoder"], 0.5)
        np.testing.assert_array_equal(pruned["encoder"][0], [1.0, 0.0, 0.0, 4.0])
        np.testing.assert_array_equal(weights["encoder"][0], [1.0, 2.0, 3.0, 4.0])

    def test_fractional_target_reports_representable_actual_sparsity(self) -> None:
        weights = {"layer": np.ones((2, 3))}
        _, result = WandaPruner(0.5).prune(weights, {"layer": np.ones(3)})
        self.assertEqual(result.target_sparsity, 0.5)
        self.assertAlmostEqual(result.actual_sparsity, 1 / 3)

    def test_missing_statistics_fail_closed(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "missing Wanda"):
            WandaPruner().compute_masks({"layer": np.ones((2, 2))}, {})


if __name__ == "__main__":
    unittest.main()
