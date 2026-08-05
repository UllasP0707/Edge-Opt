from __future__ import annotations

import unittest

import numpy as np

from edge_opt.errors import ConfigurationError
from edge_opt.pruning import (
    MagnitudePruner,
    PolynomialPruningSchedule,
    estimate_sparse_storage,
    measured_sparsity,
)


class PolynomialScheduleTests(unittest.TestCase):
    def test_cubic_schedule_matches_boundaries_and_midpoint(self) -> None:
        schedule = PolynomialPruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=0.75,
            begin_step=100,
            end_step=1_100,
            update_frequency=100,
        )
        self.assertEqual(schedule.sparsity_at(0), 0.0)
        self.assertEqual(schedule.sparsity_at(100), 0.0)
        self.assertAlmostEqual(schedule.sparsity_at(600), 0.75 * (1.0 - 0.5**3))
        self.assertEqual(schedule.sparsity_at(1_100), 0.75)
        self.assertEqual(schedule.sparsity_at(2_000), 0.75)
        self.assertTrue(schedule.is_update_step(600))
        self.assertFalse(schedule.is_update_step(650))

    def test_invalid_schedule_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            PolynomialPruningSchedule(final_sparsity=1.0)
        with self.assertRaises(ConfigurationError):
            PolynomialPruningSchedule(begin_step=10, end_step=10)


class MagnitudePrunerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.final_schedule = PolynomialPruningSchedule(
            final_sparsity=0.5, begin_step=0, end_step=10, update_frequency=1
        )

    def test_global_pruning_hits_exact_target_and_smallest_magnitudes(self) -> None:
        weights = {
            "a": np.asarray([1.0, 2.0, 100.0, 200.0]),
            "b": np.asarray([3.0, 4.0, 5.0, 6.0]),
        }
        pruner = MagnitudePruner(self.final_schedule)
        pruned, result = pruner.step(weights, step=10)
        self.assertEqual(result.pruned_parameters, 4)
        self.assertEqual(result.actual_sparsity, 0.5)
        np.testing.assert_array_equal(pruned["a"], [0.0, 0.0, 100.0, 200.0])
        np.testing.assert_array_equal(pruned["b"], [0.0, 0.0, 5.0, 6.0])
        self.assertEqual(measured_sparsity(pruned), 0.5)
        np.testing.assert_array_equal(weights["a"], [1.0, 2.0, 100.0, 200.0])

    def test_masks_are_monotonic_as_target_increases(self) -> None:
        schedule = PolynomialPruningSchedule(
            final_sparsity=0.75, begin_step=0, end_step=4, update_frequency=1
        )
        weights = {"weight": np.arange(1.0, 9.0)}
        pruner = MagnitudePruner(schedule)
        first = pruner.update_masks(weights, step=1).masks["weight"]
        changed_weights = {"weight": np.arange(8.0, 0.0, -1.0)}
        final = pruner.update_masks(changed_weights, step=4).masks["weight"]
        self.assertTrue(np.all(final[first == 0] == 0))
        self.assertEqual(int(np.count_nonzero(final == 0)), 6)

    def test_local_pruning_applies_target_to_each_tensor(self) -> None:
        weights = {"a": np.arange(1.0, 5.0), "b": np.arange(10.0, 14.0)}
        pruner = MagnitudePruner(self.final_schedule, global_pruning=False)
        _, result = pruner.step(weights, 10)
        self.assertEqual(result.pruned_parameters, 4)
        self.assertEqual(int(np.count_nonzero(result.masks["a"] == 0)), 2)
        self.assertEqual(int(np.count_nonzero(result.masks["b"] == 0)), 2)

    def test_state_round_trip_preserves_masks(self) -> None:
        weights = {"weight": np.arange(1.0, 9.0).reshape(2, 4)}
        first = MagnitudePruner(self.final_schedule)
        first.update_masks(weights, 10)
        restored = MagnitudePruner(self.final_schedule)
        restored.load_state_dict(first.state_dict())
        np.testing.assert_array_equal(restored.masks["weight"], first.masks["weight"])
        self.assertEqual(restored.last_step, 10)

    def test_sparse_storage_accounts_for_index_overhead(self) -> None:
        estimate = estimate_sparse_storage(1_000, 0.75, value_bits=8, index_bits=16)
        self.assertEqual(estimate.dense_bytes, 1_000)
        self.assertEqual(estimate.nonzero_values, 250)
        self.assertEqual(estimate.index_bytes, 500)
        self.assertEqual(estimate.sparse_bytes, 750)
        self.assertAlmostEqual(estimate.compression_ratio, 4 / 3)


if __name__ == "__main__":
    unittest.main()
