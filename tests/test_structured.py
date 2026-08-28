from __future__ import annotations

import unittest

import numpy as np

from edge_opt.errors import ConfigurationError
from edge_opt.structured import NMPruner, NMPruningPattern, nm_mask, validate_nm_mask


class NMPruningPatternTests(unittest.TestCase):
    def test_two_of_four_properties_and_serialization(self) -> None:
        pattern = NMPruningPattern(2, 4)
        self.assertEqual(pattern.label, "2:4")
        self.assertEqual(pattern.sparsity, 0.5)
        self.assertEqual(pattern.minimum_metadata_bits_per_group, 3)
        self.assertEqual(NMPruningPattern.from_dict(pattern.to_dict()), pattern)

    def test_invalid_pattern_is_rejected(self) -> None:
        for n, m in ((0, 4), (4, 4), (5, 4), (1, 1)):
            with self.subTest(n=n, m=m), self.assertRaises(ConfigurationError):
                NMPruningPattern(n, m)


class NMMaskTests(unittest.TestCase):
    def test_two_of_four_retains_two_largest_in_every_group(self) -> None:
        scores = np.asarray([[1.0, 8.0, 3.0, 7.0, 9.0, 2.0, 6.0, 4.0]])
        mask = nm_mask(scores, NMPruningPattern(2, 4))
        np.testing.assert_array_equal(
            mask,
            [[False, True, False, True, True, False, True, False]],
        )
        self.assertTrue(validate_nm_mask(mask, NMPruningPattern(2, 4)))

    def test_grouping_can_target_a_nonfinal_axis(self) -> None:
        scores = np.arange(16.0).reshape(1, 4, 2, 2)
        pattern = NMPruningPattern(2, 4, axis=1)
        mask = nm_mask(scores, pattern)
        self.assertTrue(validate_nm_mask(mask, pattern))
        np.testing.assert_array_equal(np.count_nonzero(mask, axis=1), np.full((1, 2, 2), 2))

    def test_nondivisible_axis_fails_instead_of_padding_silently(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "not divisible"):
            nm_mask(np.ones((2, 6)), NMPruningPattern(2, 4))

    def test_equal_scores_have_deterministic_mask(self) -> None:
        mask = nm_mask(np.ones((1, 4)), NMPruningPattern(2, 4))
        np.testing.assert_array_equal(mask, [[False, False, True, True]])


class NMPrunerTests(unittest.TestCase):
    def test_external_scores_control_selection_and_source_is_preserved(self) -> None:
        weights = {"layer": np.arange(1.0, 9.0).reshape(1, 8)}
        scores = {"layer": np.asarray([[100.0, 1.0, 2.0, 3.0, 1.0, 100.0, 3.0, 2.0]])}
        pruned, result = NMPruner(NMPruningPattern(2, 4)).prune(
            weights, scores=scores
        )
        self.assertEqual(result.actual_sparsity, 0.5)
        self.assertEqual(result.pruned_parameters, 4)
        self.assertTrue(validate_nm_mask(result.masks["layer"], result.pattern))
        self.assertEqual(float(pruned["layer"][0, 0]), 1.0)
        self.assertEqual(float(weights["layer"][0, 1]), 2.0)


if __name__ == "__main__":
    unittest.main()
