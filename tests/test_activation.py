from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from edge_opt.activation import (
    ActivationStatisticsCollector,
    ActivationStatisticsTable,
    ChannelStatsObserver,
)
from edge_opt.errors import ConfigurationError


class ChannelStatsObserverTests(unittest.TestCase):
    def test_last_axis_statistics_match_exact_l2_and_absmax(self) -> None:
        values = np.asarray([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        observer = ChannelStatsObserver(channel_axis=-1)
        observer.update(values)
        stats = observer.calculate()
        np.testing.assert_allclose(stats.l2_norm, np.sqrt([17.0, 29.0, 45.0]))
        np.testing.assert_array_equal(stats.absmax, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(stats.rms, np.sqrt([17.0, 29.0, 45.0]) / np.sqrt(2))
        self.assertEqual(stats.channels, 3)
        self.assertEqual(stats.values_per_channel, 2)
        self.assertEqual(stats.batches, 1)

    def test_streaming_updates_are_chunking_invariant(self) -> None:
        values = np.arange(24.0).reshape(2, 3, 4)
        whole = ChannelStatsObserver(channel_axis=1)
        whole.update(values)
        streamed = ChannelStatsObserver(channel_axis=1)
        streamed.update(values[:1])
        streamed.update(values[1:])
        np.testing.assert_allclose(streamed.calculate().l2_norm, whole.calculate().l2_norm)
        np.testing.assert_allclose(streamed.calculate().absmax, whole.calculate().absmax)

    def test_invalid_values_and_channel_changes_fail_closed(self) -> None:
        observer = ChannelStatsObserver()
        with self.assertRaises(ConfigurationError):
            observer.update([1.0, np.nan])
        observer.update(np.ones((2, 3)))
        with self.assertRaises(ConfigurationError):
            observer.update(np.ones((2, 4)))


class ActivationStatisticsCollectorTests(unittest.TestCase):
    def test_named_axes_and_json_round_trip(self) -> None:
        collector = ActivationStatisticsCollector(
            channel_axes={"conv": 1}, default_channel_axis=-1
        )
        table = collector.collect(
            [
                {
                    "linear": np.ones((2, 4)),
                    "conv": np.ones((2, 3, 2, 2)),
                },
                {
                    "linear": np.full((1, 4), 2.0),
                    "conv": np.full((1, 3, 1, 1), 2.0),
                },
            ]
        )
        self.assertEqual(table.tensors["linear"].channels, 4)
        self.assertEqual(table.tensors["linear"].values_per_channel, 3)
        self.assertEqual(table.tensors["conv"].channels, 3)
        self.assertEqual(table.tensors["conv"].values_per_channel, 9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "channel-stats.json"
            table.to_json(path)
            restored = ActivationStatisticsTable.from_json(path)
        np.testing.assert_allclose(
            restored.tensors["linear"].l2_norm, table.tensors["linear"].l2_norm
        )

    def test_empty_dataset_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            ActivationStatisticsCollector().collect([])


if __name__ == "__main__":
    unittest.main()
