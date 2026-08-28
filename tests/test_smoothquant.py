from __future__ import annotations

import unittest

import numpy as np

from edge_opt.activation import ChannelStatistics
from edge_opt.errors import ConfigurationError
from edge_opt.quantization import MinMaxObserver, QuantizationConfig, fake_quantize
from edge_opt.smoothquant import (
    SmoothQuantConfig,
    apply_smoothquant,
    smoothquant_scales,
)


def _simulated_w8a8_output(activations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    activation_observer = MinMaxObserver(QuantizationConfig(symmetric=True))
    activation_observer.update(activations)
    weight_observer = MinMaxObserver(
        QuantizationConfig(symmetric=True, per_channel=True, channel_axis=0)
    )
    weight_observer.update(weights)
    quantized_activations = fake_quantize(
        activations, activation_observer.calculate_qparams()
    )
    quantized_weights = fake_quantize(weights, weight_observer.calculate_qparams())
    return quantized_activations @ quantized_weights.T


class SmoothQuantTests(unittest.TestCase):
    def test_scales_follow_paper_equation_and_accept_collected_statistics(self) -> None:
        weights = np.asarray([[0.25, 4.0], [-1.0, 2.0]])
        statistics = ChannelStatistics(
            l2_norm=np.asarray([10.0, 2.0]),
            absmax=np.asarray([100.0, 1.0]),
            values_per_channel=4,
            batches=1,
            channel_axis=-1,
        )
        scales = smoothquant_scales(
            weights, statistics, SmoothQuantConfig(alpha=0.5)
        )
        np.testing.assert_allclose(scales, np.asarray([10.0, 0.5]))

    def test_transform_is_functionally_equivalent_in_floating_point(self) -> None:
        activations = np.asarray(
            [[100.0, 1.0], [50.0, -1.0], [-70.0, 0.5], [0.0, 1.0]]
        )
        weights = np.asarray([[0.01, 1.0], [-0.01, 0.5]])
        result = apply_smoothquant(weights, np.max(np.abs(activations), axis=0))
        baseline = activations @ weights.T
        transformed = result.transform_activations(activations) @ result.smoothed_weights.T
        np.testing.assert_allclose(transformed, baseline, rtol=1e-12, atol=1e-12)

    def test_transform_reduces_w8a8_error_for_activation_outlier_fixture(self) -> None:
        activations = np.asarray(
            [[100.0, 1.0], [50.0, -1.0], [-70.0, 0.5], [0.0, 1.0]]
        )
        weights = np.asarray([[0.01, 1.0], [-0.01, 0.5]])
        reference = activations @ weights.T
        baseline = _simulated_w8a8_output(activations, weights)
        result = apply_smoothquant(weights, np.max(np.abs(activations), axis=0))
        smoothed = _simulated_w8a8_output(
            result.transform_activations(activations), result.smoothed_weights
        )
        baseline_mse = float(np.mean(np.square(reference - baseline)))
        smoothed_mse = float(np.mean(np.square(reference - smoothed)))
        self.assertLess(smoothed_mse, baseline_mse / 100.0)

    def test_zero_channels_are_stabilized_and_invalid_shapes_fail_closed(self) -> None:
        result = apply_smoothquant(np.zeros((2, 3)), np.zeros(3))
        self.assertTrue(np.all(np.isfinite(result.scales)))
        self.assertTrue(np.all(result.scales > 0))
        with self.assertRaisesRegex(ConfigurationError, "channels must match"):
            apply_smoothquant(np.ones((2, 3)), np.ones(2))
        with self.assertRaisesRegex(ConfigurationError, "alpha"):
            SmoothQuantConfig(alpha=1.1)


if __name__ == "__main__":
    unittest.main()
