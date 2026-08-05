from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from edge_opt.errors import ConfigurationError
from edge_opt.quantization import (
    CalibrationTable,
    EntropyObserver,
    MinMaxObserver,
    QuantizationConfig,
    RepresentativeCalibrator,
    dequantize,
    fake_quantize,
    quantization_mse,
    quantize,
)


class QuantizationPrimitiveTests(unittest.TestCase):
    def test_symmetric_int8_round_trip(self) -> None:
        values = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
        observer = MinMaxObserver(QuantizationConfig(bits=8, symmetric=True))
        observer.update(values)
        params = observer.calculate_qparams()
        codes = quantize(values, params)
        restored = dequantize(codes, params)
        self.assertEqual(params.qmin, -127)
        self.assertEqual(params.qmax, 127)
        self.assertEqual(int(codes.min()), -127)
        self.assertEqual(int(codes.max()), 127)
        self.assertLessEqual(float(np.max(np.abs(restored - values))), float(params.scale))

    def test_asymmetric_quantization_preserves_zero(self) -> None:
        values = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
        observer = MinMaxObserver(QuantizationConfig(symmetric=False, narrow_range=False))
        observer.update(values)
        params = observer.calculate_qparams()
        self.assertEqual(int(quantize(np.asarray([0.0]), params)[0]), int(params.zero_point))
        self.assertAlmostEqual(float(fake_quantize(np.asarray([0.0]), params)[0]), 0.0)

    def test_per_channel_weights_receive_independent_scales(self) -> None:
        weights = np.asarray([[1.0, -1.0], [10.0, -10.0]], dtype=np.float32)
        observer = MinMaxObserver(
            QuantizationConfig(symmetric=True, per_channel=True, channel_axis=0)
        )
        observer.update(weights)
        params = observer.calculate_qparams()
        self.assertEqual(params.scale.shape, (2,))
        self.assertAlmostEqual(float(params.scale[1] / params.scale[0]), 10.0)
        restored = fake_quantize(weights, params)
        np.testing.assert_allclose(restored, weights, atol=float(params.scale.max()))

    def test_quantization_mse_is_nonnegative(self) -> None:
        values = np.asarray([-0.1, 0.2, 0.3], dtype=np.float32)
        observer = MinMaxObserver()
        observer.update(values)
        self.assertGreaterEqual(quantization_mse(values, observer.calculate_qparams()), 0.0)


class EntropyCalibrationTests(unittest.TestCase):
    def test_entropy_calibration_clips_rare_outlier(self) -> None:
        rng = np.random.default_rng(7)
        values = np.concatenate([rng.normal(0.0, 0.25, 50_000), np.asarray([20.0])])
        observer = EntropyObserver(histogram_bins=512, quantized_bins=64)
        observer.update(values[:20_000])
        observer.update(values[20_000:])
        threshold = observer.clipping_threshold()
        params = observer.calculate_qparams()
        self.assertEqual(observer.sample_count, values.size)
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 20.0)
        self.assertTrue(np.all(np.isfinite(params.scale)))

    def test_empty_observer_fails_closed(self) -> None:
        with self.assertRaises(ConfigurationError):
            EntropyObserver().calculate_qparams()

    def test_representative_pipeline_exports_named_tensors(self) -> None:
        dataset = [
            {"encoder": np.asarray([index, index + 1.0]), "decoder": np.asarray([-index])}
            for index in range(1, 5)
        ]
        calibrator = RepresentativeCalibrator(method="minmax")
        table = calibrator.calibrate(dataset)
        self.assertEqual(table.samples, 4)
        self.assertEqual(set(table.tensors), {"encoder", "decoder"})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            table.to_json(path)
            restored = CalibrationTable.from_dict(json.loads(path.read_text()))
        self.assertEqual(restored.samples, table.samples)
        np.testing.assert_allclose(restored.tensors["encoder"].scale, table.tensors["encoder"].scale)


if __name__ == "__main__":
    unittest.main()
