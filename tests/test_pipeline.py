from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from edge_opt import DType, ModelSpec, OperatorKind, OperatorSpec, TensorSpec
from edge_opt.cli import main
from edge_opt.errors import AccuracyBudgetExceeded
from edge_opt.hardware import load_builtin_profile
from edge_opt.pipeline import (
    OptimizationConfig,
    OptimizationPipeline,
    QualityConstraint,
    optimize_model_spec,
)


def tiny_model() -> ModelSpec:
    return ModelSpec(
        "tiny",
        (
            OperatorSpec(
                "projection",
                OperatorKind.LINEAR,
                (TensorSpec((1, 64), DType.FP32),),
                TensorSpec((1, 32), DType.FP32),
                weight_shape=(32, 64),
            ),
        ),
    )


class QualityGateTests(unittest.TestCase):
    def test_default_gate_is_strictly_less_than_one_percent(self) -> None:
        constraint = QualityConstraint()
        self.assertTrue(constraint.assess(0.90, 0.891).passed)
        self.assertFalse(constraint.assess(0.90, 0.89).passed)

    def test_lower_is_better_metric_reverses_degradation(self) -> None:
        constraint = QualityConstraint("mse", higher_is_better=False, max_degradation=0.01)
        self.assertTrue(constraint.assess(0.02, 0.025).passed)
        self.assertFalse(constraint.assess(0.02, 0.031).passed)


class OptimizationPipelineTests(unittest.TestCase):
    def test_static_transform_quantizes_and_prunes_weighted_ops(self) -> None:
        optimized = optimize_model_spec(
            tiny_model(), OptimizationConfig(target_sparsity=0.75)
        )
        operator = optimized.operators[0]
        self.assertEqual(operator.weight_dtype, DType.INT8)
        self.assertEqual(operator.inputs[0].dtype, DType.INT8)
        self.assertEqual(operator.sparsity, 0.75)
        self.assertEqual(operator.attributes["sparse_encoding"], "bitmap")

    def test_pipeline_calibrates_profiles_and_accepts_within_budget(self) -> None:
        pipeline = OptimizationPipeline(load_builtin_profile("arm_cortex_a76"))
        result = pipeline.run(
            tiny_model(),
            baseline_quality=0.92,
            optimized_quality=0.915,
            representative_data=[{"projection": np.linspace(-1, 1, 256)}],
        )
        self.assertTrue(result.quality.passed)
        self.assertIsNotNone(result.calibration)
        self.assertLess(
            result.optimized_profile.weight_bytes, result.baseline_profile.weight_bytes
        )
        self.assertGreater(result.predicted_speedup, 1.0)
        self.assertIn("Quality gate: **PASS**", result.to_markdown())

    def test_pipeline_rejects_artifact_outside_accuracy_budget(self) -> None:
        pipeline = OptimizationPipeline(load_builtin_profile("arm_cortex_a76"))
        with self.assertRaisesRegex(AccuracyBudgetExceeded, "artifact rejected"):
            pipeline.run(
                tiny_model(), baseline_quality=0.92, optimized_quality=0.90
            )


class CliTests(unittest.TestCase):
    def test_demo_writes_reproducible_report_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = main(["demo", "--output-dir", directory])
            self.assertEqual(status, 0)
            outputs = {path.name for path in Path(directory).iterdir()}
        self.assertEqual(
            outputs,
            {"model.json", "optimization.json", "optimization.md", "calibration.json"},
        )

    def test_cli_returns_nonzero_for_rejected_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            model_path.write_text(__import__("json").dumps(tiny_model().to_dict()))
            status = main(
                [
                    "optimize",
                    str(model_path),
                    "--baseline-quality",
                    "0.9",
                    "--optimized-quality",
                    "0.88",
                ]
            )
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()

