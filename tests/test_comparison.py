from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from edge_opt.comparison import StrategyComparison, StrategyMeasurement
from edge_opt.core import DType, ModelSpec, OperatorKind, OperatorSpec, TensorSpec
from edge_opt.errors import ConfigurationError
from edge_opt.hardware import load_builtin_profile
from edge_opt.pipeline import OptimizationConfig, QualityConstraint, optimize_model_spec
from edge_opt.profiling import RooflineProfiler


def _model() -> ModelSpec:
    return ModelSpec(
        "comparison",
        (
            OperatorSpec(
                "linear",
                OperatorKind.LINEAR,
                (TensorSpec((8, 16), DType.FP32),),
                TensorSpec((8, 8), DType.FP32),
                weight_shape=(8, 16),
            ),
        ),
    )


class StrategyComparisonTests(unittest.TestCase):
    def test_report_labels_prediction_and_preserves_rejected_candidate(self) -> None:
        profiler = RooflineProfiler(load_builtin_profile("arm_cortex_a76"))
        model = _model()
        baseline = StrategyMeasurement(
            "FP32",
            "none",
            "none",
            0.90,
            "measured",
            profiler.profile(model),
        )
        candidate_model = optimize_model_spec(
            model, OptimizationConfig(target_sparsity=0.5)
        )
        accepted = StrategyMeasurement(
            "Wanda",
            "Wanda 50%",
            "W8A8",
            0.895,
            "measured",
            profiler.profile(candidate_model),
        )
        rejected = StrategyMeasurement(
            "over-pruned",
            "magnitude 90%",
            "W8A8",
            0.87,
            "measured",
            profiler.profile(candidate_model),
        )
        comparison = StrategyComparison(
            baseline,
            (accepted, rejected),
            QualityConstraint(max_degradation=0.01),
        )
        self.assertEqual(comparison.accepted_candidates, (accepted,))
        self.assertIn("Latency evidence: **analytical prediction**", comparison.to_markdown())
        self.assertIn("FAIL", comparison.to_markdown())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            comparison.to_json(path)
            self.assertIn('"latency": "analytical_prediction"', path.read_text())

    def test_mixed_hardware_targets_fail_closed(self) -> None:
        model = _model()
        cortex = RooflineProfiler(load_builtin_profile("arm_cortex_a76")).profile(model)
        a100 = RooflineProfiler(load_builtin_profile("nvidia_a100_reference")).profile(model)
        baseline = StrategyMeasurement(
            "baseline", "none", "none", 0.0, "synthetic_fixture", cortex
        )
        candidate = StrategyMeasurement(
            "candidate", "none", "none", 0.0, "synthetic_fixture", a100
        )
        with self.assertRaisesRegex(ConfigurationError, "same hardware"):
            StrategyComparison(
                baseline,
                (candidate,),
                QualityConstraint("mse", higher_is_better=False),
            )


if __name__ == "__main__":
    unittest.main()
