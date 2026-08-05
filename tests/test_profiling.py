from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from edge_opt import (
    DType,
    HardwareProfile,
    MemoryTier,
    ModelSpec,
    OperatorKind,
    OperatorSpec,
    TensorSpec,
)
from edge_opt.hardware import load_builtin_profile
from edge_opt.profiling import RooflineProfiler, benchmark_callable, operator_flops


def target(*, sparse_compute: bool = True) -> HardwareProfile:
    return HardwareProfile(
        "test-target",
        {DType.FP32: 100e9, DType.INT8: 400e9},
        (
            MemoryTier("L1", 128, 100e9),
            MemoryTier("L2", 2_048, 20e9),
            MemoryTier("DRAM", None, 1e9),
        ),
        sparse_compute_supported=sparse_compute,
    )


class OperatorProfilingTests(unittest.TestCase):
    def test_linear_flops_are_multiply_plus_add(self) -> None:
        operator = OperatorSpec(
            "linear",
            OperatorKind.LINEAR,
            (TensorSpec((2, 4)),),
            TensorSpec((2, 3)),
            weight_shape=(3, 4),
        )
        self.assertEqual(operator_flops(operator), 48)

    def test_quantization_and_sparsity_reduce_working_set_and_compute(self) -> None:
        fp32 = OperatorSpec(
            "projection",
            OperatorKind.LINEAR,
            (TensorSpec((1, 128), DType.FP32),),
            TensorSpec((1, 128), DType.FP32),
            weight_shape=(128, 128),
            weight_dtype=DType.FP32,
        )
        optimized = OperatorSpec(
            "projection",
            OperatorKind.LINEAR,
            (TensorSpec((1, 128), DType.INT8),),
            TensorSpec((1, 128), DType.INT8),
            weight_shape=(128, 128),
            weight_dtype=DType.INT8,
            sparsity=0.75,
        )
        profiler = RooflineProfiler(target())
        baseline = profiler.profile_operator(fp32)
        result = profiler.profile_operator(optimized)
        self.assertEqual(baseline.memory_tier, "DRAM")
        self.assertEqual(result.memory_tier, "DRAM")
        self.assertLess(result.working_set_bytes, baseline.working_set_bytes)
        self.assertEqual(result.executed_flops, baseline.dense_flops // 4)
        self.assertEqual(result.weight_encoding, "bitmap")
        self.assertGreater(result.sparse_metadata_bytes, 0)

    def test_cache_tier_boundaries_are_explicit(self) -> None:
        small = OperatorSpec(
            "relu",
            OperatorKind.ACTIVATION,
            (TensorSpec((32,), DType.INT8),),
            TensorSpec((32,), DType.INT8),
        )
        medium = OperatorSpec(
            "relu",
            OperatorKind.ACTIVATION,
            (TensorSpec((256,), DType.INT8),),
            TensorSpec((256,), DType.INT8),
        )
        large = OperatorSpec(
            "relu",
            OperatorKind.ACTIVATION,
            (TensorSpec((2_000,), DType.INT8),),
            TensorSpec((2_000,), DType.INT8),
        )
        profiler = RooflineProfiler(target())
        self.assertEqual(profiler.profile_operator(small).memory_tier, "L1")
        self.assertEqual(profiler.profile_operator(medium).memory_tier, "L2")
        self.assertEqual(profiler.profile_operator(large).memory_tier, "DRAM")

    def test_unsupported_sparse_compute_retains_dense_flops(self) -> None:
        operator = OperatorSpec(
            "linear",
            OperatorKind.LINEAR,
            (TensorSpec((1, 16), DType.INT8),),
            TensorSpec((1, 16), DType.INT8),
            weight_shape=(16, 16),
            weight_dtype=DType.INT8,
            sparsity=0.5,
        )
        profile = RooflineProfiler(target(sparse_compute=False)).profile_operator(operator)
        self.assertEqual(profile.executed_flops, profile.dense_flops)


class ModelProfileTests(unittest.TestCase):
    def test_aggregate_report_serializes_json_and_markdown(self) -> None:
        operator = OperatorSpec(
            "projection",
            OperatorKind.LINEAR,
            (TensorSpec((1, 8), DType.INT8),),
            TensorSpec((1, 4), DType.INT8),
            weight_shape=(4, 8),
            weight_dtype=DType.INT8,
        )
        profile = RooflineProfiler(target()).profile(ModelSpec("tiny", (operator,)))
        self.assertEqual(profile.total_dense_flops, 64)
        self.assertIn("projection", profile.to_markdown())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.to_json(path)
            self.assertIn('"model_name": "tiny"', path.read_text())

    def test_builtin_profile_is_loadable_and_marked_illustrative(self) -> None:
        profile = load_builtin_profile("arm_cortex_a76")
        self.assertIn("illustrative-default", profile.metadata["source"])
        self.assertGreater(profile.peak_compute(DType.INT8), profile.peak_compute(DType.FP32))


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_distribution_uses_all_iterations(self) -> None:
        timestamps = iter([0.0, 0.001, 1.0, 1.003, 2.0, 2.002])
        result = benchmark_callable(
            lambda: None,
            warmup=0,
            iterations=3,
            clock=lambda: next(timestamps),
        )
        self.assertEqual(result.measured_iterations, 3)
        self.assertAlmostEqual(result.median_seconds, 0.002)
        self.assertAlmostEqual(result.min_seconds, 0.001)
        self.assertAlmostEqual(result.max_seconds, 0.003)


if __name__ == "__main__":
    unittest.main()
