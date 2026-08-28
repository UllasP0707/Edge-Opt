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
    SparseComputeCapability,
    TensorSpec,
)
from edge_opt.hardware import load_builtin_profile
from edge_opt.profiling import RooflineProfiler, benchmark_callable, operator_flops


def target(*, structured_compute: bool = False) -> HardwareProfile:
    capabilities = (
        SparseComputeCapability(
            OperatorKind.LINEAR,
            DType.INT8,
            "2:4",
            800e9,
            "test 2:4 kernel",
            "measured",
        ),
    ) if structured_compute else ()
    return HardwareProfile(
        "test-target",
        {DType.FP32: 100e9, DType.INT8: 400e9},
        (
            MemoryTier("L1", 128, 100e9),
            MemoryTier("L2", 2_048, 20e9),
            MemoryTier("DRAM", None, 1e9),
        ),
        sparse_compute_capabilities=capabilities,
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

    def test_unstructured_sparsity_reduces_storage_but_not_compute(self) -> None:
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
        self.assertEqual(result.executed_flops, result.dense_flops)
        self.assertFalse(result.sparse_compute_accelerated)
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

    def test_exact_two_of_four_capability_enables_sparse_compute(self) -> None:
        operator = OperatorSpec(
            "linear",
            OperatorKind.LINEAR,
            (TensorSpec((1, 16), DType.INT8),),
            TensorSpec((1, 16), DType.INT8),
            weight_shape=(16, 16),
            weight_dtype=DType.INT8,
            sparsity=0.5,
            attributes={"sparse_encoding": "nm", "sparsity_pattern": "2:4"},
        )
        unsupported = RooflineProfiler(target()).profile_operator(operator)
        supported = RooflineProfiler(target(structured_compute=True)).profile_operator(operator)
        self.assertEqual(unsupported.executed_flops, unsupported.dense_flops)
        self.assertFalse(unsupported.sparse_compute_accelerated)
        self.assertEqual(supported.executed_flops, supported.dense_flops // 2)
        self.assertTrue(supported.sparse_compute_accelerated)
        self.assertEqual(supported.compute_backend, "test 2:4 kernel")
        self.assertEqual(supported.compute_performance_source, "measured")
        self.assertEqual(supported.weight_encoding, "nm-2:4")

    def test_capability_does_not_match_unstructured_or_wrong_dtype(self) -> None:
        operator = OperatorSpec(
            "linear",
            OperatorKind.LINEAR,
            (TensorSpec((1, 16), DType.FP32),),
            TensorSpec((1, 16), DType.FP32),
            weight_shape=(16, 16),
            weight_dtype=DType.FP32,
            sparsity=0.5,
            attributes={"sparse_encoding": "bitmap", "sparsity_pattern": "2:4"},
        )
        profile = RooflineProfiler(target(structured_compute=True)).profile_operator(operator)
        self.assertFalse(profile.sparse_compute_accelerated)
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
        self.assertIn("not measured", profile.metadata["warning"].lower())
        self.assertGreater(profile.peak_compute(DType.INT8), profile.peak_compute(DType.FP32))
        self.assertEqual(profile.sparse_compute_capabilities, ())

    def test_a100_reference_only_accelerates_declared_two_of_four_kernel(self) -> None:
        profile = load_builtin_profile("nvidia_a100_reference")
        self.assertIn("not measured", profile.metadata["warning"].lower())
        self.assertTrue(profile.sparse_compute_capabilities)


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
