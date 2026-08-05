from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from edge_opt import (
    ConfigurationError,
    DType,
    HardwareProfile,
    MemoryTier,
    ModelSpec,
    OperatorKind,
    OperatorSpec,
    TensorSpec,
)


class CoreModelTests(unittest.TestCase):
    def test_tensor_storage_supports_packed_int4(self) -> None:
        tensor = TensorSpec((3,), DType.INT4)
        self.assertEqual(tensor.numel, 3)
        self.assertEqual(tensor.storage_bytes, 2)

    def test_model_round_trip_is_lossless(self) -> None:
        operator = OperatorSpec(
            name="projection",
            kind=OperatorKind.LINEAR,
            inputs=(TensorSpec((1, 16), name="input"),),
            output=TensorSpec((1, 8), name="output"),
            weight_shape=(8, 16),
            sparsity=0.5,
            attributes={"bias": True},
        )
        model = ModelSpec("tiny", (operator,), {"task": "test"})
        restored = ModelSpec.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(restored, model)
        self.assertEqual(model.dense_weight_bytes, 8 * 16 * 4)

    def test_invalid_shape_and_duplicate_names_fail_early(self) -> None:
        with self.assertRaises(ConfigurationError):
            TensorSpec((1, 0))
        operator = OperatorSpec(
            "duplicate",
            OperatorKind.ACTIVATION,
            (TensorSpec((1, 4)),),
            TensorSpec((1, 4)),
        )
        with self.assertRaises(ConfigurationError):
            ModelSpec("broken", (operator, operator))


class HardwareProfileTests(unittest.TestCase):
    def make_profile(self) -> HardwareProfile:
        return HardwareProfile(
            name="test-npu",
            peak_ops_per_second={DType.FP32: 1e9, DType.INT8: 4e9},
            memory_tiers=(
                MemoryTier("L1", 32 * 1024, 64e9, 1e-9),
                MemoryTier("L2", 512 * 1024, 16e9, 5e-9),
                MemoryTier("DRAM", None, 4e9, 80e-9),
            ),
            sparse_compute_supported=True,
        )

    def test_json_round_trip(self) -> None:
        profile = self.make_profile()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.to_json(path)
            restored = HardwareProfile.from_json(path)
        self.assertEqual(restored, profile)
        self.assertEqual(restored.peak_compute(DType.INT8), 4e9)

    def test_tiers_require_unbounded_backing_memory(self) -> None:
        with self.assertRaises(ConfigurationError):
            HardwareProfile(
                "broken",
                {DType.FP32: 1.0},
                (MemoryTier("L1", 1024, 1.0),),
            )


if __name__ == "__main__":
    unittest.main()
