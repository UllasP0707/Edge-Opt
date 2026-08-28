from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from edge_opt.errors import ConfigurationError
from edge_opt.pruning import PolynomialPruningSchedule
from edge_opt.torch_integration import (
    PackedLinear,
    QATConfig,
    QATLinear,
    TorchActivationStatsCollector,
    TorchFakeQuantizer,
    TorchMagnitudePruner,
    TorchWandaPruner,
    collect_torch_activation_statistics,
    convert_qat,
    export_int8_bundle,
    freeze_qat_observers,
    is_torch_available,
    prepare_qat,
)

if is_torch_available():
    import torch
    import torch.nn as nn


class MissingTorchTests(unittest.TestCase):
    @unittest.skipIf(is_torch_available(), "fallback only applies without PyTorch")
    def test_missing_extra_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"edge-opt\[torch\]"):
            TorchFakeQuantizer(None)


@unittest.skipUnless(is_torch_available(), "PyTorch optional dependency is not installed")
class TorchQATTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(5)
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def test_prepare_inserts_fake_quant_nodes_and_ste_preserves_gradients(self) -> None:
        baseline = self.make_model()
        prepared, report = prepare_qat(baseline)
        self.assertEqual(report.linear_modules, 2)
        self.assertIsInstance(prepared[0], QATLinear)
        self.assertIsInstance(baseline[0], nn.Linear)
        values = torch.randn(3, 4, requires_grad=True)
        prepared(values).sum().backward()
        self.assertIsNotNone(prepared[0].weight.grad)
        self.assertGreater(int(torch.count_nonzero(prepared[0].weight.grad)), 0)

    def test_exclusion_and_observer_freeze(self) -> None:
        prepared, report = prepare_qat(
            self.make_model(), QATConfig(excluded_module_names=frozenset({"2"}))
        )
        self.assertEqual(report.linear_modules, 1)
        self.assertEqual(report.excluded_modules, ("2",))
        prepared(torch.randn(2, 4))
        count = freeze_qat_observers(prepared)
        self.assertGreater(count, 0)

    def test_convert_packs_int8_weights_and_exports_bundle(self) -> None:
        prepared, _ = prepare_qat(self.make_model())
        sample = torch.randn(4, 4)
        prepared(sample)
        converted = convert_qat(prepared)
        self.assertIsInstance(converted[0], PackedLinear)
        self.assertEqual(converted[0].qweight.dtype, torch.int8)
        self.assertEqual(tuple(converted(sample).shape), (4, 2))
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = export_int8_bundle(converted, directory)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["format"], "edge-opt-int8-v1")
            self.assertTrue((Path(directory) / "weights.npz").exists())

    def test_torch_pruner_reaches_scheduled_target(self) -> None:
        model = nn.Linear(8, 4, bias=False)
        schedule = PolynomialPruningSchedule(
            final_sparsity=0.75, begin_step=0, end_step=4, update_frequency=1
        )
        pruner = TorchMagnitudePruner(schedule)
        result = pruner.update(model, 4)
        self.assertIsNotNone(result)
        self.assertEqual(result.pruned_parameters, 24)
        self.assertEqual(int(torch.count_nonzero(model.weight)), 8)

    def test_activation_hooks_collect_linear_input_channel_statistics(self) -> None:
        model = self.make_model()
        samples = [torch.ones(2, 4), torch.full((1, 4), 2.0)]
        table = collect_torch_activation_statistics(model, samples)
        self.assertEqual(set(table.tensors), {"0", "2"})
        self.assertEqual(table.tensors["0"].channels, 4)
        self.assertEqual(table.tensors["0"].values_per_channel, 3)
        np.testing.assert_allclose(table.tensors["0"].l2_norm, np.sqrt(6.0))
        self.assertEqual(table.tensors["2"].channels, 8)
        self.assertTrue(model.training)

    def test_activation_collector_rejects_unknown_module(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "were not found"):
            TorchActivationStatsCollector(
                self.make_model(), module_names={"missing"}
            ).attach()

    def test_torch_wanda_uses_collected_activation_statistics(self) -> None:
        model = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 2))
        stats = collect_torch_activation_statistics(
            model,
            [torch.tensor([[100.0, 1.0, 1.0, 1.0]])],
        )
        pruned, result = TorchWandaPruner(0.5).prune(model, stats)
        self.assertEqual(result.actual_sparsity, 0.5)
        self.assertEqual(int(torch.count_nonzero(pruned.weight)), 4)
        self.assertEqual(float(pruned.weight[0, 0].detach()), 1.0)
        self.assertEqual(float(pruned.weight[0, 1].detach()), 0.0)
        self.assertEqual(float(model.weight[0, 1].detach()), 2.0)


if __name__ == "__main__":
    unittest.main()
