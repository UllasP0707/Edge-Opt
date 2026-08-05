# Edge-Opt

Edge-Opt is a hardware-aware model optimization and profiling toolchain for neural networks
targeting constrained CPUs, NPUs, DSPs, and Edge TPU-class runtimes. It combines:

- quantization-aware training (QAT) with straight-through-estimator fake quantization;
- streaming min/max and KL-divergence (entropy) activation calibration;
- deterministic magnitude pruning with a polynomial decay schedule;
- sparse encoding estimates that include index or bitmap metadata;
- cache-aware roofline analysis across L1, L2, and off-chip DRAM;
- measured micro-benchmark distributions; and
- a fail-closed quality gate whose default is strict degradation `< 0.01`.

The analytical profiler does not pretend to be a vendor simulator. Its latency predictions are
only as accurate as the target compute, bandwidth, cache, and sparse-kernel values supplied in the
hardware profile. The built-in Cortex-A76 profile is explicitly illustrative and should be replaced
with measurements from the deployment board.

## Installation

Edge-Opt requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the optional PyTorch integration for QAT and packed INT8 export:

```bash
python -m pip install -e '.[torch]'
```

## Quick start

Run the deterministic end-to-end demonstration:

```bash
edge-opt demo --output-dir reports/demo
```

It writes the source model, entropy calibration table, accepted optimized model, full JSON evidence,
and a Markdown report. The current analytical demo produces:

| Metric | FP32 baseline | INT8 + 65% pruning |
|---|---:|---:|
| Accuracy | 0.924 | 0.918 |
| Strict degradation gate | — | 0.006 `<` 0.010 (pass) |
| Encoded weights | 148.0 MiB | 17.575 MiB |
| Predicted latency | 7.764 ms | 0.923 ms |
| Predicted speedup | 1.0x | 8.41x |

These are reproducible analytical outputs from the bundled reference profile, not measured silicon
claims. Bitmap metadata is included in the optimized size, which is why the result is larger than
the ideal zero-free payload.

Profile a model graph directly:

```bash
edge-opt profile examples/spatial_audio_model.json \
  --hardware arm_cortex_a76 \
  --format markdown \
  --output reports/baseline.md
```

Apply the portable optimization description and enforce measured quality:

```bash
edge-opt optimize examples/spatial_audio_model.json \
  --baseline-quality 0.924 \
  --optimized-quality 0.918 \
  --target-sparsity 0.65 \
  --max-degradation 0.01 \
  --model-output reports/optimized-model.json \
  --output reports/optimization.md
```

The command returns status `2` and does not write an accepted optimized artifact if the degradation
is equal to or greater than the configured limit.

## PyTorch QAT workflow

```python
import torch

from edge_opt import (
    PolynomialPruningSchedule,
    QATConfig,
    TorchMagnitudePruner,
    convert_qat,
    export_int8_bundle,
    freeze_qat_observers,
    prepare_qat,
)

model = MyModel()
model, report = prepare_qat(model, QATConfig())
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
pruner = TorchMagnitudePruner(
    PolynomialPruningSchedule(
        initial_sparsity=0.0,
        final_sparsity=0.65,
        begin_step=1_000,
        end_step=10_000,
        update_frequency=100,
    )
)

for step, (inputs, targets) in enumerate(train_loader):
    pruner.update(model, step)
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    pruner.mask_gradients(model)
    optimizer.step()
    pruner.enforce(model)

freeze_qat_observers(model)
converted = convert_qat(model)
export_int8_bundle(converted, "artifacts/int8")
```

`prepare_qat` replaces supported `torch.nn.Linear` and `torch.nn.Conv2d` modules with trainable
wrappers containing input, per-channel weight, and output fake-quant nodes. Rounding and clipping
are present in the forward pass; an identity straight-through estimator carries gradients backward.
The converted modules hold real INT8 weights. Their plain-PyTorch forward method is a correctness
reference that dequantizes before the floating-point kernel. Feed the exported bundle to the target
compiler/runtime for accelerated execution.

See [examples/qat_workflow.py](examples/qat_workflow.py) for a runnable toy training loop.

## Polynomial pruning

The schedule implements the requested decay exactly:

```text
s(t) = sf + (si - sf) * (1 - (t - t0) / (tend - t0))^p
```

with `p=3` by default. Global pruning selects the lowest magnitudes across all eligible tensors;
local mode applies the target independently to each tensor. Stable name/index ordering resolves
ties deterministically. Masks are monotonic by default, serialized with the pruner state, applied to
gradients before the optimizer step, and re-enforced afterward.

## Entropy calibration

`EntropyObserver` streams finite activation values into a rebinnable absolute-magnitude histogram.
For each clipping candidate, it collapses the distribution into the target number of quantized bins,
expands it again, and selects the boundary with minimum KL divergence. This prevents a handful of
outliers from consuming most of the INT8 dynamic range. Use `MinMaxObserver` when clipping is not
acceptable or when per-channel weight ranges are required.

## Roofline model

For each operator, Edge-Opt computes:

```text
operational intensity = executed operations / transferred bytes
ridge point           = peak operations/s / tier bandwidth
attainable throughput = min(peak operations/s, bandwidth * intensity)
predicted latency     = max(compute time, transfer time + access latency)
```

The operator working set selects the smallest memory tier it fits in. Sparse compute reduces executed
operations only if the hardware profile says sparse execution is supported. Sparse storage uses the
configured `bitmap`, `coordinate`, `ideal`, or `dense` encoding and falls back to dense storage when
metadata would make compression larger.

For a measured latency distribution, use `benchmark_callable` with warmup iterations and, for
asynchronous accelerators, a device synchronization callback.

## Hardware profiles

A profile is ordinary JSON:

```json
{
  "name": "my-edge-npu",
  "peak_ops_per_second": {"fp32": 25000000000, "int8": 200000000000},
  "memory_tiers": [
    {"name": "L1", "capacity_bytes": 65536, "bandwidth_bytes_per_second": 128000000000},
    {"name": "L2", "capacity_bytes": 524288, "bandwidth_bytes_per_second": 64000000000},
    {"name": "DRAM", "capacity_bytes": null, "bandwidth_bytes_per_second": 20000000000}
  ],
  "sparse_compute_supported": true,
  "sparse_storage_supported": true
}
```

Use sustained, end-to-end measurements rather than marketing peak values when decisions depend on
the predicted latency. Cache capacities must be ordered from smallest to largest; the final backing
tier must be unbounded.

## Project layout

```text
src/edge_opt/
  core.py               portable tensor/operator/model IR
  quantization.py       observers, calibration, and NumPy quantization
  pruning.py            polynomial schedule and magnitude masks
  profiling.py          cache-aware roofline and micro-benchmarking
  torch_integration.py  optional QAT, pruning adapter, and INT8 export
  pipeline.py           optimization transform and strict quality gate
  cli.py                profile, optimize, profiles, and demo commands
```

Additional design rationale and limitations are in [docs/architecture.md](docs/architecture.md).

## Development

```bash
python -m pip install -e '.[dev]'
make test
make lint
make demo
```

The dependency-light core suite uses `unittest`; `pytest` discovers the same tests. PyTorch tests are
skipped when the optional extra is absent and run in a dedicated CI job when it is present.

## References

- Williams, Waterman, and Patterson, [Roofline: an insightful visual performance model for multicore
  architectures](https://doi.org/10.1145/1498765.1498785).
- PyTorch, [Quantization-Aware Training](https://pytorch.org/blog/quantization-aware-training/).
- PyTorch, [torchao quantization workflows](https://docs.pytorch.org/ao/stable/workflows/index.html).

## License

Apache License 2.0. See [LICENSE](LICENSE).

