# Edge-Opt

Edge-Opt is a hardware-aware model optimization and profiling toolchain for neural networks
targeting constrained CPUs, NPUs, DSPs, and Edge TPU-class runtimes. It combines:

- quantization-aware training (QAT) with straight-through-estimator fake quantization;
- streaming min/max and KL-divergence (entropy) activation calibration;
- deterministic magnitude pruning with a polynomial decay schedule;
- activation-aware Wanda pruning using per-input-channel calibration statistics;
- N:M semi-structured masks, including hardware-oriented 2:4 sparsity;
- SmoothQuant channel balancing with optional LayerNorm folding;
- sparse encoding estimates that include index or bitmap metadata;
- exact pattern/dtype/kernel capability matching in cache-aware roofline analysis;
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

Run the deterministic modern-method comparison:

```bash
python examples/advanced_optimization.py --output-dir reports/advanced
```

This evaluates magnitude pruning, Wanda, Wanda 2:4, and Wanda 2:4 + SmoothQuant on the same
synthetic fixture. Its quality column is labeled `synthetic_fixture`, while every latency and
speedup column is labeled `analytical_prediction`. It is regression evidence for the algorithms,
not a real-model accuracy or measured-board benchmark.

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

## Activation-aware optimization

One representative-data pass now supplies both Wanda's input-channel L2 norms and SmoothQuant's
input-channel absolute maxima:

```python
from edge_opt import (
    NMPruningPattern,
    TorchSmoothQuantizer,
    TorchWandaPruner,
    collect_torch_activation_statistics,
)

statistics = collect_torch_activation_statistics(model, calibration_loader)
model, pruning = TorchWandaPruner(
    0.5,
    pattern=NMPruningPattern(2, 4),
).prune(model, statistics)
model, smoothing = TorchSmoothQuantizer().transform(model, statistics)
```

Wanda scores each linear weight as `abs(weight) * input_activation_l2_norm` and prunes within each
output row. With a 2:4 constraint, exactly two weights are retained in each contiguous group of
four along the input-feature axis. This pattern improves storage estimates everywhere, but it only
receives a compute-speedup credit when the hardware profile declares an exact operator, dtype, and
2:4 kernel match.

SmoothQuant computes a per-input-channel scale
`activation_absmax**alpha / weight_absmax**(1-alpha)`, divides activations by that scale, and
multiplies the matching weight columns by it. `TorchSmoothQuantizer` keeps the division explicit for
a portable reference path. `fold_smoothquant_layer_norm` instead folds the inverse scale into an
affine LayerNorm that feeds one or more linear projections, eliminating the runtime division.

The calibration data must represent deployment inputs. After pruning or quantization, evaluate the
actual task metric on a held-out set and pass that measurement through `QualityConstraint`; output
MSE from the synthetic example is not a substitute for application accuracy.

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
operational intensity = dense-equivalent useful operations / transferred bytes
ridge point           = peak operations/s / tier bandwidth
attainable throughput = min(peak operations/s, bandwidth * intensity)
predicted latency     = max(compute time, transfer time + access latency)
```

The operator working set selects the smallest memory tier it fits in. Sparse compute reduces executed
operations only when an exact sparse capability matches the operator kind, weight dtype, sparsity,
and pattern. Sparse storage uses `bitmap`, `coordinate`, `nm`, `ideal`, or `dense` encoding and falls
back to dense storage when metadata would make compression larger. For a matched sparse kernel, the
roofline uses dense-equivalent useful work and the declared effective sparse throughput, avoiding a
double-counted speedup.

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
  "sparse_compute_capabilities": [
    {
      "operator_kind": "linear",
      "weight_dtype": "int8",
      "pattern": "2:4",
      "effective_peak_ops_per_second": 400000000000,
      "backend": "measured vendor 2:4 GEMM",
      "performance_source": "measured"
    }
  ],
  "sparse_storage_supported": true
}
```

Use sustained, end-to-end measurements rather than marketing peak values when decisions depend on
the predicted latency. A legacy blanket `sparse_compute_supported: true` is rejected because it
cannot establish which patterns or kernels are executable. Cache capacities must be ordered from
smallest to largest; the final backing tier must be unbounded.

The bundled `arm_cortex_a76` profile is illustrative and declares no sparse-compute kernel. The
bundled `nvidia_a100_reference` profile uses vendor-spec dense and 2:4 compute peaks, but its cache
bandwidth and latency values are illustrative; its reports therefore state that latency is not a
measured end-to-end result.

## Project layout

```text
src/edge_opt/
  activation.py         reusable per-channel representative-data statistics
  comparison.py         evidence-labeled strategy comparison reports
  core.py               portable tensor/operator/model IR
  quantization.py       observers, calibration, and NumPy quantization
  pruning.py            polynomial schedule and magnitude masks
  smoothquant.py        activation-aware W8A8 channel balancing
  structured.py         deterministic N:M masks and metadata accounting
  wanda.py              activation-aware pruning scores and masks
  profiling.py          cache-aware roofline and micro-benchmarking
  torch_integration.py  optional QAT, modern transforms, and INT8 export
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
- Sun et al., [A Simple and Effective Pruning Approach for Large Language
  Models (Wanda)](https://arxiv.org/abs/2306.11695).
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for
  Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html).
- Mishra et al., [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378).
- Yuan et al., [LLM Inference Unveiled: Survey and Roofline Model
  Insights](https://arxiv.org/abs/2402.16363).
- PyTorch, [Quantization-Aware Training](https://pytorch.org/blog/quantization-aware-training/).
- PyTorch, [torchao quantization workflows](https://docs.pytorch.org/ao/stable/workflows/index.html).

## License

Apache License 2.0. See [LICENSE](LICENSE).
