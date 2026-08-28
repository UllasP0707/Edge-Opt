# Architecture and design constraints

Edge-Opt separates optimization behavior from deployment-backend behavior. The core algorithms use
NumPy and a small, serializable intermediate representation (IR). PyTorch is an optional adapter, and
vendor compilers consume the packed export rather than being hidden behind a misleading generic
latency number.

## Data flow

```mermaid
flowchart LR
  A["FP32 model and measured baseline quality"] --> B{"Pruning strategy"}
  B -->|baseline| C["Polynomial magnitude pruning"]
  B -->|activation-aware| D["Wanda pruning"]
  E["Representative activation tensors"] --> F["Per-channel L2 and absmax statistics"]
  F --> D
  D --> G["Optional N:M constraint"]
  F --> H["SmoothQuant channel balancing"]
  C --> I["QAT or PTQ"]
  G --> H
  H --> I
  E --> J["Entropy calibration"]
  J --> I
  I --> K["Packed INT8 weights and ModelSpec"]
  L["Measured or explicitly sourced hardware profile"] --> M["Cache-aware roofline profiler"]
  K --> M
  M --> N["Analytical latency, SRAM, DRAM, and bottleneck report"]
  K --> O["Measure optimized task quality"]
  O --> P{"Degradation < limit?"}
  P -->|yes| Q["Accepted artifact and evidence bundle"]
  P -->|no| R["Reject without accepted artifact"]
```

The quality measurement is deliberately external to the static transform. Accuracy cannot be
inferred from sparsity, calibration error, or model size; it must be evaluated on the task's held-out
dataset. `OptimizationPipeline` accepts those measurements and applies the gate.

## Quantization choices

- Weights default to signed symmetric INT8, with independent output-channel scales.
- Activations default to signed symmetric INT8 but support unsigned affine quantization.
- Fake quantization calculates `round(x / scale + zero_point)`, clamps to the integer range, and
  dequantizes back to the training dtype.
- The PyTorch backward pass uses `x + (fake_quantized(x) - x).detach()`, the identity STE.
- Entropy calibration is per tensor. Per-channel calibration uses the min/max observer.
- Channel statistics retain both L2 norms for Wanda and absolute maxima for SmoothQuant.
- SmoothQuant scales corresponding activation channels and weight columns inversely, preserving the
  floating-point function. Its explicit PyTorch reference path exports the scale; its LayerNorm fold
  removes runtime division when the graph relationship is supplied explicitly.
- Packed PyTorch modules are reference kernels, not claims of native integer execution.

## Sparsity and storage are separate capabilities

A zero weight only reduces latency if the target has a matching sparse kernel, and only reduces
memory if the runtime stores a sparse representation. `HardwareProfile` therefore keeps
`sparse_storage_supported` separate from a list of exact `SparseComputeCapability` entries. Each
entry binds an operator kind, weight dtype, N:M pattern, backend, effective peak, and evidence
source. A blanket sparse-compute boolean is intentionally rejected.

The default bitmap representation stores one bit per original weight plus each nonzero value. The
coordinate representation accounts for configurable index width and block size. `ideal` exists to
show the theoretical payload floor but should not be used for deployment planning unless the runtime
really provides equivalent compression. N:M storage includes the information-theoretic minimum bits
needed to identify retained positions in each group. That is a storage lower bound, not a claim that
every runtime uses an equally compact layout.

## Roofline interpretation

Edge-Opt counts a multiply and an add as two operations. A layer's input, output, and encoded weights
form its working set. This deliberately simple static model does not yet simulate tiling, tensor
lifetime overlap, cache conflict misses, DMA/compute overlap, operator fusion, thermal throttling, or
runtime scheduling. It is appropriate for sensitivity analysis and early architecture decisions.
Final latency claims require `benchmark_callable` on the intended runtime and board.

Operational intensity uses dense-equivalent useful operations divided by transferred bytes. When an
exact sparse capability matches, physical executed operations are reported separately and latency
uses the capability's effective sparse throughput. This convention prevents counting a 2:4 benefit
once by halving work and again by doubling effective peak. Every JSON and Markdown report labels the
latency as `analytical_prediction` and carries the hardware profile's source and warning.

## Supported static operators

The IR derives operation counts for linear, 2-D convolution, depthwise convolution, matrix
multiplication, activation, pooling, and elementwise operators. Other operations must provide an
explicit nonnegative `flops` attribute. This fails early instead of silently treating unknown work as
free.

## Quality-gate semantics

For higher-is-better metrics, degradation is `baseline - optimized`. For lower-is-better metrics it
is `optimized - baseline`. Improvements are negative degradation and pass. The default limit is
strict: a value equal to `0.01`, within floating-point comparison tolerance, fails. Non-finite quality
measurements are invalid.

## Extension points

- Add a hardware target by constructing a `HardwareProfile` or loading its JSON.
- Add a sparse kernel only through an exact `SparseComputeCapability`; do not infer it from zeros.
- Add vendor packing by translating `edge-opt-int8-v1/manifest.json` and `weights.npz`.
- Add a framework adapter that emits `ModelSpec` and consumes calibration tables.
- Add operator-specific FLOP calculation in `operator_flops` while retaining explicit validation.
- Replace the analytical latency result with board measurements in downstream reports.
