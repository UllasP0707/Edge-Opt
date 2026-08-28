# Changelog

## Unreleased

- Added streaming per-channel activation statistics shared by Wanda and SmoothQuant.
- Added one-shot Wanda pruning with optional N:M constraints and PyTorch adapters.
- Added deterministic 2:4 semi-structured sparsity, storage metadata, and INT8 export tags.
- Replaced blanket sparse-compute assumptions with exact pattern/operator/dtype capabilities.
- Added a vendor-spec A100 2:4 reference profile without claiming measured end-to-end latency.
- Added SmoothQuant transforms, explicit PyTorch reference modules, and LayerNorm folding.
- Added evidence-labeled strategy comparison reports and a deterministic advanced example.

## 0.1.0 - 2026-08-04

- Added a serializable model/operator IR and configurable memory hierarchy.
- Added INT8 fake quantization, per-channel ranges, and KL entropy calibration.
- Added cubic polynomial magnitude pruning with exact, deterministic masks.
- Added sparse-storage accounting and cache-aware roofline profiling.
- Added PyTorch QAT graph rewriting, pruning hooks, packed INT8 modules, and export.
- Added a strict quality-budget optimization pipeline, CLI, reports, examples, and CI.
