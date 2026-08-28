"""Cache-aware roofline profiling and repeatable latency micro-benchmarks."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .core import ModelSpec, OperatorKind, OperatorSpec
from .errors import ConfigurationError
from .hardware import HardwareProfile, MemoryTier
from .pruning import estimate_sparse_storage
from .structured import NMPruningPattern

Bottleneck = Literal["compute", "memory"]


def operator_flops(operator: OperatorSpec) -> int:
    """Calculate dense multiply/add operations for a supported operator."""

    if "flops" in operator.attributes:
        flops = int(operator.attributes["flops"])
        if flops < 0:
            raise ConfigurationError(f"operator {operator.name!r} has negative FLOPs")
        return flops

    if operator.kind is OperatorKind.LINEAR:
        if operator.weight_shape is None or len(operator.weight_shape) != 2:
            raise ConfigurationError("linear operators require [out, in] weight_shape")
        input_features = operator.weight_shape[1]
        return 2 * operator.output.numel * input_features

    if operator.kind in {OperatorKind.CONV2D, OperatorKind.DEPTHWISE_CONV2D}:
        if operator.weight_shape is None or len(operator.weight_shape) != 4:
            raise ConfigurationError("convolution operators require [out, in/groups, kh, kw]")
        kernel_work = int(np.prod(operator.weight_shape[1:]))
        return 2 * operator.output.numel * kernel_work

    if operator.kind is OperatorKind.MATMUL:
        if {"m", "n", "k"}.issubset(operator.attributes):
            batch = int(operator.attributes.get("batch", 1))
            return 2 * batch * int(operator.attributes["m"]) * int(
                operator.attributes["n"]
            ) * int(operator.attributes["k"])
        if len(operator.inputs) < 2:
            raise ConfigurationError("matmul needs two inputs or explicit m/n/k attributes")
        left, right = operator.inputs[:2]
        if len(left.shape) < 2 or len(right.shape) < 2:
            raise ConfigurationError("matmul inputs must have rank >= 2")
        batch = int(np.prod(operator.output.shape[:-2])) if len(operator.output.shape) > 2 else 1
        return 2 * batch * left.shape[-2] * right.shape[-1] * left.shape[-1]

    if operator.kind in {
        OperatorKind.ACTIVATION,
        OperatorKind.POOL,
        OperatorKind.ELEMENTWISE,
    }:
        return operator.output.numel * int(operator.attributes.get("ops_per_element", 1))

    raise ConfigurationError(
        f"operator {operator.name!r} needs an explicit 'flops' attribute for kind "
        f"{operator.kind.value!r}"
    )


@dataclass(frozen=True)
class WeightStorage:
    """Encoded weight footprint and selected sparse representation."""

    bytes: int
    dense_bytes: int
    encoding: str
    metadata_bytes: int


def weight_storage(operator: OperatorSpec, hardware: HardwareProfile) -> WeightStorage:
    """Calculate target-side weight storage, including sparse metadata."""

    dense_bytes = operator.dense_weight_bytes
    if dense_bytes == 0 or operator.sparsity == 0 or not hardware.sparse_storage_supported:
        return WeightStorage(dense_bytes, dense_bytes, "dense", 0)

    encoding = str(operator.attributes.get("sparse_encoding", "bitmap"))
    nonzero = int(np.ceil(operator.weight_elements * (1.0 - operator.sparsity)))
    value_bytes = (nonzero * operator.weight_dtype.bits + 7) // 8
    if encoding == "ideal":
        encoded_bytes = value_bytes
        metadata_bytes = 0
    elif encoding == "bitmap":
        metadata_bytes = (operator.weight_elements + 7) // 8
        encoded_bytes = value_bytes + metadata_bytes
    elif encoding == "coordinate":
        estimate = estimate_sparse_storage(
            operator.weight_elements,
            operator.sparsity,
            value_bits=operator.weight_dtype.bits,
            index_bits=int(operator.attributes.get("sparse_index_bits", 16)),
            block_size=int(operator.attributes.get("sparse_block_size", 1)),
        )
        encoded_bytes = estimate.sparse_bytes
        metadata_bytes = estimate.index_bytes
    elif encoding == "nm":
        pattern_label = operator.attributes.get("sparsity_pattern")
        if not isinstance(pattern_label, str):
            raise ConfigurationError("N:M storage requires a sparsity_pattern attribute")
        try:
            n_text, m_text = pattern_label.split(":", maxsplit=1)
            pattern = NMPruningPattern(int(n_text), int(m_text))
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(
                f"invalid N:M sparsity pattern {pattern_label!r}"
            ) from exc
        if not np.isclose(operator.sparsity, pattern.sparsity):
            raise ConfigurationError(
                f"operator sparsity does not match declared {pattern.label} pattern"
            )
        if operator.weight_elements % pattern.m:
            raise ConfigurationError("N:M weight count is not divisible by group size")
        groups = operator.weight_elements // pattern.m
        nonzero = groups * pattern.n
        value_bytes = (nonzero * operator.weight_dtype.bits + 7) // 8
        metadata_bytes = (
            groups * pattern.minimum_metadata_bits_per_group + 7
        ) // 8
        encoded_bytes = value_bytes + metadata_bytes
        encoding = f"nm-{pattern.label}"
    elif encoding == "dense":
        encoded_bytes = dense_bytes
        metadata_bytes = 0
    else:
        raise ConfigurationError(f"unknown sparse encoding {encoding!r}")

    if encoded_bytes >= dense_bytes:
        return WeightStorage(dense_bytes, dense_bytes, "dense", 0)
    return WeightStorage(encoded_bytes, dense_bytes, encoding, metadata_bytes)


def select_memory_tier(working_set_bytes: int, tiers: tuple[MemoryTier, ...]) -> MemoryTier:
    """Choose the smallest cache or backing tier able to hold the working set."""

    for tier in tiers:
        if tier.capacity_bytes is None or working_set_bytes <= tier.capacity_bytes:
            return tier
    raise ConfigurationError("memory hierarchy has no tier capable of holding the working set")


@dataclass(frozen=True)
class OperatorProfile:
    """Roofline metrics for one model operator."""

    name: str
    kind: str
    arithmetic_dtype: str
    dense_flops: int
    executed_flops: int
    activation_bytes: int
    weight_bytes: int
    dense_weight_bytes: int
    sparse_metadata_bytes: int
    weight_encoding: str
    total_bytes: int
    working_set_bytes: int
    memory_tier: str
    operational_intensity: float
    ridge_point: float
    attainable_ops_per_second: float
    compute_time_seconds: float
    memory_time_seconds: float
    predicted_latency_seconds: float
    bottleneck: Bottleneck
    sparse_compute_accelerated: bool
    sparsity_pattern: str | None
    compute_backend: str | None
    compute_performance_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "arithmetic_dtype": self.arithmetic_dtype,
            "dense_flops": self.dense_flops,
            "executed_flops": self.executed_flops,
            "activation_bytes": self.activation_bytes,
            "weight_bytes": self.weight_bytes,
            "dense_weight_bytes": self.dense_weight_bytes,
            "sparse_metadata_bytes": self.sparse_metadata_bytes,
            "weight_encoding": self.weight_encoding,
            "total_bytes": self.total_bytes,
            "working_set_bytes": self.working_set_bytes,
            "memory_tier": self.memory_tier,
            "operational_intensity": self.operational_intensity,
            "ridge_point": self.ridge_point,
            "attainable_ops_per_second": self.attainable_ops_per_second,
            "compute_time_seconds": self.compute_time_seconds,
            "memory_time_seconds": self.memory_time_seconds,
            "predicted_latency_seconds": self.predicted_latency_seconds,
            "bottleneck": self.bottleneck,
            "sparse_compute_accelerated": self.sparse_compute_accelerated,
            "sparsity_pattern": self.sparsity_pattern,
            "compute_backend": self.compute_backend,
            "compute_performance_source": self.compute_performance_source,
        }


@dataclass(frozen=True)
class ModelProfile:
    """Aggregated roofline report for a complete model graph."""

    model_name: str
    hardware_name: str
    operators: tuple[OperatorProfile, ...]
    total_dense_flops: int
    total_executed_flops: int
    total_bytes: int
    weight_bytes: int
    dense_weight_bytes: int
    operational_intensity: float
    predicted_latency_seconds: float
    bottleneck_counts: dict[str, int]
    memory_tier_counts: dict[str, int]

    @property
    def weight_compression_ratio(self) -> float:
        return self.dense_weight_bytes / self.weight_bytes if self.weight_bytes else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hardware_name": self.hardware_name,
            "summary": {
                "total_dense_flops": self.total_dense_flops,
                "total_executed_flops": self.total_executed_flops,
                "total_bytes": self.total_bytes,
                "weight_bytes": self.weight_bytes,
                "dense_weight_bytes": self.dense_weight_bytes,
                "weight_compression_ratio": self.weight_compression_ratio,
                "operational_intensity": self.operational_intensity,
                "predicted_latency_seconds": self.predicted_latency_seconds,
                "bottleneck_counts": self.bottleneck_counts,
                "memory_tier_counts": self.memory_tier_counts,
            },
            "operators": [operator.to_dict() for operator in self.operators],
        }

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def to_markdown(self) -> str:
        lines = [
            f"# Edge-Opt profile: {self.model_name}",
            "",
            f"Target: **{self.hardware_name}**",
            "",
            f"- Predicted latency: {self.predicted_latency_seconds * 1_000:.3f} ms",
            f"- Executed operations: {self.total_executed_flops:,}",
            f"- Data movement: {self.total_bytes:,} bytes",
            f"- Operational intensity: {self.operational_intensity:.3f} ops/byte",
            f"- Encoded weights: {self.weight_bytes:,} bytes "
            f"({self.weight_compression_ratio:.2f}x compression)",
            "",
            "| Operator | Tier | Encoding | Sparse kernel | Ops/byte | Bound | Latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines.extend(
            f"| {item.name} | {item.memory_tier} | {item.weight_encoding} | "
            f"{item.compute_backend or 'none'} | "
            f"{item.operational_intensity:.3f} | {item.bottleneck} | "
            f"{item.predicted_latency_seconds * 1_000:.3f} |"
            for item in self.operators
        )
        return "\n".join(lines) + "\n"


class RooflineProfiler:
    """Static profiler using target compute peaks and cache-tier bandwidth."""

    def __init__(self, hardware: HardwareProfile) -> None:
        self.hardware = hardware

    def profile_operator(self, operator: OperatorSpec) -> OperatorProfile:
        dense_flops = operator_flops(operator)
        sparse_capability = self.hardware.sparse_capability(operator)
        if sparse_capability is not None:
            executed_flops = int(np.ceil(dense_flops * (1.0 - operator.sparsity)))
        else:
            executed_flops = dense_flops
        storage = weight_storage(operator, self.hardware)
        activation_bytes = sum(item.storage_bytes for item in operator.inputs)
        activation_bytes += operator.output.storage_bytes
        total_bytes = activation_bytes + storage.bytes
        working_set = total_bytes
        tier = select_memory_tier(working_set, self.hardware.memory_tiers)
        arithmetic_dtype = (
            operator.weight_dtype if operator.weight_elements else operator.output.dtype
        )
        peak_compute = (
            sparse_capability.effective_peak_ops_per_second
            if sparse_capability is not None
            else self.hardware.peak_compute(arithmetic_dtype)
        )
        # Roofline work is expressed as dense-equivalent useful operations. A sparse
        # capability advertises an effective peak, while executed_flops records the
        # nonzero physical work for diagnostics.
        intensity = dense_flops / total_bytes if total_bytes else float("inf")
        ridge_point = peak_compute / tier.bandwidth_bytes_per_second
        attainable = min(peak_compute, tier.bandwidth_bytes_per_second * intensity)
        compute_time = dense_flops / peak_compute
        memory_time = total_bytes / tier.bandwidth_bytes_per_second + tier.latency_seconds
        bottleneck: Bottleneck = "compute" if compute_time >= memory_time else "memory"
        predicted = max(compute_time, memory_time)
        return OperatorProfile(
            name=operator.name,
            kind=operator.kind.value,
            arithmetic_dtype=arithmetic_dtype.value,
            dense_flops=dense_flops,
            executed_flops=executed_flops,
            activation_bytes=activation_bytes,
            weight_bytes=storage.bytes,
            dense_weight_bytes=storage.dense_bytes,
            sparse_metadata_bytes=storage.metadata_bytes,
            weight_encoding=storage.encoding,
            total_bytes=total_bytes,
            working_set_bytes=working_set,
            memory_tier=tier.name,
            operational_intensity=intensity,
            ridge_point=ridge_point,
            attainable_ops_per_second=attainable,
            compute_time_seconds=compute_time,
            memory_time_seconds=memory_time,
            predicted_latency_seconds=predicted,
            bottleneck=bottleneck,
            sparse_compute_accelerated=sparse_capability is not None,
            sparsity_pattern=operator.attributes.get("sparsity_pattern"),
            compute_backend=sparse_capability.backend if sparse_capability else None,
            compute_performance_source=(
                sparse_capability.performance_source
                if sparse_capability
                else "dense_profile"
            ),
        )

    def profile(self, model: ModelSpec) -> ModelProfile:
        operators = tuple(self.profile_operator(operator) for operator in model.operators)
        dense_flops = sum(operator.dense_flops for operator in operators)
        executed_flops = sum(operator.executed_flops for operator in operators)
        total_bytes = sum(operator.total_bytes for operator in operators)
        weight_bytes = sum(operator.weight_bytes for operator in operators)
        dense_weight_bytes = sum(operator.dense_weight_bytes for operator in operators)
        return ModelProfile(
            model_name=model.name,
            hardware_name=self.hardware.name,
            operators=operators,
            total_dense_flops=dense_flops,
            total_executed_flops=executed_flops,
            total_bytes=total_bytes,
            weight_bytes=weight_bytes,
            dense_weight_bytes=dense_weight_bytes,
            operational_intensity=dense_flops / total_bytes if total_bytes else float("inf"),
            predicted_latency_seconds=sum(
                operator.predicted_latency_seconds for operator in operators
            ),
            bottleneck_counts=dict(Counter(operator.bottleneck for operator in operators)),
            memory_tier_counts=dict(Counter(operator.memory_tier for operator in operators)),
        )


@dataclass(frozen=True)
class BenchmarkResult:
    """Wall-clock latency distribution for a callable."""

    warmup_iterations: int
    measured_iterations: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float
    standard_deviation_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "p95_seconds": self.p95_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "standard_deviation_seconds": self.standard_deviation_seconds,
        }


def benchmark_callable(
    function: Callable[[], Any],
    *,
    warmup: int = 10,
    iterations: int = 100,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], Any] | None = None,
) -> BenchmarkResult:
    """Measure a zero-argument inference callable with warmup and optional sync."""

    if warmup < 0 or iterations <= 0:
        raise ConfigurationError("warmup must be nonnegative and iterations must be positive")
    sync = synchronize or (lambda: None)
    for _ in range(warmup):
        function()
        sync()
    timings: list[float] = []
    for _ in range(iterations):
        sync()
        start = clock()
        function()
        sync()
        elapsed = clock() - start
        if elapsed < 0:
            raise ConfigurationError("benchmark clock moved backwards")
        timings.append(elapsed)
    return BenchmarkResult(
        warmup_iterations=warmup,
        measured_iterations=iterations,
        mean_seconds=statistics.fmean(timings),
        median_seconds=statistics.median(timings),
        p95_seconds=float(np.percentile(timings, 95)),
        min_seconds=min(timings),
        max_seconds=max(timings),
        standard_deviation_seconds=statistics.pstdev(timings),
    )
