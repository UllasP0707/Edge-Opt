"""Hardware target descriptions for cache-aware roofline analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from .core import DType, OperatorKind, OperatorSpec
from .errors import ConfigurationError
from .structured import NMPruningPattern

BUILTIN_PROFILES = ("arm_cortex_a76", "nvidia_a100_reference")


def load_builtin_profile(name: str) -> HardwareProfile:
    """Load an analytical reference profile packaged with Edge-Opt."""

    if name not in BUILTIN_PROFILES:
        raise ConfigurationError(
            f"unknown built-in profile {name!r}; choose one of {', '.join(BUILTIN_PROFILES)}"
        )
    resource = files("edge_opt").joinpath("profiles").joinpath(f"{name}.json")
    with as_file(resource) as path:
        return HardwareProfile.from_json(path)


@dataclass(frozen=True)
class MemoryTier:
    """One level in a target's memory hierarchy."""

    name: str
    capacity_bytes: int | None
    bandwidth_bytes_per_second: float
    latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("memory tier name must not be empty")
        if self.capacity_bytes is not None and self.capacity_bytes <= 0:
            raise ConfigurationError("memory capacity must be positive or null")
        if self.bandwidth_bytes_per_second <= 0:
            raise ConfigurationError("memory bandwidth must be positive")
        if self.latency_seconds < 0:
            raise ConfigurationError("memory latency must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capacity_bytes": self.capacity_bytes,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "latency_seconds": self.latency_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryTier:
        return cls(
            name=value["name"],
            capacity_bytes=value.get("capacity_bytes"),
            bandwidth_bytes_per_second=float(value["bandwidth_bytes_per_second"]),
            latency_seconds=float(value.get("latency_seconds", 0.0)),
        )


@dataclass(frozen=True)
class SparseComputeCapability:
    """One target kernel that accelerates an exact sparsity pattern."""

    operator_kind: OperatorKind
    weight_dtype: DType
    pattern: str
    effective_peak_ops_per_second: float
    backend: str
    performance_source: str = "vendor_spec"

    def __post_init__(self) -> None:
        if not isinstance(self.operator_kind, OperatorKind):
            object.__setattr__(self, "operator_kind", OperatorKind(self.operator_kind))
        if not isinstance(self.weight_dtype, DType):
            object.__setattr__(self, "weight_dtype", DType(self.weight_dtype))
        self.parsed_pattern()
        if self.effective_peak_ops_per_second <= 0:
            raise ConfigurationError("sparse effective peak compute must be positive")
        if not self.backend.strip():
            raise ConfigurationError("sparse compute capability backend must not be empty")
        if self.performance_source not in {"measured", "vendor_spec", "analytical"}:
            raise ConfigurationError(
                "sparse performance_source must be measured, vendor_spec, or analytical"
            )

    def parsed_pattern(self) -> NMPruningPattern:
        try:
            n_text, m_text = self.pattern.split(":", maxsplit=1)
            return NMPruningPattern(int(n_text), int(m_text))
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(
                f"invalid sparse compute pattern {self.pattern!r}; expected N:M"
            ) from exc

    def matches(self, operator: OperatorSpec) -> bool:
        if (
            operator.kind is not self.operator_kind
            or operator.weight_dtype is not self.weight_dtype
        ):
            return False
        if operator.attributes.get("sparsity_pattern") != self.pattern:
            return False
        return math.isclose(
            operator.sparsity,
            self.parsed_pattern().sparsity,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_kind": self.operator_kind.value,
            "weight_dtype": self.weight_dtype.value,
            "pattern": self.pattern,
            "effective_peak_ops_per_second": self.effective_peak_ops_per_second,
            "backend": self.backend,
            "performance_source": self.performance_source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SparseComputeCapability:
        return cls(
            operator_kind=OperatorKind(value["operator_kind"]),
            weight_dtype=DType(value["weight_dtype"]),
            pattern=str(value["pattern"]),
            effective_peak_ops_per_second=float(value["effective_peak_ops_per_second"]),
            backend=str(value["backend"]),
            performance_source=str(value.get("performance_source", "vendor_spec")),
        )


@dataclass(frozen=True)
class HardwareProfile:
    """Compute throughput and ordered memory hierarchy for a target device."""

    name: str
    peak_ops_per_second: Mapping[DType, float]
    memory_tiers: tuple[MemoryTier, ...]
    sparse_compute_capabilities: tuple[SparseComputeCapability, ...] = ()
    sparse_storage_supported: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("hardware profile name must not be empty")
        normalized_compute = {
            DType(dtype): float(value) for dtype, value in self.peak_ops_per_second.items()
        }
        if not normalized_compute or any(value <= 0 for value in normalized_compute.values()):
            raise ConfigurationError("peak compute values must be positive")
        if not self.memory_tiers:
            raise ConfigurationError("hardware profile needs at least one memory tier")
        capacities = [
            int(tier.capacity_bytes)
            for tier in self.memory_tiers
            if tier.capacity_bytes is not None
        ]
        if capacities != sorted(capacities):
            raise ConfigurationError("memory tiers must be ordered by increasing capacity")
        if self.memory_tiers[-1].capacity_bytes is not None:
            raise ConfigurationError("last memory tier must be unbounded (usually DRAM)")
        object.__setattr__(self, "peak_ops_per_second", normalized_compute)
        object.__setattr__(
            self, "sparse_compute_capabilities", tuple(self.sparse_compute_capabilities)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def peak_compute(self, dtype: DType) -> float:
        """Return target throughput for a data type, failing on unsupported precision."""

        try:
            return self.peak_ops_per_second[dtype]
        except KeyError as exc:
            raise ConfigurationError(
                f"hardware profile {self.name!r} has no throughput for {dtype.value}"
            ) from exc

    def sparse_capability(self, operator: OperatorSpec) -> SparseComputeCapability | None:
        """Return an exact matching sparse kernel, never a blanket sparsity assumption."""

        matches = [
            capability
            for capability in self.sparse_compute_capabilities
            if capability.matches(operator)
        ]
        if len(matches) > 1:
            raise ConfigurationError(
                f"hardware profile {self.name!r} has duplicate sparse capabilities for "
                f"operator {operator.name!r}"
            )
        return matches[0] if matches else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "peak_ops_per_second": {
                dtype.value: value for dtype, value in self.peak_ops_per_second.items()
            },
            "memory_tiers": [tier.to_dict() for tier in self.memory_tiers],
            "sparse_compute_capabilities": [
                capability.to_dict() for capability in self.sparse_compute_capabilities
            ],
            "sparse_storage_supported": self.sparse_storage_supported,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HardwareProfile:
        if value.get("sparse_compute_supported"):
            raise ConfigurationError(
                "sparse_compute_supported is unsafe and no longer accepted; declare exact "
                "sparse_compute_capabilities instead"
            )
        return cls(
            name=value["name"],
            peak_ops_per_second={
                DType(dtype): float(throughput)
                for dtype, throughput in value["peak_ops_per_second"].items()
            },
            memory_tiers=tuple(MemoryTier.from_dict(item) for item in value["memory_tiers"]),
            sparse_compute_capabilities=tuple(
                SparseComputeCapability.from_dict(item)
                for item in value.get("sparse_compute_capabilities", ())
            ),
            sparse_storage_supported=bool(value.get("sparse_storage_supported", True)),
            metadata=value.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> HardwareProfile:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
