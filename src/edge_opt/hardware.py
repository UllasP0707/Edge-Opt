"""Hardware target descriptions for cache-aware roofline analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Mapping

from .core import DType
from .errors import ConfigurationError


BUILTIN_PROFILES = ("arm_cortex_a76",)


def load_builtin_profile(name: str) -> HardwareProfile:
    """Load an analytical reference profile packaged with Edge-Opt."""

    if name not in BUILTIN_PROFILES:
        raise ConfigurationError(
            f"unknown built-in profile {name!r}; choose one of {', '.join(BUILTIN_PROFILES)}"
        )
    resource = files("edge_opt").joinpath("profiles", f"{name}.json")
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
class HardwareProfile:
    """Compute throughput and ordered memory hierarchy for a target device."""

    name: str
    peak_ops_per_second: Mapping[DType, float]
    memory_tiers: tuple[MemoryTier, ...]
    sparse_compute_supported: bool = False
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
        finite_tiers = [tier for tier in self.memory_tiers if tier.capacity_bytes is not None]
        capacities = [tier.capacity_bytes for tier in finite_tiers]
        if capacities != sorted(capacities):
            raise ConfigurationError("memory tiers must be ordered by increasing capacity")
        if self.memory_tiers[-1].capacity_bytes is not None:
            raise ConfigurationError("last memory tier must be unbounded (usually DRAM)")
        object.__setattr__(self, "peak_ops_per_second", normalized_compute)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def peak_compute(self, dtype: DType) -> float:
        """Return target throughput for a data type, failing on unsupported precision."""

        try:
            return self.peak_ops_per_second[dtype]
        except KeyError as exc:
            raise ConfigurationError(
                f"hardware profile {self.name!r} has no throughput for {dtype.value}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "peak_ops_per_second": {
                dtype.value: value for dtype, value in self.peak_ops_per_second.items()
            },
            "memory_tiers": [tier.to_dict() for tier in self.memory_tiers],
            "sparse_compute_supported": self.sparse_compute_supported,
            "sparse_storage_supported": self.sparse_storage_supported,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HardwareProfile:
        return cls(
            name=value["name"],
            peak_ops_per_second={
                DType(dtype): float(throughput)
                for dtype, throughput in value["peak_ops_per_second"].items()
            },
            memory_tiers=tuple(MemoryTier.from_dict(item) for item in value["memory_tiers"]),
            sparse_compute_supported=bool(value.get("sparse_compute_supported", False)),
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
