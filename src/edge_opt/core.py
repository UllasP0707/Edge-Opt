"""Framework-neutral model description used by optimizers and profilers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Iterable, Mapping

from .errors import ConfigurationError


class DType(str, Enum):
    """Storage and arithmetic data types understood by Edge-Opt."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"

    @property
    def bits(self) -> int:
        return {
            DType.FP32: 32,
            DType.FP16: 16,
            DType.BF16: 16,
            DType.INT8: 8,
            DType.UINT8: 8,
            DType.INT4: 4,
        }[self]


class OperatorKind(str, Enum):
    """Operator families with well-defined profiling semantics."""

    CONV2D = "conv2d"
    DEPTHWISE_CONV2D = "depthwise_conv2d"
    LINEAR = "linear"
    MATMUL = "matmul"
    ACTIVATION = "activation"
    POOL = "pool"
    ELEMENTWISE = "elementwise"
    OTHER = "other"


def _validated_shape(shape: Iterable[int], field_name: str) -> tuple[int, ...]:
    normalized = tuple(int(dimension) for dimension in shape)
    if not normalized:
        raise ConfigurationError(f"{field_name} must contain at least one dimension")
    if any(dimension <= 0 for dimension in normalized):
        raise ConfigurationError(f"{field_name} dimensions must all be positive: {normalized}")
    return normalized


@dataclass(frozen=True)
class TensorSpec:
    """Shape and storage description for one tensor."""

    shape: tuple[int, ...]
    dtype: DType = DType.FP32
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _validated_shape(self.shape, "tensor shape"))
        if not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))

    @property
    def numel(self) -> int:
        return reduce(mul, self.shape, 1)

    @property
    def storage_bytes(self) -> int:
        return (self.numel * self.dtype.bits + 7) // 8

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorSpec:
        return cls(
            shape=tuple(value["shape"]),
            dtype=DType(value.get("dtype", DType.FP32.value)),
            name=value.get("name"),
        )


@dataclass(frozen=True)
class OperatorSpec:
    """One model operator and the information needed for static profiling."""

    name: str
    kind: OperatorKind
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    weight_shape: tuple[int, ...] | None = None
    weight_dtype: DType = DType.FP32
    sparsity: float = 0.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("operator name must not be empty")
        if not isinstance(self.kind, OperatorKind):
            object.__setattr__(self, "kind", OperatorKind(self.kind))
        if not self.inputs:
            raise ConfigurationError(f"operator {self.name!r} must have at least one input")
        if self.weight_shape is not None:
            object.__setattr__(
                self, "weight_shape", _validated_shape(self.weight_shape, "weight shape")
            )
        if not isinstance(self.weight_dtype, DType):
            object.__setattr__(self, "weight_dtype", DType(self.weight_dtype))
        if not 0.0 <= self.sparsity < 1.0:
            raise ConfigurationError("operator sparsity must be in [0, 1)")
        object.__setattr__(self, "attributes", dict(self.attributes))

    @property
    def weight_elements(self) -> int:
        if self.weight_shape is None:
            return 0
        return reduce(mul, self.weight_shape, 1)

    @property
    def dense_weight_bytes(self) -> int:
        return (self.weight_elements * self.weight_dtype.bits + 7) // 8

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "inputs": [item.to_dict() for item in self.inputs],
            "output": self.output.to_dict(),
            "weight_shape": list(self.weight_shape) if self.weight_shape else None,
            "weight_dtype": self.weight_dtype.value,
            "sparsity": self.sparsity,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorSpec:
        weight_shape = value.get("weight_shape")
        return cls(
            name=value["name"],
            kind=OperatorKind(value["kind"]),
            inputs=tuple(TensorSpec.from_dict(item) for item in value["inputs"]),
            output=TensorSpec.from_dict(value["output"]),
            weight_shape=tuple(weight_shape) if weight_shape else None,
            weight_dtype=DType(value.get("weight_dtype", DType.FP32.value)),
            sparsity=float(value.get("sparsity", 0.0)),
            attributes=value.get("attributes", {}),
        )


@dataclass(frozen=True)
class ModelSpec:
    """Ordered operator graph used for portable analysis and report export."""

    name: str
    operators: tuple[OperatorSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("model name must not be empty")
        if not self.operators:
            raise ConfigurationError("model must contain at least one operator")
        names = [operator.name for operator in self.operators]
        if len(set(names)) != len(names):
            raise ConfigurationError("operator names must be unique")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dense_weight_bytes(self) -> int:
        return sum(operator.dense_weight_bytes for operator in self.operators)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operators": [operator.to_dict() for operator in self.operators],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelSpec:
        return cls(
            name=value["name"],
            operators=tuple(OperatorSpec.from_dict(item) for item in value["operators"]),
            metadata=value.get("metadata", {}),
        )

