"""Hardware-oriented N:M semi-structured sparsity masks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .errors import ConfigurationError

Mask = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class NMPruningPattern:
    """Retain exactly ``n`` values in every contiguous group of ``m``."""

    n: int
    m: int
    axis: int = -1

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 1 or self.n >= self.m:
            raise ConfigurationError("N:M sparsity requires 0 < N < M")

    @property
    def sparsity(self) -> float:
        return 1.0 - self.n / self.m

    @property
    def label(self) -> str:
        return f"{self.n}:{self.m}"

    @property
    def minimum_metadata_bits_per_group(self) -> int:
        """Information-theoretic minimum bits needed to identify retained positions."""

        return math.ceil(math.log2(math.comb(self.m, self.n)))

    def to_dict(self) -> dict[str, int]:
        return {"n": self.n, "m": self.m, "axis": self.axis}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NMPruningPattern:
        return cls(n=int(value["n"]), m=int(value["m"]), axis=int(value.get("axis", -1)))


def _grouped(values: npt.ArrayLike, pattern: NMPruningPattern) -> tuple[np.ndarray, int]:
    array = np.asarray(values)
    if array.ndim == 0 or array.size == 0:
        raise ConfigurationError("N:M pruning requires a nonempty tensor with rank >= 1")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ConfigurationError("N:M pruning values must be numeric")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ConfigurationError("N:M pruning values must be finite")
    axis = pattern.axis % array.ndim
    axis_size = array.shape[axis]
    if axis_size % pattern.m:
        raise ConfigurationError(
            f"N:M axis length {axis_size} is not divisible by group size {pattern.m}"
        )
    moved = np.moveaxis(array, axis, -1)
    return moved.reshape(*moved.shape[:-1], axis_size // pattern.m, pattern.m), axis


def nm_mask(scores: npt.ArrayLike, pattern: NMPruningPattern) -> Mask:
    """Retain the ``n`` largest scores in every contiguous group of ``m``."""

    array = np.asarray(scores, dtype=np.float64)
    grouped, axis = _grouped(array, pattern)
    grouped_mask = np.zeros(grouped.shape, dtype=np.bool_)
    # Stable ascending order deterministically prunes lower column indices on ties.
    retained = np.argsort(grouped, axis=-1, kind="stable")[..., -pattern.n :]
    np.put_along_axis(grouped_mask, retained, True, axis=-1)
    moved_mask = grouped_mask.reshape(np.moveaxis(array, axis, -1).shape)
    return np.moveaxis(moved_mask, -1, axis)


def validate_nm_mask(mask: npt.ArrayLike, pattern: NMPruningPattern) -> bool:
    """Return whether every complete group retains exactly ``n`` positions."""

    array = np.asarray(mask, dtype=np.bool_)
    grouped, _ = _grouped(array, pattern)
    return bool(np.all(np.count_nonzero(grouped, axis=-1) == pattern.n))


@dataclass(frozen=True)
class NMPruningResult:
    """Exact masks and measurements from N:M pruning."""

    pattern: NMPruningPattern
    actual_sparsity: float
    pruned_parameters: int
    total_parameters: int
    masks: Mapping[str, Mask]


class NMPruner:
    """Apply magnitude- or externally-scored N:M pruning to named tensors."""

    def __init__(self, pattern: NMPruningPattern) -> None:
        self.pattern = pattern

    def compute_masks(
        self,
        weights: Mapping[str, npt.ArrayLike],
        *,
        scores: Mapping[str, npt.ArrayLike] | None = None,
    ) -> NMPruningResult:
        if not weights:
            raise ConfigurationError("N:M pruning requires at least one weight tensor")
        if scores is not None:
            missing = set(weights) - set(scores)
            if missing:
                raise ConfigurationError(
                    "missing N:M scores for: " + ", ".join(sorted(missing))
                )
        masks: dict[str, Mask] = {}
        total = 0
        pruned = 0
        for name in sorted(weights):
            weight = np.asarray(weights[name])
            metric = np.abs(weight) if scores is None else np.asarray(scores[name])
            if metric.shape != weight.shape:
                raise ConfigurationError(f"N:M score shape does not match weight {name!r}")
            mask = nm_mask(metric, self.pattern)
            masks[name] = mask
            total += mask.size
            pruned += mask.size - int(np.count_nonzero(mask))
        return NMPruningResult(
            pattern=self.pattern,
            actual_sparsity=pruned / total,
            pruned_parameters=pruned,
            total_parameters=total,
            masks=masks,
        )

    def prune(
        self,
        weights: Mapping[str, npt.ArrayLike],
        *,
        scores: Mapping[str, npt.ArrayLike] | None = None,
        inplace: bool = False,
    ) -> tuple[dict[str, npt.NDArray[np.generic]], NMPruningResult]:
        result = self.compute_masks(weights, scores=scores)
        output: dict[str, npt.NDArray[np.generic]] = {}
        for name, values in weights.items():
            source = np.asarray(values)
            destination = source if inplace else source.copy()
            destination *= result.masks[name]
            output[name] = destination
        return output, result
