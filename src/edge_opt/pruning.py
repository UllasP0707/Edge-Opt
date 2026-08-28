"""Scheduled magnitude pruning with deterministic, monotonic masks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .errors import ConfigurationError

Array = npt.NDArray[np.generic]


@dataclass(frozen=True)
class PolynomialPruningSchedule:
    """Cubic sparsity schedule used by magnitude-based pruning.

    Between ``begin_step`` and ``end_step`` this evaluates

    ``final + (initial - final) * (1 - progress) ** power``.

    Masks are only meant to be refreshed on ``update_frequency`` boundaries,
    while ``sparsity_at`` itself is continuous and useful for diagnostics.
    """

    initial_sparsity: float = 0.0
    final_sparsity: float = 0.75
    begin_step: int = 0
    end_step: int = 10_000
    update_frequency: int = 100
    power: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial_sparsity <= self.final_sparsity < 1.0:
            raise ConfigurationError(
                "sparsities must satisfy 0 <= initial_sparsity <= final_sparsity < 1"
            )
        if self.begin_step < 0 or self.end_step <= self.begin_step:
            raise ConfigurationError("end_step must be greater than nonnegative begin_step")
        if self.update_frequency <= 0:
            raise ConfigurationError("update_frequency must be positive")
        if self.power <= 0:
            raise ConfigurationError("polynomial schedule power must be positive")

    @property
    def pruning_updates(self) -> int:
        duration = self.end_step - self.begin_step
        return (duration + self.update_frequency - 1) // self.update_frequency

    def sparsity_at(self, step: int) -> float:
        if step <= self.begin_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.final_sparsity
        progress = (step - self.begin_step) / (self.end_step - self.begin_step)
        return float(
            self.final_sparsity
            + (self.initial_sparsity - self.final_sparsity)
            * (1.0 - progress) ** self.power
        )

    def is_update_step(self, step: int) -> bool:
        if step < self.begin_step or step > self.end_step:
            return False
        return (step - self.begin_step) % self.update_frequency == 0 or step == self.end_step


def measured_sparsity(weights: Mapping[str, npt.ArrayLike]) -> float:
    """Return the exact fraction of zero-valued parameters."""

    total = 0
    zeros = 0
    for values in weights.values():
        array = np.asarray(values)
        total += array.size
        zeros += int(np.count_nonzero(array == 0))
    return float(zeros / total) if total else 0.0


@dataclass(frozen=True)
class PruningStepResult:
    """Masks and statistics produced at one pruning update."""

    step: int
    target_sparsity: float
    actual_sparsity: float
    pruned_parameters: int
    total_parameters: int
    masks: Mapping[str, npt.NDArray[np.bool_]]


@dataclass(frozen=True)
class SparseStorageEstimate:
    """Storage comparison for a coordinate-style sparse encoding."""

    dense_bytes: int
    sparse_bytes: int
    nonzero_values: int
    index_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.dense_bytes / self.sparse_bytes if self.sparse_bytes else float("inf")


def estimate_sparse_storage(
    num_elements: int,
    sparsity: float,
    *,
    value_bits: int = 8,
    index_bits: int = 16,
    block_size: int = 1,
) -> SparseStorageEstimate:
    """Estimate block-coordinate sparse storage including index overhead."""

    if num_elements < 0:
        raise ConfigurationError("num_elements must not be negative")
    if not 0.0 <= sparsity < 1.0:
        raise ConfigurationError("sparsity must be in [0, 1)")
    if value_bits <= 0 or index_bits < 0 or block_size <= 0:
        raise ConfigurationError("bit widths and block size must be valid")
    nonzero = int(np.ceil(num_elements * (1.0 - sparsity)))
    stored_blocks = int(np.ceil(nonzero / block_size))
    value_bytes = (nonzero * value_bits + 7) // 8
    index_bytes = (stored_blocks * index_bits + 7) // 8
    dense_bytes = (num_elements * value_bits + 7) // 8
    return SparseStorageEstimate(dense_bytes, value_bytes + index_bytes, nonzero, index_bytes)


class MagnitudePruner:
    """Unstructured magnitude pruner for framework-neutral weight mappings.

    With ``global_pruning=True`` one threshold is selected across every tensor.
    Stable sorting and tensor-name ordering make tied magnitudes deterministic.
    By default masks are monotonic, so a parameter cannot regrow after pruning.
    """

    def __init__(
        self,
        schedule: PolynomialPruningSchedule | None = None,
        *,
        global_pruning: bool = True,
        allow_regrowth: bool = False,
    ) -> None:
        self.schedule = schedule or PolynomialPruningSchedule()
        self.global_pruning = global_pruning
        self.allow_regrowth = allow_regrowth
        self.masks: dict[str, npt.NDArray[np.bool_]] = {}
        self.last_step: int | None = None

    @staticmethod
    def _validate_weights(weights: Mapping[str, npt.ArrayLike]) -> dict[str, Array]:
        if not weights:
            raise ConfigurationError("at least one weight tensor is required")
        arrays: dict[str, Array] = {}
        for name in sorted(weights):
            if not name:
                raise ConfigurationError("weight names must not be empty")
            array = np.asarray(weights[name])
            if array.size == 0:
                raise ConfigurationError(f"weight tensor {name!r} is empty")
            if not np.issubdtype(array.dtype, np.number):
                raise ConfigurationError(f"weight tensor {name!r} must be numeric")
            arrays[name] = array
        return arrays

    @staticmethod
    def _prune_lowest(
        arrays: Mapping[str, Array],
        existing: Mapping[str, npt.NDArray[np.bool_]],
        prune_count: int,
    ) -> dict[str, npt.NDArray[np.bool_]]:
        masks = {name: mask.copy() for name, mask in existing.items()}
        candidates: list[tuple[float, str, int]] = []
        for name, array in arrays.items():
            flat_values = np.abs(array.astype(np.float64, copy=False).ravel())
            flat_mask = masks[name].ravel()
            candidates.extend(
                (float(flat_values[index]), name, int(index))
                for index in np.flatnonzero(flat_mask)
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, name, flat_index in candidates[:prune_count]:
            masks[name].ravel()[flat_index] = False
        return masks

    def _initial_masks(self, arrays: Mapping[str, Array]) -> dict[str, npt.NDArray[np.bool_]]:
        masks: dict[str, npt.NDArray[np.bool_]] = {}
        for name, array in arrays.items():
            prior = self.masks.get(name)
            if prior is not None and prior.shape != array.shape:
                raise ConfigurationError(
                    f"weight tensor {name!r} changed shape after pruning began"
                )
            masks[name] = (
                np.ones_like(array, dtype=np.bool_)
                if prior is None or self.allow_regrowth
                else prior.copy()
            )
        if set(self.masks) - set(arrays):
            raise ConfigurationError("weight tensors were removed after pruning began")
        return masks

    def _global_masks(
        self, arrays: Mapping[str, Array], target_sparsity: float
    ) -> dict[str, npt.NDArray[np.bool_]]:
        masks = self._initial_masks(arrays)
        total = sum(array.size for array in arrays.values())
        target_pruned = int(round(total * target_sparsity))
        already_pruned = sum(mask.size - int(np.count_nonzero(mask)) for mask in masks.values())
        return self._prune_lowest(arrays, masks, max(0, target_pruned - already_pruned))

    def _local_masks(
        self, arrays: Mapping[str, Array], target_sparsity: float
    ) -> dict[str, npt.NDArray[np.bool_]]:
        initial = self._initial_masks(arrays)
        result: dict[str, npt.NDArray[np.bool_]] = {}
        for name, array in arrays.items():
            target_pruned = int(round(array.size * target_sparsity))
            already_pruned = array.size - int(np.count_nonzero(initial[name]))
            result.update(
                self._prune_lowest(
                    {name: array},
                    {name: initial[name]},
                    max(0, target_pruned - already_pruned),
                )
            )
        return result

    def update_masks(
        self, weights: Mapping[str, npt.ArrayLike], step: int
    ) -> PruningStepResult:
        """Refresh masks for a scheduled step without modifying caller-owned arrays."""

        if step < 0:
            raise ConfigurationError("training step must not be negative")
        if self.last_step is not None and step < self.last_step:
            raise ConfigurationError("pruning steps must be monotonically increasing")
        arrays = self._validate_weights(weights)
        target = self.schedule.sparsity_at(step)
        if self.global_pruning:
            masks = self._global_masks(arrays, target)
        else:
            masks = self._local_masks(arrays, target)
        self.masks = masks
        self.last_step = step
        total = sum(mask.size for mask in masks.values())
        retained = sum(int(np.count_nonzero(mask)) for mask in masks.values())
        pruned = total - retained
        return PruningStepResult(
            step=step,
            target_sparsity=target,
            actual_sparsity=pruned / total,
            pruned_parameters=pruned,
            total_parameters=total,
            masks={name: mask.copy() for name, mask in masks.items()},
        )

    def maybe_update(
        self, weights: Mapping[str, npt.ArrayLike], step: int
    ) -> PruningStepResult | None:
        """Refresh masks only when the schedule says an update is due."""

        if not self.schedule.is_update_step(step):
            return None
        return self.update_masks(weights, step)

    def apply_masks(
        self,
        weights: Mapping[str, npt.ArrayLike],
        *,
        inplace: bool = False,
    ) -> dict[str, Array]:
        """Apply current masks, optionally updating writable NumPy arrays in place."""

        arrays = self._validate_weights(weights)
        if set(arrays) != set(self.masks):
            raise ConfigurationError("masks must be initialized for every weight tensor")
        output: dict[str, Array] = {}
        for name, array in arrays.items():
            destination = array if inplace else array.copy()
            destination[~self.masks[name]] = 0
            output[name] = destination
        return output

    def step(
        self,
        weights: Mapping[str, npt.ArrayLike],
        step: int,
        *,
        inplace: bool = False,
    ) -> tuple[dict[str, Array], PruningStepResult]:
        """Update masks and apply them in one operation."""

        result = self.update_masks(weights, step)
        return self.apply_masks(weights, inplace=inplace), result

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_step": self.last_step,
            "masks": {
                name: {"shape": list(mask.shape), "values": mask.astype(np.uint8).ravel().tolist()}
                for name, mask in self.masks.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        masks: dict[str, npt.NDArray[np.bool_]] = {}
        for name, encoded in state.get("masks", {}).items():
            shape = tuple(int(value) for value in encoded["shape"])
            values = np.asarray(encoded["values"], dtype=np.bool_)
            if values.size != int(np.prod(shape)):
                raise ConfigurationError(f"serialized mask {name!r} has an invalid shape")
            masks[name] = values.reshape(shape)
        self.masks = masks
        last_step = state.get("last_step")
        self.last_step = int(last_step) if last_step is not None else None
