"""Wanda activation-aware, per-output pruning for linear weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .activation import ChannelStatistics
from .errors import ConfigurationError
from .structured import NMPruningPattern, nm_mask

FloatArray = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]


def _activation_norms(values: ChannelStatistics | npt.ArrayLike) -> FloatArray:
    norms = values.l2_norm if isinstance(values, ChannelStatistics) else values
    array = np.asarray(norms, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ConfigurationError("Wanda activation norms must be a nonempty vector")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ConfigurationError("Wanda activation norms must be finite and nonnegative")
    return array


def wanda_scores(
    weights: npt.ArrayLike,
    input_statistics: ChannelStatistics | npt.ArrayLike,
) -> FloatArray:
    """Return the Wanda metric ``abs(W_ij) * ||X_j||_2``.

    The implementation follows the paper's linear-layer formulation: weights
    must have shape ``[output_features, input_features]`` and input activation
    statistics must have one L2 norm per input feature.
    """

    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ConfigurationError("Wanda currently supports nonempty 2-D linear weights")
    if not np.all(np.isfinite(array)):
        raise ConfigurationError("Wanda weights must be finite")
    norms = _activation_norms(input_statistics)
    if norms.size != array.shape[1]:
        raise ConfigurationError(
            f"Wanda received {norms.size} activation channels for "
            f"{array.shape[1]} input features"
        )
    return np.asarray(np.abs(array) * norms[np.newaxis, :], dtype=np.float64)


def wanda_mask(
    weights: npt.ArrayLike,
    input_statistics: ChannelStatistics | npt.ArrayLike,
    sparsity: float,
    pattern: NMPruningPattern | None = None,
) -> Mask:
    """Select the lowest Wanda scores independently in every output row."""

    if not 0.0 <= sparsity < 1.0:
        raise ConfigurationError("Wanda sparsity must be in [0, 1)")
    scores = wanda_scores(weights, input_statistics)
    if pattern is not None:
        if pattern.axis % scores.ndim != scores.ndim - 1:
            raise ConfigurationError("Wanda N:M grouping must use the input-feature axis")
        if not np.isclose(sparsity, pattern.sparsity):
            raise ConfigurationError(
                f"Wanda sparsity {sparsity} does not match {pattern.label} "
                f"sparsity {pattern.sparsity}"
            )
        return nm_mask(scores, pattern)
    pruned_per_row = int(scores.shape[1] * sparsity)
    mask = np.ones(scores.shape, dtype=np.bool_)
    if pruned_per_row == 0:
        return mask
    for row in range(scores.shape[0]):
        # Stable sorting makes equal-score selection deterministic by column.
        indices = np.argsort(scores[row], kind="stable")[:pruned_per_row]
        mask[row, indices] = False
    return mask


@dataclass(frozen=True)
class WandaPruningResult:
    """Masks and exact sparsity measurements from one-shot Wanda pruning."""

    target_sparsity: float
    actual_sparsity: float
    pruned_parameters: int
    total_parameters: int
    masks: Mapping[str, Mask]
    layer_sparsity: Mapping[str, float]


class WandaPruner:
    """One-shot Wanda pruning over named linear weight tensors."""

    def __init__(
        self,
        sparsity: float = 0.5,
        *,
        pattern: NMPruningPattern | None = None,
    ) -> None:
        if not 0.0 <= sparsity < 1.0:
            raise ConfigurationError("Wanda sparsity must be in [0, 1)")
        self.sparsity = sparsity
        self.pattern = pattern
        if pattern is not None and not np.isclose(sparsity, pattern.sparsity):
            raise ConfigurationError(
                f"Wanda sparsity {sparsity} does not match {pattern.label} "
                f"sparsity {pattern.sparsity}"
            )

    def compute_masks(
        self,
        weights: Mapping[str, npt.ArrayLike],
        activation_statistics: Mapping[str, ChannelStatistics | npt.ArrayLike],
    ) -> WandaPruningResult:
        if not weights:
            raise ConfigurationError("Wanda requires at least one weight tensor")
        missing = set(weights) - set(activation_statistics)
        if missing:
            raise ConfigurationError(
                "missing Wanda activation statistics for: " + ", ".join(sorted(missing))
            )
        masks: dict[str, Mask] = {}
        layer_sparsity: dict[str, float] = {}
        total = 0
        pruned = 0
        for name in sorted(weights):
            if not name:
                raise ConfigurationError("Wanda weight names must not be empty")
            mask = wanda_mask(
                weights[name],
                activation_statistics[name],
                self.sparsity,
                self.pattern,
            )
            layer_pruned = mask.size - int(np.count_nonzero(mask))
            masks[name] = mask
            layer_sparsity[name] = layer_pruned / mask.size
            total += mask.size
            pruned += layer_pruned
        return WandaPruningResult(
            target_sparsity=self.sparsity,
            actual_sparsity=pruned / total,
            pruned_parameters=pruned,
            total_parameters=total,
            masks=masks,
            layer_sparsity=layer_sparsity,
        )

    def prune(
        self,
        weights: Mapping[str, npt.ArrayLike],
        activation_statistics: Mapping[str, ChannelStatistics | npt.ArrayLike],
        *,
        inplace: bool = False,
    ) -> tuple[dict[str, npt.NDArray[np.generic]], WandaPruningResult]:
        """Compute Wanda masks and apply them to caller-owned or copied arrays."""

        result = self.compute_masks(weights, activation_statistics)
        pruned: dict[str, npt.NDArray[np.generic]] = {}
        for name, values in weights.items():
            source = np.asarray(values)
            destination = source if inplace else source.copy()
            destination *= result.masks[name]
            pruned[name] = destination
        return pruned, result
