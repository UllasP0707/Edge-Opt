"""Activation-aware SmoothQuant transforms for linear operators.

SmoothQuant migrates per-input-channel activation range into the corresponding
weight columns without changing the floating-point function.  The transformed
pair is suitable for ordinary per-tensor activation and per-output-channel
weight quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .activation import ChannelStatistics
from .errors import ConfigurationError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SmoothQuantConfig:
    """Configuration for the SmoothQuant channel-balancing transform."""

    alpha: float = 0.5
    minimum_magnitude: float = 1e-5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ConfigurationError("SmoothQuant alpha must be in [0, 1]")
        if not np.isfinite(self.minimum_magnitude) or self.minimum_magnitude <= 0:
            raise ConfigurationError(
                "SmoothQuant minimum magnitude must be finite and positive"
            )


@dataclass(frozen=True)
class SmoothQuantResult:
    """Scale vector and transformed weights for one linear operator."""

    scales: FloatArray
    smoothed_weights: FloatArray
    activation_absmax: FloatArray
    weight_absmax: FloatArray
    alpha: float

    def __post_init__(self) -> None:
        scales = np.asarray(self.scales, dtype=np.float64)
        weights = np.asarray(self.smoothed_weights, dtype=np.float64)
        activation_absmax = np.asarray(self.activation_absmax, dtype=np.float64)
        weight_absmax = np.asarray(self.weight_absmax, dtype=np.float64)
        if weights.ndim != 2 or weights.size == 0:
            raise ConfigurationError("SmoothQuant weights must be a nonempty matrix")
        expected_shape = (weights.shape[1],)
        for name, values in (
            ("scales", scales),
            ("activation maxima", activation_absmax),
            ("weight maxima", weight_absmax),
        ):
            if values.shape != expected_shape:
                raise ConfigurationError(
                    f"SmoothQuant {name} must have shape {expected_shape}, got {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ConfigurationError(f"SmoothQuant {name} must be finite")
        if np.any(scales <= 0):
            raise ConfigurationError("SmoothQuant scales must be positive")
        if np.any(activation_absmax < 0) or np.any(weight_absmax < 0):
            raise ConfigurationError("SmoothQuant channel maxima must not be negative")
        if not np.all(np.isfinite(weights)):
            raise ConfigurationError("SmoothQuant weights must be finite")
        if not 0.0 <= self.alpha <= 1.0:
            raise ConfigurationError("SmoothQuant alpha must be in [0, 1]")
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "smoothed_weights", weights)
        object.__setattr__(self, "activation_absmax", activation_absmax)
        object.__setattr__(self, "weight_absmax", weight_absmax)

    @property
    def smoothed_activation_absmax(self) -> FloatArray:
        return np.asarray(self.activation_absmax / self.scales, dtype=np.float64)

    @property
    def smoothed_weight_absmax(self) -> FloatArray:
        return np.asarray(
            np.max(np.abs(self.smoothed_weights), axis=0), dtype=np.float64
        )

    def transform_activations(self, activations: npt.ArrayLike) -> FloatArray:
        """Apply the inverse channel scales to a linear-layer input tensor."""

        values = np.asarray(activations, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != self.scales.size:
            raise ConfigurationError(
                "SmoothQuant activation's final dimension must match the scale vector"
            )
        if not np.all(np.isfinite(values)):
            raise ConfigurationError("SmoothQuant activations must be finite")
        return np.asarray(values / self.scales, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "scales": self.scales.tolist(),
            "activation_absmax": self.activation_absmax.tolist(),
            "weight_absmax": self.weight_absmax.tolist(),
            "smoothed_activation_absmax": self.smoothed_activation_absmax.tolist(),
            "smoothed_weight_absmax": self.smoothed_weight_absmax.tolist(),
        }


def _activation_absmax(
    statistics: ChannelStatistics | npt.ArrayLike,
) -> FloatArray:
    if isinstance(statistics, ChannelStatistics):
        values = statistics.absmax
    else:
        values = np.asarray(statistics, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ConfigurationError(
            "SmoothQuant activation maxima must be a nonempty channel vector"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ConfigurationError(
            "SmoothQuant activation maxima must be finite and nonnegative"
        )
    return values


def smoothquant_scales(
    weights: npt.ArrayLike,
    activation_statistics: ChannelStatistics | npt.ArrayLike,
    config: SmoothQuantConfig | None = None,
) -> FloatArray:
    """Compute ``act_max**alpha / weight_max**(1-alpha)`` per input channel."""

    settings = config or SmoothQuantConfig()
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ConfigurationError("SmoothQuant weights must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ConfigurationError("SmoothQuant weights must be finite")
    activation_absmax = _activation_absmax(activation_statistics)
    if activation_absmax.shape != (matrix.shape[1],):
        raise ConfigurationError(
            "SmoothQuant activation channels must match the linear input dimension"
        )
    weight_absmax = np.max(np.abs(matrix), axis=0)
    safe_activation = np.maximum(activation_absmax, settings.minimum_magnitude)
    safe_weight = np.maximum(weight_absmax, settings.minimum_magnitude)
    scales = np.power(safe_activation, settings.alpha) / np.power(
        safe_weight, 1.0 - settings.alpha
    )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ConfigurationError("SmoothQuant produced invalid channel scales")
    return np.asarray(scales, dtype=np.float64)


def apply_smoothquant(
    weights: npt.ArrayLike,
    activation_statistics: ChannelStatistics | npt.ArrayLike,
    config: SmoothQuantConfig | None = None,
) -> SmoothQuantResult:
    """Scale linear weight columns using representative activation maxima."""

    settings = config or SmoothQuantConfig()
    matrix = np.asarray(weights, dtype=np.float64)
    activation_absmax = _activation_absmax(activation_statistics)
    scales = smoothquant_scales(matrix, activation_absmax, settings)
    return SmoothQuantResult(
        scales=scales,
        smoothed_weights=matrix * scales[np.newaxis, :],
        activation_absmax=activation_absmax,
        weight_absmax=np.max(np.abs(matrix), axis=0),
        alpha=settings.alpha,
    )
