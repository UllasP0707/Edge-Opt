"""INT8 quantization, fake-quantization, and representative calibration."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .errors import ConfigurationError

Array = npt.NDArray[np.generic]


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization scheme shared by observers and fake-quantizers."""

    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    channel_axis: int = 0
    narrow_range: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 16:
            raise ConfigurationError("quantization bit width must be between 2 and 16")
        if self.per_channel and not self.symmetric:
            raise ConfigurationError("per-channel affine quantization is not supported")

    @property
    def qmin(self) -> int:
        if self.symmetric:
            return -self.qmax
        return 0

    @property
    def qmax(self) -> int:
        if self.symmetric:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1 - int(self.narrow_range)


@dataclass(frozen=True)
class QuantizationParams:
    """Scale and zero-point needed for quantize/dequantize operations."""

    scale: Array
    zero_point: Array
    qmin: int
    qmax: int
    axis: int | None = None

    def __post_init__(self) -> None:
        scale = np.asarray(self.scale, dtype=np.float64)
        zero_point = np.asarray(self.zero_point, dtype=np.int64)
        if scale.size == 0 or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ConfigurationError("quantization scale must be finite and positive")
        if scale.shape != zero_point.shape:
            raise ConfigurationError("scale and zero-point shapes must match")
        if self.qmin >= self.qmax:
            raise ConfigurationError("qmin must be less than qmax")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "zero_point", zero_point)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.tolist(),
            "zero_point": self.zero_point.tolist(),
            "qmin": self.qmin,
            "qmax": self.qmax,
            "axis": self.axis,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QuantizationParams:
        return cls(
            scale=np.asarray(value["scale"], dtype=np.float64),
            zero_point=np.asarray(value["zero_point"], dtype=np.int64),
            qmin=int(value["qmin"]),
            qmax=int(value["qmax"]),
            axis=value.get("axis"),
        )


def _quantization_params(
    minimum: Array | float,
    maximum: Array | float,
    config: QuantizationConfig,
) -> QuantizationParams:
    lower = np.asarray(minimum, dtype=np.float64)
    upper = np.asarray(maximum, dtype=np.float64)
    if lower.shape != upper.shape:
        raise ConfigurationError("observed minimum and maximum shapes must match")
    if np.any(lower > upper):
        raise ConfigurationError("observed minimum must not exceed maximum")

    epsilon = np.finfo(np.float32).eps
    if config.symmetric:
        absolute_max = np.maximum(np.abs(lower), np.abs(upper))
        scale = np.maximum(absolute_max / config.qmax, epsilon)
        zero_point = np.zeros_like(scale, dtype=np.int64)
    else:
        scale = np.maximum((upper - lower) / (config.qmax - config.qmin), epsilon)
        zero_point = np.clip(
            np.rint(config.qmin - lower / scale), config.qmin, config.qmax
        ).astype(np.int64)
    return QuantizationParams(
        scale=scale,
        zero_point=zero_point,
        qmin=config.qmin,
        qmax=config.qmax,
        axis=config.channel_axis if config.per_channel else None,
    )


def _broadcast_params(values: Array, params: QuantizationParams) -> tuple[Array, Array]:
    if params.axis is None:
        return params.scale, params.zero_point
    axis = params.axis % values.ndim
    if params.scale.ndim != 1 or params.scale.shape[0] != values.shape[axis]:
        raise ConfigurationError(
            f"per-channel parameters have shape {params.scale.shape}, expected "
            f"({values.shape[axis]},) for axis {axis}"
        )
    shape = [1] * values.ndim
    shape[axis] = params.scale.shape[0]
    return params.scale.reshape(shape), params.zero_point.reshape(shape)


def quantize(values: npt.ArrayLike, params: QuantizationParams) -> npt.NDArray[np.int64]:
    """Quantize floating-point values to integer codes."""

    array = np.asarray(values)
    scale, zero_point = _broadcast_params(array, params)
    return np.clip(np.rint(array / scale) + zero_point, params.qmin, params.qmax).astype(
        np.int64
    )


def dequantize(values: npt.ArrayLike, params: QuantizationParams) -> npt.NDArray[np.float64]:
    """Convert integer codes back to floating-point values."""

    array = np.asarray(values)
    scale, zero_point = _broadcast_params(array, params)
    return (array.astype(np.float64) - zero_point) * scale


def fake_quantize(values: npt.ArrayLike, params: QuantizationParams) -> Array:
    """Simulate quantization rounding while retaining a floating-point dtype."""

    source = np.asarray(values)
    restored = dequantize(quantize(source, params), params)
    return restored.astype(source.dtype if np.issubdtype(source.dtype, np.floating) else np.float32)


def quantization_mse(values: npt.ArrayLike, params: QuantizationParams) -> float:
    """Mean squared quantization error for diagnostics."""

    source = np.asarray(values, dtype=np.float64)
    error = source - fake_quantize(source, params)
    return float(np.mean(np.square(error)))


class Observer(Protocol):
    """Protocol implemented by calibration observers."""

    def update(self, values: npt.ArrayLike) -> None: ...

    def calculate_qparams(self) -> QuantizationParams: ...


class MinMaxObserver:
    """Streaming min/max observer with optional per-channel statistics."""

    def __init__(self, config: QuantizationConfig | None = None) -> None:
        self.config = config or QuantizationConfig()
        self.minimum: Array | None = None
        self.maximum: Array | None = None

    def update(self, values: npt.ArrayLike) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return
        finite = np.where(np.isfinite(array), array, np.nan)
        if np.all(np.isnan(finite)):
            return
        if self.config.per_channel:
            axis = self.config.channel_axis % array.ndim
            reduce_axes = tuple(index for index in range(array.ndim) if index != axis)
            current_min = np.nanmin(finite, axis=reduce_axes)
            current_max = np.nanmax(finite, axis=reduce_axes)
        else:
            current_min = np.asarray(np.nanmin(finite))
            current_max = np.asarray(np.nanmax(finite))
        if self.minimum is not None and current_min.shape != self.minimum.shape:
            raise ConfigurationError("observer channel count changed between batches")
        self.minimum = current_min if self.minimum is None else np.minimum(self.minimum, current_min)
        self.maximum = current_max if self.maximum is None else np.maximum(self.maximum, current_max)

    def calculate_qparams(self) -> QuantizationParams:
        if self.minimum is None or self.maximum is None:
            raise ConfigurationError("observer has not received any finite samples")
        return _quantization_params(self.minimum, self.maximum, self.config)


def _kl_divergence(reference: Array, candidate: Array) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mask = reference > 0
    if np.any(candidate[mask] <= 0):
        return float("inf")
    return float(np.sum(reference[mask] * np.log(reference[mask] / candidate[mask])))


class EntropyObserver:
    """Streaming histogram observer using KL divergence to select a clipping range.

    Histograms track absolute activation magnitudes. When the observed range grows,
    previous counts are rebinned rather than discarded. The selected threshold is
    the distribution boundary whose quantized approximation minimizes KL divergence.
    """

    def __init__(
        self,
        config: QuantizationConfig | None = None,
        *,
        histogram_bins: int = 2048,
        quantized_bins: int | None = None,
    ) -> None:
        self.config = config or QuantizationConfig(symmetric=True, per_channel=False)
        if self.config.per_channel:
            raise ConfigurationError("entropy calibration currently supports per-tensor ranges")
        if histogram_bins < 16:
            raise ConfigurationError("entropy histogram needs at least 16 bins")
        default_quantized_bins = 1 << min(self.config.bits - 1, 8)
        self.quantized_bins = quantized_bins or default_quantized_bins
        if not 2 <= self.quantized_bins < histogram_bins:
            raise ConfigurationError("quantized_bins must be in [2, histogram_bins)")
        self.histogram_bins = histogram_bins
        self.histogram = np.zeros(histogram_bins, dtype=np.float64)
        self.range_max = 0.0
        self.observed_min = float("inf")
        self.observed_max = float("-inf")

    @property
    def sample_count(self) -> int:
        return int(np.rint(np.sum(self.histogram)))

    def update(self, values: npt.ArrayLike) -> None:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return
        self.observed_min = min(self.observed_min, float(np.min(finite)))
        self.observed_max = max(self.observed_max, float(np.max(finite)))
        magnitudes = np.abs(finite)
        batch_max = float(np.max(magnitudes))
        if batch_max == 0.0:
            self.histogram[0] += finite.size
            return

        new_range = max(self.range_max, batch_max)
        if self.range_max > 0.0 and new_range > self.range_max:
            old_width = self.range_max / self.histogram_bins
            old_centers = (np.arange(self.histogram_bins) + 0.5) * old_width
            self.histogram, _ = np.histogram(
                old_centers,
                bins=self.histogram_bins,
                range=(0.0, new_range),
                weights=self.histogram,
            )
            self.histogram = self.histogram.astype(np.float64)
        additions, _ = np.histogram(
            magnitudes, bins=self.histogram_bins, range=(0.0, new_range)
        )
        self.histogram += additions
        self.range_max = new_range

    def _expanded_quantized_distribution(self, distribution: Array) -> Array:
        expanded = np.zeros_like(distribution, dtype=np.float64)
        for indices in np.array_split(np.arange(distribution.size), self.quantized_bins):
            nonzero = indices[distribution[indices] > 0]
            if nonzero.size:
                expanded[nonzero] = np.sum(distribution[indices]) / nonzero.size
        return expanded

    def clipping_threshold(self) -> float:
        if self.sample_count == 0 or self.range_max <= 0:
            return 0.0
        best_bin = self.histogram_bins
        best_divergence = float("inf")
        for threshold_bin in range(self.quantized_bins, self.histogram_bins + 1):
            reference = self.histogram[:threshold_bin].copy()
            if threshold_bin < self.histogram_bins:
                reference[-1] += np.sum(self.histogram[threshold_bin:])
            total = np.sum(reference)
            if total <= 0:
                continue
            candidate = self._expanded_quantized_distribution(reference)
            candidate_total = np.sum(candidate)
            if candidate_total <= 0:
                continue
            divergence = _kl_divergence(reference / total, candidate / candidate_total)
            if divergence < best_divergence:
                best_divergence = divergence
                best_bin = threshold_bin
        return self.range_max * best_bin / self.histogram_bins

    def calculate_qparams(self) -> QuantizationParams:
        if self.sample_count == 0:
            raise ConfigurationError("observer has not received any finite samples")
        threshold = self.clipping_threshold()
        if self.config.symmetric:
            minimum, maximum = -threshold, threshold
        else:
            minimum = max(self.observed_min, -threshold)
            maximum = min(self.observed_max, threshold)
            if minimum == maximum:
                maximum = minimum + np.finfo(np.float32).eps
        return _quantization_params(minimum, maximum, self.config)


@dataclass(frozen=True)
class CalibrationTable:
    """Named activation quantization parameters produced by calibration."""

    tensors: Mapping[str, QuantizationParams]
    method: str
    samples: int

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ConfigurationError("calibration table must contain at least one tensor")
        if self.samples <= 0:
            raise ConfigurationError("calibration sample count must be positive")
        object.__setattr__(self, "tensors", dict(self.tensors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "samples": self.samples,
            "tensors": {name: params.to_dict() for name, params in self.tensors.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationTable:
        return cls(
            tensors={
                name: QuantizationParams.from_dict(params)
                for name, params in value["tensors"].items()
            },
            method=value["method"],
            samples=int(value["samples"]),
        )

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


class RepresentativeCalibrator:
    """Collect named tensors from a representative dataset into a calibration table."""

    def __init__(
        self,
        config: QuantizationConfig | None = None,
        *,
        method: str = "entropy",
        histogram_bins: int = 2048,
    ) -> None:
        if method not in {"entropy", "minmax"}:
            raise ConfigurationError("calibration method must be 'entropy' or 'minmax'")
        self.config = config or QuantizationConfig(symmetric=True)
        self.method = method
        self.histogram_bins = histogram_bins
        self._observers: dict[str, Observer] = {}
        self._samples = 0

    def _make_observer(self) -> Observer:
        if self.method == "entropy":
            return EntropyObserver(self.config, histogram_bins=self.histogram_bins)
        return MinMaxObserver(self.config)

    def observe(self, activations: Mapping[str, npt.ArrayLike]) -> None:
        if not activations:
            raise ConfigurationError("representative sample has no activation tensors")
        for name, values in activations.items():
            if not name:
                raise ConfigurationError("activation tensor names must not be empty")
            self._observers.setdefault(name, self._make_observer()).update(values)
        self._samples += 1

    def calibrate(
        self, representative_data: Iterable[Mapping[str, npt.ArrayLike]]
    ) -> CalibrationTable:
        for activations in representative_data:
            self.observe(activations)
        if self._samples == 0:
            raise ConfigurationError("representative dataset is empty")
        return CalibrationTable(
            tensors={name: observer.calculate_qparams() for name, observer in self._observers.items()},
            method=self.method,
            samples=self._samples,
        )

