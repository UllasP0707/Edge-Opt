"""Streaming channel-wise activation statistics for pruning and quantization."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .errors import ConfigurationError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ChannelStatistics:
    """Final activation statistics for one feature/channel dimension."""

    l2_norm: FloatArray
    absmax: FloatArray
    values_per_channel: int
    batches: int
    channel_axis: int

    def __post_init__(self) -> None:
        l2_norm = np.asarray(self.l2_norm, dtype=np.float64)
        absmax = np.asarray(self.absmax, dtype=np.float64)
        if l2_norm.ndim != 1 or l2_norm.size == 0:
            raise ConfigurationError("channel L2 norms must be a nonempty vector")
        if absmax.shape != l2_norm.shape:
            raise ConfigurationError("channel L2 norms and absolute maxima must have equal shape")
        if not np.all(np.isfinite(l2_norm)) or not np.all(np.isfinite(absmax)):
            raise ConfigurationError("channel statistics must be finite")
        if np.any(l2_norm < 0) or np.any(absmax < 0):
            raise ConfigurationError("channel statistics must not be negative")
        if self.values_per_channel <= 0 or self.batches <= 0:
            raise ConfigurationError("channel statistics require observed activation values")
        object.__setattr__(self, "l2_norm", l2_norm)
        object.__setattr__(self, "absmax", absmax)

    @property
    def channels(self) -> int:
        return int(self.l2_norm.size)

    @property
    def rms(self) -> FloatArray:
        """Root-mean-square activation magnitude for each channel."""

        return np.asarray(
            self.l2_norm / np.sqrt(self.values_per_channel), dtype=np.float64
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "l2_norm": self.l2_norm.tolist(),
            "absmax": self.absmax.tolist(),
            "values_per_channel": self.values_per_channel,
            "batches": self.batches,
            "channel_axis": self.channel_axis,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChannelStatistics:
        return cls(
            l2_norm=np.asarray(value["l2_norm"], dtype=np.float64),
            absmax=np.asarray(value["absmax"], dtype=np.float64),
            values_per_channel=int(value["values_per_channel"]),
            batches=int(value["batches"]),
            channel_axis=int(value["channel_axis"]),
        )


class ChannelStatsObserver:
    """Accumulate exact per-channel L2 norms and absolute maxima.

    The reduction covers every dimension except ``channel_axis``. This makes
    ``channel_axis=-1`` appropriate for linear-layer inputs and ``1`` for
    NCHW convolution inputs. Updates may have different batch or spatial sizes,
    but the channel count must stay constant.
    """

    def __init__(self, channel_axis: int = -1) -> None:
        self.channel_axis = channel_axis
        self._sum_squares: FloatArray | None = None
        self._absmax: FloatArray | None = None
        self._values_per_channel = 0
        self._batches = 0

    @property
    def initialized(self) -> bool:
        return self._sum_squares is not None

    def update(self, values: npt.ArrayLike) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.size == 0:
            raise ConfigurationError("channel observer requires a nonempty tensor with rank >= 1")
        if not np.all(np.isfinite(array)):
            raise ConfigurationError("channel observer received non-finite activation values")
        axis = self.channel_axis % array.ndim
        channel_first = np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)
        sum_squares = np.asarray(
            np.sum(np.square(channel_first), axis=1, dtype=np.float64),
            dtype=np.float64,
        )
        absmax = np.asarray(np.max(np.abs(channel_first), axis=1), dtype=np.float64)
        if self._sum_squares is not None and self._sum_squares.shape != sum_squares.shape:
            raise ConfigurationError("activation channel count changed between batches")
        if self._sum_squares is None:
            self._sum_squares = sum_squares
            self._absmax = absmax
        else:
            assert self._absmax is not None
            self._sum_squares += sum_squares
            self._absmax = np.maximum(self._absmax, absmax)
        self._values_per_channel += int(channel_first.shape[1])
        self._batches += 1

    def calculate(self) -> ChannelStatistics:
        if self._sum_squares is None or self._absmax is None:
            raise ConfigurationError("channel observer has not received activation values")
        return ChannelStatistics(
            l2_norm=np.sqrt(self._sum_squares),
            absmax=self._absmax.copy(),
            values_per_channel=self._values_per_channel,
            batches=self._batches,
            channel_axis=self.channel_axis,
        )


@dataclass(frozen=True)
class ActivationStatisticsTable:
    """Serializable channel statistics keyed by module or tensor name."""

    tensors: Mapping[str, ChannelStatistics]

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ConfigurationError("activation statistics table must not be empty")
        if any(not name for name in self.tensors):
            raise ConfigurationError("activation statistic names must not be empty")
        object.__setattr__(self, "tensors", dict(self.tensors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "edge-opt-channel-statistics-v1",
            "tensors": {name: stats.to_dict() for name, stats in self.tensors.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationStatisticsTable:
        format_name = value.get("format", "edge-opt-channel-statistics-v1")
        if format_name != "edge-opt-channel-statistics-v1":
            raise ConfigurationError(f"unsupported activation statistics format {format_name!r}")
        return cls(
            tensors={
                name: ChannelStatistics.from_dict(stats)
                for name, stats in value["tensors"].items()
            }
        )

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> ActivationStatisticsTable:
        return cls.from_dict(json.loads(Path(path).read_text()))


class ActivationStatisticsCollector:
    """Collect named activation tensors with configurable channel axes."""

    def __init__(
        self,
        *,
        channel_axes: Mapping[str, int] | None = None,
        default_channel_axis: int = -1,
    ) -> None:
        self.channel_axes = dict(channel_axes or {})
        self.default_channel_axis = default_channel_axis
        self._observers: dict[str, ChannelStatsObserver] = {}

    def observe(self, activations: Mapping[str, npt.ArrayLike]) -> None:
        if not activations:
            raise ConfigurationError("representative sample has no activation tensors")
        for name, values in activations.items():
            if not name:
                raise ConfigurationError("activation tensor names must not be empty")
            axis = self.channel_axes.get(name, self.default_channel_axis)
            observer = self._observers.setdefault(name, ChannelStatsObserver(axis))
            if observer.channel_axis != axis:
                raise ConfigurationError(f"channel axis changed for activation {name!r}")
            observer.update(values)

    def collect(
        self, representative_data: Iterable[Mapping[str, npt.ArrayLike]]
    ) -> ActivationStatisticsTable:
        observed = False
        for activations in representative_data:
            self.observe(activations)
            observed = True
        if not observed:
            raise ConfigurationError("representative dataset is empty")
        return self.table()

    def table(self) -> ActivationStatisticsTable:
        return ActivationStatisticsTable(
            {name: observer.calculate() for name, observer in self._observers.items()}
        )
