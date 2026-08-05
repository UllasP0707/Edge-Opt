"""End-to-end model-spec optimization with a fail-closed quality gate."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy.typing as npt

from .core import DType, ModelSpec, OperatorSpec, TensorSpec
from .errors import AccuracyBudgetExceeded, ConfigurationError
from .hardware import HardwareProfile
from .profiling import ModelProfile, RooflineProfiler
from .quantization import (
    CalibrationTable,
    QuantizationConfig,
    RepresentativeCalibrator,
)


@dataclass(frozen=True)
class QualityConstraint:
    """Maximum tolerated absolute metric degradation.

    Accuracy-like metrics use ``higher_is_better=True``. Error or loss metrics
    use ``False``. The default bound is strict: a degradation exactly equal to
    0.01 fails the ``< 1.0%`` requirement.
    """

    metric_name: str = "accuracy"
    higher_is_better: bool = True
    max_degradation: float = 0.01
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.metric_name.strip():
            raise ConfigurationError("quality metric name must not be empty")
        if self.max_degradation <= 0:
            raise ConfigurationError("max_degradation must be positive")

    def assess(self, baseline: float, optimized: float) -> QualityAssessment:
        if not math.isfinite(baseline) or not math.isfinite(optimized):
            raise ConfigurationError("quality measurements must be finite")
        degradation = baseline - optimized if self.higher_is_better else optimized - baseline
        equal_to_limit = math.isclose(
            degradation, self.max_degradation, rel_tol=0.0, abs_tol=1e-12
        )
        passed = degradation < self.max_degradation and not (self.strict and equal_to_limit)
        if not self.strict and equal_to_limit:
            passed = True
        return QualityAssessment(
            metric_name=self.metric_name,
            baseline=float(baseline),
            optimized=float(optimized),
            degradation=float(degradation),
            max_degradation=self.max_degradation,
            strict=self.strict,
            passed=passed,
        )


@dataclass(frozen=True)
class QualityAssessment:
    metric_name: str
    baseline: float
    optimized: float
    degradation: float
    max_degradation: float
    strict: bool
    passed: bool

    @property
    def comparison(self) -> str:
        return "<" if self.strict else "<="

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline": self.baseline,
            "optimized": self.optimized,
            "degradation": self.degradation,
            "constraint": {
                "comparison": self.comparison,
                "max_degradation": self.max_degradation,
            },
            "passed": self.passed,
        }


@dataclass(frozen=True)
class OptimizationConfig:
    """Static transformation applied after pruning/QAT training is complete."""

    target_sparsity: float = 0.65
    weight_dtype: DType = DType.INT8
    activation_dtype: DType = DType.INT8
    sparse_encoding: str = "bitmap"
    calibration_method: str = "entropy"

    def __post_init__(self) -> None:
        if not isinstance(self.weight_dtype, DType):
            object.__setattr__(self, "weight_dtype", DType(self.weight_dtype))
        if not isinstance(self.activation_dtype, DType):
            object.__setattr__(self, "activation_dtype", DType(self.activation_dtype))
        if not 0.0 <= self.target_sparsity < 1.0:
            raise ConfigurationError("target sparsity must be in [0, 1)")
        if self.weight_dtype not in {DType.INT8, DType.INT4}:
            raise ConfigurationError("optimized weights must be INT8 or INT4")
        if self.activation_dtype not in {DType.INT8, DType.UINT8}:
            raise ConfigurationError("optimized activations must be INT8 or UINT8")
        if self.sparse_encoding not in {"bitmap", "coordinate", "ideal", "dense"}:
            raise ConfigurationError("unsupported sparse encoding")
        if self.calibration_method not in {"entropy", "minmax"}:
            raise ConfigurationError("calibration method must be entropy or minmax")


def _quantized_tensor(tensor: TensorSpec, dtype: DType) -> TensorSpec:
    return TensorSpec(shape=tensor.shape, dtype=dtype, name=tensor.name)


def optimize_model_spec(model: ModelSpec, config: OptimizationConfig) -> ModelSpec:
    """Describe the post-QAT, quantized and pruned form of a model graph."""

    optimized: list[OperatorSpec] = []
    for operator in model.operators:
        weighted = operator.weight_shape is not None
        attributes = dict(operator.attributes)
        if weighted and config.target_sparsity > 0:
            attributes["sparse_encoding"] = config.sparse_encoding
        optimized.append(
            OperatorSpec(
                name=operator.name,
                kind=operator.kind,
                inputs=tuple(
                    _quantized_tensor(tensor, config.activation_dtype)
                    for tensor in operator.inputs
                ),
                output=_quantized_tensor(operator.output, config.activation_dtype),
                weight_shape=operator.weight_shape,
                weight_dtype=config.weight_dtype if weighted else operator.weight_dtype,
                sparsity=config.target_sparsity if weighted else 0.0,
                attributes=attributes,
            )
        )
    metadata = dict(model.metadata)
    metadata["edge_opt"] = {
        "target_sparsity": config.target_sparsity,
        "weight_dtype": config.weight_dtype.value,
        "activation_dtype": config.activation_dtype.value,
        "sparse_encoding": config.sparse_encoding,
    }
    return ModelSpec(f"{model.name}-optimized", tuple(optimized), metadata)


@dataclass(frozen=True)
class OptimizationResult:
    """Profiles, calibration, and quality evidence from an accepted run."""

    baseline_model: ModelSpec
    optimized_model: ModelSpec
    baseline_profile: ModelProfile
    optimized_profile: ModelProfile
    quality: QualityAssessment
    config: OptimizationConfig
    calibration: CalibrationTable | None = None

    @property
    def predicted_speedup(self) -> float:
        denominator = self.optimized_profile.predicted_latency_seconds
        return self.baseline_profile.predicted_latency_seconds / denominator if denominator else 1.0

    @property
    def weight_reduction(self) -> float:
        baseline = self.baseline_profile.weight_bytes
        if not baseline:
            return 0.0
        return 1.0 - self.optimized_profile.weight_bytes / baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.quality.passed,
            "quality": self.quality.to_dict(),
            "optimization": {
                "target_sparsity": self.config.target_sparsity,
                "weight_dtype": self.config.weight_dtype.value,
                "activation_dtype": self.config.activation_dtype.value,
                "sparse_encoding": self.config.sparse_encoding,
                "calibration_method": self.config.calibration_method,
            },
            "comparison": {
                "predicted_speedup": self.predicted_speedup,
                "weight_reduction": self.weight_reduction,
            },
            "baseline": self.baseline_profile.to_dict(),
            "optimized": self.optimized_profile.to_dict(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "optimized_model": self.optimized_model.to_dict(),
        }

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def to_markdown(self) -> str:
        quality_status = "PASS" if self.quality.passed else "FAIL"
        lines = [
            f"# Edge-Opt optimization: {self.baseline_model.name}",
            "",
            f"Quality gate: **{quality_status}** — {self.quality.metric_name} degradation "
            f"{self.quality.degradation:.6f} {self.quality.comparison} "
            f"{self.quality.max_degradation:.6f}",
            "",
            "| Metric | Baseline | Optimized | Change |",
            "|---|---:|---:|---:|",
            f"| {self.quality.metric_name} | {self.quality.baseline:.6f} | "
            f"{self.quality.optimized:.6f} | {-self.quality.degradation:+.6f} |",
            f"| Predicted latency (ms) | "
            f"{self.baseline_profile.predicted_latency_seconds * 1_000:.3f} | "
            f"{self.optimized_profile.predicted_latency_seconds * 1_000:.3f} | "
            f"{self.predicted_speedup:.2f}x speedup |",
            f"| Encoded weights (MiB) | "
            f"{self.baseline_profile.weight_bytes / 2**20:.3f} | "
            f"{self.optimized_profile.weight_bytes / 2**20:.3f} | "
            f"{self.weight_reduction:.1%} reduction |",
            f"| Executed operations | {self.baseline_profile.total_executed_flops:,} | "
            f"{self.optimized_profile.total_executed_flops:,} | — |",
            "",
            "## Optimized operator roofline",
            "",
            "| Operator | Tier | Ops/byte | Bound | Latency (ms) |",
            "|---|---:|---:|---:|---:|",
        ]
        lines.extend(
            f"| {operator.name} | {operator.memory_tier} | "
            f"{operator.operational_intensity:.3f} | {operator.bottleneck} | "
            f"{operator.predicted_latency_seconds * 1_000:.3f} |"
            for operator in self.optimized_profile.operators
        )
        return "\n".join(lines) + "\n"


class OptimizationPipeline:
    """Transform, calibrate, profile, and enforce the configured quality bound."""

    def __init__(
        self,
        hardware: HardwareProfile,
        config: OptimizationConfig | None = None,
        quality_constraint: QualityConstraint | None = None,
    ) -> None:
        self.hardware = hardware
        self.config = config or OptimizationConfig()
        self.quality_constraint = quality_constraint or QualityConstraint()

    def run(
        self,
        model: ModelSpec,
        *,
        baseline_quality: float,
        optimized_quality: float,
        representative_data: Iterable[Mapping[str, npt.ArrayLike]] | None = None,
    ) -> OptimizationResult:
        optimized_model = optimize_model_spec(model, self.config)
        calibration = None
        if representative_data is not None:
            calibration = RepresentativeCalibrator(
                QuantizationConfig(
                    bits=self.config.activation_dtype.bits,
                    symmetric=self.config.activation_dtype is DType.INT8,
                    narrow_range=self.config.activation_dtype is DType.INT8,
                ),
                method=self.config.calibration_method,
            ).calibrate(representative_data)
        quality = self.quality_constraint.assess(baseline_quality, optimized_quality)
        if not quality.passed:
            raise AccuracyBudgetExceeded(
                f"{quality.metric_name} degradation {quality.degradation:.6f} does not satisfy "
                f"{quality.comparison} {quality.max_degradation:.6f}; optimized artifact rejected"
            )
        profiler = RooflineProfiler(self.hardware)
        return OptimizationResult(
            baseline_model=model,
            optimized_model=optimized_model,
            baseline_profile=profiler.profile(model),
            optimized_profile=profiler.profile(optimized_model),
            quality=quality,
            config=self.config,
            calibration=calibration,
        )
