"""Evidence-labeled comparison reports for optimization strategies."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigurationError
from .pipeline import QualityAssessment, QualityConstraint
from .profiling import ModelProfile

QualityEvidence = Literal["measured", "synthetic_fixture", "user_supplied"]


@dataclass(frozen=True)
class StrategyMeasurement:
    """Quality and analytical profile for one concretely evaluated strategy."""

    name: str
    pruning: str
    quantization: str
    quality: float
    quality_evidence: QualityEvidence
    profile: ModelProfile
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("strategy name", self.name),
            ("pruning method", self.pruning),
            ("quantization method", self.quantization),
        ):
            if not value.strip():
                raise ConfigurationError(f"{label} must not be empty")
        if not math.isfinite(self.quality):
            raise ConfigurationError("strategy quality must be finite")
        if self.quality_evidence not in {
            "measured",
            "synthetic_fixture",
            "user_supplied",
        }:
            raise ConfigurationError("unsupported quality evidence source")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pruning": self.pruning,
            "quantization": self.quantization,
            "quality": self.quality,
            "quality_evidence": self.quality_evidence,
            "profile": self.profile.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StrategyComparison:
    """Compare evaluated candidates under one strict task-quality constraint."""

    baseline: StrategyMeasurement
    candidates: tuple[StrategyMeasurement, ...]
    quality_constraint: QualityConstraint

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ConfigurationError("strategy comparison requires at least one candidate")
        strategies = (self.baseline, *self.candidates)
        names = [strategy.name for strategy in strategies]
        if len(names) != len(set(names)):
            raise ConfigurationError("strategy names must be unique")
        targets = {strategy.profile.hardware_name for strategy in strategies}
        if len(targets) != 1:
            raise ConfigurationError(
                "strategy profiles must use the same hardware target"
            )
        latency_evidence = {strategy.profile.latency_evidence for strategy in strategies}
        if latency_evidence != {"analytical_prediction"}:
            raise ConfigurationError(
                "StrategyComparison currently accepts analytical roofline profiles only"
            )
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def assessment(self, strategy: StrategyMeasurement) -> QualityAssessment:
        return self.quality_constraint.assess(self.baseline.quality, strategy.quality)

    def predicted_speedup(self, strategy: StrategyMeasurement) -> float:
        latency = strategy.profile.predicted_latency_seconds
        if latency == 0:
            return 1.0
        return self.baseline.profile.predicted_latency_seconds / latency

    @property
    def accepted_candidates(self) -> tuple[StrategyMeasurement, ...]:
        return tuple(
            candidate for candidate in self.candidates if self.assessment(candidate).passed
        )

    def to_dict(self) -> dict[str, Any]:
        strategies: list[dict[str, Any]] = []
        for strategy in (self.baseline, *self.candidates):
            assessment = self.assessment(strategy)
            item = strategy.to_dict()
            item["quality_gate"] = assessment.to_dict()
            item["predicted_speedup_vs_baseline"] = self.predicted_speedup(strategy)
            strategies.append(item)
        return {
            "evidence": {
                "latency": "analytical_prediction",
                "hardware_profile_source": self.baseline.profile.hardware_profile_source,
                "warning": self.baseline.profile.hardware_profile_warning,
                "quality_sources": sorted(
                    {
                        strategy.quality_evidence
                        for strategy in (self.baseline, *self.candidates)
                    }
                ),
            },
            "quality_constraint": {
                "metric_name": self.quality_constraint.metric_name,
                "comparison": "<" if self.quality_constraint.strict else "<=",
                "max_degradation": self.quality_constraint.max_degradation,
                "higher_is_better": self.quality_constraint.higher_is_better,
            },
            "strategies": strategies,
        }

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def to_markdown(self) -> str:
        profile = self.baseline.profile
        lines = [
            "# Edge-Opt strategy comparison",
            "",
            f"Target: **{profile.hardware_name}**",
            "",
            "Latency evidence: **analytical prediction**",
            f"Hardware values: {profile.hardware_profile_source}",
            *(
                [f"Warning: {profile.hardware_profile_warning}"]
                if profile.hardware_profile_warning
                else []
            ),
            "",
            f"Quality gate: {self.quality_constraint.metric_name} degradation "
            f"{('<' if self.quality_constraint.strict else '<=')} "
            f"{self.quality_constraint.max_degradation:.6f}",
            "",
            "| Strategy | Pruning | Quantization | Quality (evidence) | "
            "Degradation | Gate | Predicted latency (ms) | Predicted speedup | "
            "Weights (KiB) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for strategy in (self.baseline, *self.candidates):
            assessment = self.assessment(strategy)
            lines.append(
                f"| {strategy.name} | {strategy.pruning} | {strategy.quantization} | "
                f"{strategy.quality:.6f} ({strategy.quality_evidence}) | "
                f"{assessment.degradation:.6f} | "
                f"{'PASS' if assessment.passed else 'FAIL'} | "
                f"{strategy.profile.predicted_latency_seconds * 1_000:.6f} | "
                f"{self.predicted_speedup(strategy):.2f}x | "
                f"{strategy.profile.weight_bytes / 1024:.3f} |"
            )
        return "\n".join(lines) + "\n"
