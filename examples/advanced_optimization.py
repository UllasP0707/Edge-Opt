"""Reproducible baseline-to-modern-method comparison on a synthetic fixture.

The quality values in this example are output MSE on deterministic synthetic
data.  Latency values remain analytical roofline predictions from the selected
hardware profile; neither is presented as a real-model or measured-board claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from edge_opt import (
    ChannelStatsObserver,
    DType,
    EntropyObserver,
    MagnitudePruner,
    MinMaxObserver,
    ModelSpec,
    NMPruningPattern,
    OperatorKind,
    OperatorSpec,
    OptimizationConfig,
    PolynomialPruningSchedule,
    QualityConstraint,
    QuantizationConfig,
    RooflineProfiler,
    StrategyComparison,
    StrategyMeasurement,
    TensorSpec,
    WandaPruner,
    apply_smoothquant,
    fake_quantize,
    load_builtin_profile,
    optimize_model_spec,
)


def _model_spec() -> ModelSpec:
    return ModelSpec(
        "advanced-linear-fixture",
        (
            OperatorSpec(
                "projection",
                OperatorKind.LINEAR,
                (TensorSpec((128, 8), DType.FP32, "input"),),
                TensorSpec((128, 4), DType.FP32, "output"),
                weight_shape=(4, 8),
            ),
        ),
        {"evidence": "deterministic synthetic fixture; not a model benchmark"},
    )


def _w8a8_output(
    activations: np.ndarray,
    weights: np.ndarray,
    *,
    entropy_activations: bool,
) -> np.ndarray:
    activation_observer = (
        EntropyObserver(QuantizationConfig(symmetric=True), histogram_bins=512)
        if entropy_activations
        else MinMaxObserver(QuantizationConfig(symmetric=True))
    )
    activation_observer.update(activations)
    weight_observer = MinMaxObserver(
        QuantizationConfig(symmetric=True, per_channel=True, channel_axis=0)
    )
    weight_observer.update(weights)
    quantized_activations = fake_quantize(
        activations, activation_observer.calculate_qparams()
    )
    quantized_weights = fake_quantize(weights, weight_observer.calculate_qparams())
    return quantized_activations @ quantized_weights.T


def _mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean(np.square(reference - candidate)))


def build_comparison() -> StrategyComparison:
    """Evaluate magnitude, Wanda, 2:4 Wanda, and SmoothQuant on one fixture."""

    random = np.random.default_rng(23)
    channel_ranges = np.asarray([100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03])
    calibration = random.normal(size=(256, 8)) * channel_ranges
    evaluation = random.normal(size=(128, 8)) * channel_ranges
    weights = random.normal(size=(4, 8)) * (0.03 / channel_ranges)
    reference = evaluation @ weights.T

    statistics_observer = ChannelStatsObserver(channel_axis=-1)
    statistics_observer.update(calibration)
    statistics = statistics_observer.calculate()

    schedule = PolynomialPruningSchedule(
        final_sparsity=0.5,
        begin_step=0,
        end_step=1,
        update_frequency=1,
    )
    magnitude_weights, magnitude_result = MagnitudePruner(
        schedule, global_pruning=False
    ).step({"projection": weights}, 1)
    wanda_weights, wanda_result = WandaPruner(0.5).prune(
        {"projection": weights}, {"projection": statistics}
    )
    two_of_four = NMPruningPattern(2, 4)
    structured_weights, structured_result = WandaPruner(
        0.5, pattern=two_of_four
    ).prune({"projection": weights}, {"projection": statistics})

    magnitude_output = _w8a8_output(
        evaluation, magnitude_weights["projection"], entropy_activations=True
    )
    wanda_output = _w8a8_output(
        evaluation, wanda_weights["projection"], entropy_activations=True
    )
    structured_output = _w8a8_output(
        evaluation, structured_weights["projection"], entropy_activations=True
    )
    smoothquant = apply_smoothquant(
        structured_weights["projection"], statistics
    )
    smoothquant_output = _w8a8_output(
        smoothquant.transform_activations(evaluation),
        smoothquant.smoothed_weights,
        entropy_activations=False,
    )

    model = _model_spec()
    profiler = RooflineProfiler(load_builtin_profile("nvidia_a100_reference"))
    baseline_profile = profiler.profile(model)
    unstructured_profile = profiler.profile(
        optimize_model_spec(model, OptimizationConfig(target_sparsity=0.5))
    )
    structured_profile = profiler.profile(
        optimize_model_spec(
            model,
            OptimizationConfig(
                target_sparsity=0.5,
                sparsity_pattern=two_of_four,
            ),
        )
    )

    baseline = StrategyMeasurement(
        "FP32 baseline",
        "none",
        "none",
        0.0,
        "synthetic_fixture",
        baseline_profile,
        {"seed": 23, "metric": "output MSE against the FP32 fixture"},
    )
    candidates = (
        StrategyMeasurement(
            "Magnitude + entropy PTQ",
            "local magnitude 50%",
            "entropy W8A8",
            _mse(reference, magnitude_output),
            "synthetic_fixture",
            unstructured_profile,
            {"actual_sparsity": magnitude_result.actual_sparsity},
        ),
        StrategyMeasurement(
            "Wanda + entropy PTQ",
            "Wanda 50%",
            "entropy W8A8",
            _mse(reference, wanda_output),
            "synthetic_fixture",
            unstructured_profile,
            {"actual_sparsity": wanda_result.actual_sparsity},
        ),
        StrategyMeasurement(
            "Wanda 2:4 + entropy PTQ",
            "Wanda 2:4",
            "entropy W8A8",
            _mse(reference, structured_output),
            "synthetic_fixture",
            structured_profile,
            {"actual_sparsity": structured_result.actual_sparsity},
        ),
        StrategyMeasurement(
            "Wanda 2:4 + SmoothQuant",
            "Wanda 2:4",
            "SmoothQuant W8A8",
            _mse(reference, smoothquant_output),
            "synthetic_fixture",
            structured_profile,
            {
                "actual_sparsity": structured_result.actual_sparsity,
                "smoothquant_alpha": smoothquant.alpha,
            },
        ),
    )
    return StrategyComparison(
        baseline,
        candidates,
        QualityConstraint(
            "output_mse",
            higher_is_better=False,
            max_degradation=0.01,
            strict=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/advanced")
    arguments = parser.parse_args()
    output_directory = Path(arguments.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison()
    comparison.to_json(output_directory / "comparison.json")
    (output_directory / "comparison.md").write_text(comparison.to_markdown())
    print(comparison.to_markdown(), end="")


if __name__ == "__main__":
    main()
