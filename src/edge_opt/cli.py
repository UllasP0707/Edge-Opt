"""Command-line interface for profiling and optimization reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .core import DType, ModelSpec, OperatorKind, OperatorSpec, TensorSpec
from .errors import EdgeOptError
from .hardware import BUILTIN_PROFILES, HardwareProfile, load_builtin_profile
from .pipeline import OptimizationConfig, OptimizationPipeline, QualityConstraint
from .profiling import RooflineProfiler


def _load_model(path: str | Path) -> ModelSpec:
    return ModelSpec.from_dict(json.loads(Path(path).read_text()))


def _load_hardware(value: str) -> HardwareProfile:
    return (
        load_builtin_profile(value)
        if value in BUILTIN_PROFILES
        else HardwareProfile.from_json(value)
    )


def _write_or_print(content: str, output: str | None) -> None:
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    else:
        print(content, end="")


def _demo_model() -> ModelSpec:
    """Static encoder/latent/decoder graph with a ~148 MiB FP32 weight footprint."""

    return ModelSpec(
        "spatial-audio-latent-pipeline",
        (
            OperatorSpec(
                "encoder_projection",
                OperatorKind.LINEAR,
                (TensorSpec((1, 4_096), DType.FP32, "audio_embedding"),),
                TensorSpec((1, 4_096), DType.FP32, "encoded"),
                weight_shape=(4_096, 4_096),
            ),
            OperatorSpec(
                "latent_mixer",
                OperatorKind.LINEAR,
                (TensorSpec((1, 4_096), DType.FP32, "encoded"),),
                TensorSpec((1, 4_096), DType.FP32, "latent"),
                weight_shape=(4_096, 4_096),
            ),
            OperatorSpec(
                "decoder_projection",
                OperatorKind.LINEAR,
                (TensorSpec((1, 4_096), DType.FP32, "latent"),),
                TensorSpec((1, 1_280), DType.FP32, "spatial_output"),
                weight_shape=(1_280, 4_096),
            ),
        ),
        {"task": "generative encoder-decoder / spatial audio latent pipeline"},
    )


def _representative_demo_data() -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    return [
        {
            "encoder_projection": rng.normal(0, 0.5, 2_048).astype(np.float32),
            "latent_mixer": rng.normal(0, 0.35, 2_048).astype(np.float32),
            "decoder_projection": rng.normal(0, 0.6, 1_280).astype(np.float32),
        }
        for _ in range(8)
    ]


def _command_profile(arguments: argparse.Namespace) -> int:
    profile = RooflineProfiler(_load_hardware(arguments.hardware)).profile(
        _load_model(arguments.model)
    )
    content = (
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n"
        if arguments.format == "json"
        else profile.to_markdown()
    )
    _write_or_print(content, arguments.output)
    return 0


def _pipeline_from_arguments(arguments: argparse.Namespace) -> OptimizationPipeline:
    return OptimizationPipeline(
        _load_hardware(arguments.hardware),
        OptimizationConfig(
            target_sparsity=arguments.target_sparsity,
            weight_dtype=DType(arguments.weight_dtype),
            activation_dtype=DType(arguments.activation_dtype),
            sparse_encoding=arguments.sparse_encoding,
            calibration_method=arguments.calibration,
        ),
        QualityConstraint(
            metric_name=arguments.metric,
            higher_is_better=not arguments.lower_is_better,
            max_degradation=arguments.max_degradation,
            strict=True,
        ),
    )


def _command_optimize(arguments: argparse.Namespace) -> int:
    result = _pipeline_from_arguments(arguments).run(
        _load_model(arguments.model),
        baseline_quality=arguments.baseline_quality,
        optimized_quality=arguments.optimized_quality,
    )
    content = (
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
        if arguments.format == "json"
        else result.to_markdown()
    )
    _write_or_print(content, arguments.output)
    if arguments.model_output:
        model_output = Path(arguments.model_output)
        model_output.parent.mkdir(parents=True, exist_ok=True)
        model_output.write_text(
            json.dumps(result.optimized_model.to_dict(), indent=2, sort_keys=True) + "\n"
        )
    return 0


def _command_demo(arguments: argparse.Namespace) -> int:
    destination = Path(arguments.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model = _demo_model()
    pipeline = OptimizationPipeline(
        load_builtin_profile("arm_cortex_a76"),
        OptimizationConfig(target_sparsity=0.65),
        QualityConstraint(max_degradation=0.01, strict=True),
    )
    result = pipeline.run(
        model,
        baseline_quality=0.924,
        optimized_quality=0.918,
        representative_data=_representative_demo_data(),
    )
    (destination / "model.json").write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    result.to_json(destination / "optimization.json")
    (destination / "optimization.md").write_text(result.to_markdown())
    if result.calibration:
        result.calibration.to_json(destination / "calibration.json")
    print(f"Demo accepted; reports written to {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-opt", description="Hardware-aware model optimization and roofline profiling"
    )
    parser.add_argument("--version", action="version", version="edge-opt 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profiles = subparsers.add_parser("profiles", help="list built-in hardware profiles")
    profiles.set_defaults(handler=lambda _: print("\n".join(BUILTIN_PROFILES)) or 0)

    profile = subparsers.add_parser("profile", help="profile a ModelSpec JSON file")
    profile.add_argument("model")
    profile.add_argument("--hardware", default="arm_cortex_a76")
    profile.add_argument("--format", choices=("json", "markdown"), default="markdown")
    profile.add_argument("--output")
    profile.set_defaults(handler=_command_profile)

    optimize = subparsers.add_parser("optimize", help="optimize, gate, and profile a model")
    optimize.add_argument("model")
    optimize.add_argument("--baseline-quality", type=float, required=True)
    optimize.add_argument("--optimized-quality", type=float, required=True)
    optimize.add_argument("--metric", default="accuracy")
    optimize.add_argument("--lower-is-better", action="store_true")
    optimize.add_argument("--max-degradation", type=float, default=0.01)
    optimize.add_argument("--target-sparsity", type=float, default=0.65)
    optimize.add_argument("--weight-dtype", choices=("int8", "int4"), default="int8")
    optimize.add_argument("--activation-dtype", choices=("int8", "uint8"), default="int8")
    optimize.add_argument(
        "--sparse-encoding",
        choices=("bitmap", "coordinate", "ideal", "dense"),
        default="bitmap",
    )
    optimize.add_argument("--calibration", choices=("entropy", "minmax"), default="entropy")
    optimize.add_argument("--hardware", default="arm_cortex_a76")
    optimize.add_argument("--format", choices=("json", "markdown"), default="markdown")
    optimize.add_argument("--output")
    optimize.add_argument("--model-output")
    optimize.set_defaults(handler=_command_optimize)

    demo = subparsers.add_parser("demo", help="run the deterministic spatial-audio demo")
    demo.add_argument("--output-dir", default="reports/demo")
    demo.set_defaults(handler=_command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (EdgeOptError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"edge-opt: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
