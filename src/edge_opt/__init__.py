"""Edge-Opt public package API."""

from .core import DType, ModelSpec, OperatorKind, OperatorSpec, TensorSpec
from .errors import AccuracyBudgetExceeded, ConfigurationError, EdgeOptError
from .hardware import BUILTIN_PROFILES, HardwareProfile, MemoryTier, load_builtin_profile
from .pipeline import (
    OptimizationConfig,
    OptimizationPipeline,
    OptimizationResult,
    QualityAssessment,
    QualityConstraint,
    optimize_model_spec,
)
from .profiling import (
    BenchmarkResult,
    ModelProfile,
    OperatorProfile,
    RooflineProfiler,
    benchmark_callable,
    operator_flops,
)
from .pruning import (
    MagnitudePruner,
    PolynomialPruningSchedule,
    PruningStepResult,
    estimate_sparse_storage,
    measured_sparsity,
)
from .quantization import (
    CalibrationTable,
    EntropyObserver,
    MinMaxObserver,
    QuantizationConfig,
    QuantizationParams,
    RepresentativeCalibrator,
    dequantize,
    fake_quantize,
    quantize,
)
from .torch_integration import (
    QATConfig,
    QATPreparationReport,
    TorchMagnitudePruner,
    convert_qat,
    export_int8_bundle,
    freeze_qat_observers,
    is_torch_available,
    prepare_qat,
    set_fake_quantization,
)

__all__ = [
    "AccuracyBudgetExceeded",
    "BUILTIN_PROFILES",
    "BenchmarkResult",
    "ConfigurationError",
    "CalibrationTable",
    "DType",
    "EdgeOptError",
    "EntropyObserver",
    "MinMaxObserver",
    "HardwareProfile",
    "MemoryTier",
    "MagnitudePruner",
    "ModelSpec",
    "ModelProfile",
    "OperatorKind",
    "OperatorProfile",
    "OperatorSpec",
    "OptimizationConfig",
    "OptimizationPipeline",
    "OptimizationResult",
    "PolynomialPruningSchedule",
    "PruningStepResult",
    "QATConfig",
    "QATPreparationReport",
    "QualityAssessment",
    "QualityConstraint",
    "QuantizationConfig",
    "QuantizationParams",
    "RepresentativeCalibrator",
    "RooflineProfiler",
    "TensorSpec",
    "TorchMagnitudePruner",
    "benchmark_callable",
    "convert_qat",
    "dequantize",
    "estimate_sparse_storage",
    "export_int8_bundle",
    "fake_quantize",
    "freeze_qat_observers",
    "is_torch_available",
    "load_builtin_profile",
    "measured_sparsity",
    "operator_flops",
    "optimize_model_spec",
    "prepare_qat",
    "quantize",
    "set_fake_quantization",
]

__version__ = "0.1.0"
