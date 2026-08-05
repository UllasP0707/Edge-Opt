"""Edge-Opt public package API."""

from .core import DType, ModelSpec, OperatorKind, OperatorSpec, TensorSpec
from .errors import AccuracyBudgetExceeded, ConfigurationError, EdgeOptError
from .hardware import BUILTIN_PROFILES, HardwareProfile, MemoryTier, load_builtin_profile
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
from .pruning import (
    MagnitudePruner,
    PolynomialPruningSchedule,
    PruningStepResult,
    estimate_sparse_storage,
    measured_sparsity,
)
from .profiling import (
    BenchmarkResult,
    ModelProfile,
    OperatorProfile,
    RooflineProfiler,
    benchmark_callable,
    operator_flops,
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
    "PolynomialPruningSchedule",
    "PruningStepResult",
    "QuantizationConfig",
    "QuantizationParams",
    "RepresentativeCalibrator",
    "RooflineProfiler",
    "TensorSpec",
    "benchmark_callable",
    "dequantize",
    "estimate_sparse_storage",
    "fake_quantize",
    "load_builtin_profile",
    "measured_sparsity",
    "operator_flops",
    "quantize",
]

__version__ = "0.1.0"
