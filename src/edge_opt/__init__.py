"""Edge-Opt public package API."""

from .core import DType, ModelSpec, OperatorKind, OperatorSpec, TensorSpec
from .errors import AccuracyBudgetExceeded, ConfigurationError, EdgeOptError
from .hardware import HardwareProfile, MemoryTier
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

__all__ = [
    "AccuracyBudgetExceeded",
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
    "OperatorKind",
    "OperatorSpec",
    "PolynomialPruningSchedule",
    "PruningStepResult",
    "QuantizationConfig",
    "QuantizationParams",
    "RepresentativeCalibrator",
    "TensorSpec",
    "dequantize",
    "estimate_sparse_storage",
    "fake_quantize",
    "measured_sparsity",
    "quantize",
]

__version__ = "0.1.0"
