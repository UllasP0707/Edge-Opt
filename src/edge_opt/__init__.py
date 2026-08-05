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
    "ModelSpec",
    "OperatorKind",
    "OperatorSpec",
    "QuantizationConfig",
    "QuantizationParams",
    "RepresentativeCalibrator",
    "TensorSpec",
    "dequantize",
    "fake_quantize",
    "quantize",
]

__version__ = "0.1.0"
