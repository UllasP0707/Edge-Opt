"""Optional PyTorch adapters for QAT, scheduled pruning, and INT8 export.

The converted modules store genuine INT8 weights but intentionally use a
dequantizing reference kernel in plain PyTorch. Vendor runtimes should consume
the exported weights and metadata to obtain hardware-accelerated execution.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigurationError
from .pruning import MagnitudePruner, PolynomialPruningSchedule, PruningStepResult
from .quantization import QuantizationConfig

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - the fallback is exercised without the extra
    torch = None
    nn = None
    functional = None
    TORCH_AVAILABLE = False


def is_torch_available() -> bool:
    return TORCH_AVAILABLE


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch integration requires the optional dependency: "
            "install Edge-Opt with `pip install 'edge-opt[torch]'`"
        )


@dataclass(frozen=True)
class QATConfig:
    """Fake-quantization behavior for supported PyTorch modules."""

    weight_bits: int = 8
    activation_bits: int = 8
    per_channel_weights: bool = True
    symmetric_activations: bool = True
    quantize_output: bool = True
    observer_momentum: float = 0.95
    excluded_module_names: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        QuantizationConfig(bits=self.weight_bits)
        QuantizationConfig(bits=self.activation_bits)
        if not 0.0 <= self.observer_momentum < 1.0:
            raise ConfigurationError("observer_momentum must be in [0, 1)")
        object.__setattr__(self, "excluded_module_names", frozenset(self.excluded_module_names))


@dataclass(frozen=True)
class QATPreparationReport:
    """Summary of graph modules replaced during QAT preparation."""

    linear_modules: int
    convolution_modules: int
    excluded_modules: tuple[str, ...]

    @property
    def total_modules(self) -> int:
        return self.linear_modules + self.convolution_modules


if TORCH_AVAILABLE:

    class TorchFakeQuantizer(nn.Module):
        """Observer-backed fake quantization with an identity STE backward pass."""

        def __init__(
            self,
            config: QuantizationConfig,
            *,
            momentum: float = 0.95,
        ) -> None:
            super().__init__()
            self.config = config
            self.momentum = momentum
            self.observer_enabled = True
            self.fake_quant_enabled = True
            self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
            self.register_buffer("zero_point", torch.zeros(1, dtype=torch.int64))
            self.register_buffer("observed_min", torch.zeros(1, dtype=torch.float32))
            self.register_buffer("observed_max", torch.zeros(1, dtype=torch.float32))
            self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

        def _batch_range(self, values: Any) -> tuple[Any, Any]:
            detached = values.detach().to(torch.float32)
            if self.config.per_channel:
                axis = self.config.channel_axis % detached.ndim
                reduce_axes = tuple(index for index in range(detached.ndim) if index != axis)
                return detached.amin(dim=reduce_axes), detached.amax(dim=reduce_axes)
            return detached.amin().reshape(1), detached.amax().reshape(1)

        def _update_observer(self, values: Any) -> None:
            batch_min, batch_max = self._batch_range(values)
            if not bool(self.initialized.item()) or self.observed_min.shape != batch_min.shape:
                self.observed_min = batch_min
                self.observed_max = batch_max
                self.initialized.fill_(True)
            else:
                momentum = self.momentum
                self.observed_min.mul_(momentum).add_(batch_min * (1.0 - momentum))
                self.observed_max.mul_(momentum).add_(batch_max * (1.0 - momentum))
            self._refresh_qparams()

        def _refresh_qparams(self) -> None:
            epsilon = torch.finfo(torch.float32).eps
            if self.config.symmetric:
                absolute_max = torch.maximum(self.observed_min.abs(), self.observed_max.abs())
                self.scale = torch.clamp(absolute_max / self.config.qmax, min=epsilon)
                self.zero_point = torch.zeros_like(self.scale, dtype=torch.int64)
            else:
                scale = (self.observed_max - self.observed_min) / (
                    self.config.qmax - self.config.qmin
                )
                self.scale = torch.clamp(scale, min=epsilon)
                zero_point = torch.round(self.config.qmin - self.observed_min / self.scale)
                self.zero_point = torch.clamp(
                    zero_point, self.config.qmin, self.config.qmax
                ).to(torch.int64)

        def _broadcast(self, values: Any) -> tuple[Any, Any]:
            if not self.config.per_channel:
                return self.scale, self.zero_point
            axis = self.config.channel_axis % values.ndim
            shape = [1] * values.ndim
            shape[axis] = self.scale.numel()
            return self.scale.reshape(shape), self.zero_point.reshape(shape)

        def quantize(self, values: Any) -> Any:
            scale, zero_point = self._broadcast(values)
            codes = torch.round(values / scale) + zero_point
            dtype = torch.int8 if self.config.symmetric else torch.uint8
            return torch.clamp(codes, self.config.qmin, self.config.qmax).to(dtype)

        def dequantize(self, values: Any) -> Any:
            scale, zero_point = self._broadcast(values)
            return (values.to(scale.dtype) - zero_point) * scale

        def forward(self, values: Any) -> Any:
            if self.observer_enabled:
                self._update_observer(values)
            if not self.fake_quant_enabled or not bool(self.initialized.item()):
                return values
            scale, zero_point = self._broadcast(values)
            codes = torch.clamp(
                torch.round(values / scale) + zero_point,
                self.config.qmin,
                self.config.qmax,
            )
            restored = (codes - zero_point) * scale
            return values + (restored - values).detach()

        def freeze_observer(self) -> None:
            self.observer_enabled = False

        def enable_observer(self) -> None:
            self.observer_enabled = True

        def disable_fake_quant(self) -> None:
            self.fake_quant_enabled = False

        def enable_fake_quant(self) -> None:
            self.fake_quant_enabled = True


    class QATLinear(nn.Module):
        """Linear layer with fake-quantized inputs, weights, and outputs."""

        def __init__(self, module: Any, config: QATConfig) -> None:
            super().__init__()
            self.in_features = module.in_features
            self.out_features = module.out_features
            self.weight = module.weight
            self.bias = module.bias
            self.input_fake_quant = TorchFakeQuantizer(
                QuantizationConfig(
                    bits=config.activation_bits,
                    symmetric=config.symmetric_activations,
                    narrow_range=True,
                ),
                momentum=config.observer_momentum,
            )
            self.weight_fake_quant = TorchFakeQuantizer(
                QuantizationConfig(
                    bits=config.weight_bits,
                    symmetric=True,
                    per_channel=config.per_channel_weights,
                    channel_axis=0,
                ),
                momentum=config.observer_momentum,
            )
            self.output_fake_quant = (
                TorchFakeQuantizer(
                    QuantizationConfig(
                        bits=config.activation_bits,
                        symmetric=config.symmetric_activations,
                        narrow_range=True,
                    ),
                    momentum=config.observer_momentum,
                )
                if config.quantize_output
                else nn.Identity()
            )

        def forward(self, values: Any) -> Any:
            result = functional.linear(
                self.input_fake_quant(values), self.weight_fake_quant(self.weight), self.bias
            )
            return self.output_fake_quant(result)


    class QATConv2d(nn.Module):
        """Conv2d layer with fake-quantized inputs, weights, and outputs."""

        def __init__(self, module: Any, config: QATConfig) -> None:
            super().__init__()
            self.in_channels = module.in_channels
            self.out_channels = module.out_channels
            self.kernel_size = module.kernel_size
            self.stride = module.stride
            self.padding = module.padding
            self.dilation = module.dilation
            self.groups = module.groups
            self.padding_mode = module.padding_mode
            self.weight = module.weight
            self.bias = module.bias
            self.input_fake_quant = TorchFakeQuantizer(
                QuantizationConfig(
                    bits=config.activation_bits,
                    symmetric=config.symmetric_activations,
                    narrow_range=True,
                ),
                momentum=config.observer_momentum,
            )
            self.weight_fake_quant = TorchFakeQuantizer(
                QuantizationConfig(
                    bits=config.weight_bits,
                    symmetric=True,
                    per_channel=config.per_channel_weights,
                    channel_axis=0,
                ),
                momentum=config.observer_momentum,
            )
            self.output_fake_quant = (
                TorchFakeQuantizer(
                    QuantizationConfig(
                        bits=config.activation_bits,
                        symmetric=config.symmetric_activations,
                        narrow_range=True,
                    ),
                    momentum=config.observer_momentum,
                )
                if config.quantize_output
                else nn.Identity()
            )

        def forward(self, values: Any) -> Any:
            if self.padding_mode != "zeros":
                raise ConfigurationError("QATConv2d currently supports zero padding mode only")
            result = functional.conv2d(
                self.input_fake_quant(values),
                self.weight_fake_quant(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            return self.output_fake_quant(result)


    class _PackedMixin:
        def _initialize_quantized_state(self, module: Any) -> None:
            if not bool(module.weight_fake_quant.initialized.item()):
                module.weight_fake_quant(module.weight)
            qweight = module.weight_fake_quant.quantize(module.weight.detach())
            self.register_buffer("qweight", qweight)
            self.register_buffer("weight_scale", module.weight_fake_quant.scale.detach().clone())
            self.register_buffer(
                "weight_zero_point", module.weight_fake_quant.zero_point.detach().clone()
            )
            input_quantizer = module.input_fake_quant
            if not bool(input_quantizer.initialized.item()):
                raise ConfigurationError(
                    "QAT conversion requires at least one representative/training forward pass"
                )
            self.register_buffer("input_scale", input_quantizer.scale.detach().clone())
            self.register_buffer("input_zero_point", input_quantizer.zero_point.detach().clone())
            self.qmin = input_quantizer.config.qmin
            self.qmax = input_quantizer.config.qmax
            output_quantizer = module.output_fake_quant
            self.output_quantized = isinstance(output_quantizer, TorchFakeQuantizer)
            if self.output_quantized:
                if not bool(output_quantizer.initialized.item()):
                    raise ConfigurationError(
                        "QAT output observer needs a representative/training forward pass"
                    )
                self.register_buffer("output_scale", output_quantizer.scale.detach().clone())
                self.register_buffer(
                    "output_zero_point", output_quantizer.zero_point.detach().clone()
                )
                self.output_qmin = output_quantizer.config.qmin
                self.output_qmax = output_quantizer.config.qmax

        def _dequantized_weight(self, dtype: Any) -> Any:
            shape = [self.weight_scale.numel()] + [1] * (self.qweight.ndim - 1)
            scale = self.weight_scale.reshape(shape)
            zero_point = self.weight_zero_point.reshape(shape)
            return ((self.qweight.to(scale.dtype) - zero_point) * scale).to(dtype)

        def _fake_quantized_input(self, values: Any) -> Any:
            codes = torch.clamp(
                torch.round(values / self.input_scale) + self.input_zero_point,
                self.qmin,
                self.qmax,
            )
            return ((codes - self.input_zero_point) * self.input_scale).to(values.dtype)

        def _fake_quantized_output(self, values: Any) -> Any:
            if not self.output_quantized:
                return values
            codes = torch.clamp(
                torch.round(values / self.output_scale) + self.output_zero_point,
                self.output_qmin,
                self.output_qmax,
            )
            return ((codes - self.output_zero_point) * self.output_scale).to(values.dtype)


    class PackedLinear(_PackedMixin, nn.Module):
        """INT8-weight linear module with a portable dequantizing reference kernel."""

        def __init__(self, module: QATLinear) -> None:
            super().__init__()
            self.in_features = module.in_features
            self.out_features = module.out_features
            self._initialize_quantized_state(module)
            if module.bias is None:
                self.register_buffer("bias", None)
            else:
                self.register_buffer("bias", module.bias.detach().clone())

        def forward(self, values: Any) -> Any:
            result = functional.linear(
                self._fake_quantized_input(values),
                self._dequantized_weight(values.dtype),
                self.bias,
            )
            return self._fake_quantized_output(result)


    class PackedConv2d(_PackedMixin, nn.Module):
        """INT8-weight Conv2d module with a portable dequantizing reference kernel."""

        def __init__(self, module: QATConv2d) -> None:
            super().__init__()
            self.stride = module.stride
            self.padding = module.padding
            self.dilation = module.dilation
            self.groups = module.groups
            self._initialize_quantized_state(module)
            if module.bias is None:
                self.register_buffer("bias", None)
            else:
                self.register_buffer("bias", module.bias.detach().clone())

        def forward(self, values: Any) -> Any:
            result = functional.conv2d(
                self._fake_quantized_input(values),
                self._dequantized_weight(values.dtype),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            return self._fake_quantized_output(result)


else:

    class _MissingTorch:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    TorchFakeQuantizer = _MissingTorch
    QATLinear = _MissingTorch
    QATConv2d = _MissingTorch
    PackedLinear = _MissingTorch
    PackedConv2d = _MissingTorch


def _replace_qat_modules(
    module: Any,
    config: QATConfig,
    *,
    prefix: str = "",
    counts: dict[str, int],
    excluded: list[str],
) -> None:
    for child_name, child in list(module.named_children()):
        qualified_name = f"{prefix}.{child_name}" if prefix else child_name
        if qualified_name in config.excluded_module_names:
            excluded.append(qualified_name)
            continue
        if isinstance(child, (QATLinear, QATConv2d, PackedLinear, PackedConv2d)):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, child_name, QATLinear(child, config))
            counts["linear"] += 1
        elif isinstance(child, nn.Conv2d):
            setattr(module, child_name, QATConv2d(child, config))
            counts["conv2d"] += 1
        else:
            _replace_qat_modules(
                child,
                config,
                prefix=qualified_name,
                counts=counts,
                excluded=excluded,
            )


def prepare_qat(
    model: Any,
    config: QATConfig | None = None,
    *,
    inplace: bool = False,
) -> tuple[Any, QATPreparationReport]:
    """Replace supported PyTorch layers with trainable fake-quantized modules."""

    _require_torch()
    prepared = model if inplace else copy.deepcopy(model)
    resolved = config or QATConfig()
    counts = {"linear": 0, "conv2d": 0}
    excluded: list[str] = []
    if isinstance(prepared, nn.Linear):
        prepared = QATLinear(prepared, resolved)
        counts["linear"] = 1
    elif isinstance(prepared, nn.Conv2d):
        prepared = QATConv2d(prepared, resolved)
        counts["conv2d"] = 1
    else:
        _replace_qat_modules(prepared, resolved, counts=counts, excluded=excluded)
    return prepared, QATPreparationReport(counts["linear"], counts["conv2d"], tuple(excluded))


def iter_fake_quantizers(model: Any) -> Iterable[Any]:
    _require_torch()
    return (module for module in model.modules() if isinstance(module, TorchFakeQuantizer))


def freeze_qat_observers(model: Any) -> int:
    """Freeze ranges after calibration/training stabilization."""

    quantizers = list(iter_fake_quantizers(model))
    for quantizer in quantizers:
        quantizer.freeze_observer()
    return len(quantizers)


def set_fake_quantization(model: Any, enabled: bool) -> int:
    quantizers = list(iter_fake_quantizers(model))
    for quantizer in quantizers:
        quantizer.enable_fake_quant() if enabled else quantizer.disable_fake_quant()
    return len(quantizers)


def _replace_packed_modules(module: Any) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, QATLinear):
            setattr(module, child_name, PackedLinear(child))
        elif isinstance(child, QATConv2d):
            setattr(module, child_name, PackedConv2d(child))
        else:
            _replace_packed_modules(child)


def convert_qat(model: Any, *, inplace: bool = False) -> Any:
    """Replace QAT modules with portable INT8-weight reference modules."""

    _require_torch()
    converted = model if inplace else copy.deepcopy(model)
    if isinstance(converted, QATLinear):
        converted = PackedLinear(converted)
    elif isinstance(converted, QATConv2d):
        converted = PackedConv2d(converted)
    else:
        _replace_packed_modules(converted)
    return converted


class TorchMagnitudePruner:
    """Apply the framework-neutral polynomial pruner to PyTorch parameters."""

    def __init__(
        self,
        schedule: PolynomialPruningSchedule | None = None,
        *,
        global_pruning: bool = True,
        parameter_names: Iterable[str] | None = None,
    ) -> None:
        _require_torch()
        self.pruner = MagnitudePruner(schedule, global_pruning=global_pruning)
        self.parameter_names = set(parameter_names) if parameter_names is not None else None
        self._torch_masks: dict[str, Any] = {}

    def _parameters(self, model: Any) -> dict[str, Any]:
        return {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.ndim >= 2
            and (self.parameter_names is None or name in self.parameter_names)
        }

    def update(self, model: Any, step: int) -> PruningStepResult | None:
        parameters = self._parameters(model)
        if not self.pruner.schedule.is_update_step(step):
            return None
        numpy_weights = {
            name: parameter.detach().cpu().numpy() for name, parameter in parameters.items()
        }
        result = self.pruner.update_masks(numpy_weights, step)
        self._torch_masks = {
            name: torch.as_tensor(mask, device=parameters[name].device)
            for name, mask in result.masks.items()
        }
        self.enforce(model)
        return result

    def enforce(self, model: Any) -> None:
        """Reapply masks after optimizer updates to prevent parameter regrowth."""

        parameters = self._parameters(model)
        with torch.no_grad():
            for name, mask in self._torch_masks.items():
                parameters[name].mul_(mask)

    def mask_gradients(self, model: Any) -> None:
        """Zero gradients for pruned parameters before an optimizer step."""

        parameters = self._parameters(model)
        for name, mask in self._torch_masks.items():
            if parameters[name].grad is not None:
                parameters[name].grad.mul_(mask)


def export_int8_bundle(model: Any, directory: str | Path) -> Path:
    """Export packed weights as NPZ files plus a vendor-neutral JSON manifest."""

    _require_torch()
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"format": "edge-opt-int8-v1", "modules": {}}
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, (PackedLinear, PackedConv2d)):
            continue
        prefix = name or "root"
        arrays[f"{prefix}.qweight"] = module.qweight.detach().cpu().numpy()
        arrays[f"{prefix}.weight_scale"] = module.weight_scale.detach().cpu().numpy()
        arrays[f"{prefix}.weight_zero_point"] = (
            module.weight_zero_point.detach().cpu().numpy()
        )
        if module.bias is not None:
            arrays[f"{prefix}.bias"] = module.bias.detach().cpu().numpy()
        manifest["modules"][name] = {
            "type": type(module).__name__,
            "qweight": f"{prefix}.qweight",
            "weight_scale": f"{prefix}.weight_scale",
            "weight_zero_point": f"{prefix}.weight_zero_point",
            "input_scale": module.input_scale.detach().cpu().tolist(),
            "input_zero_point": module.input_zero_point.detach().cpu().tolist(),
            "qmin": module.qmin,
            "qmax": module.qmax,
        }
    if not manifest["modules"]:
        raise ConfigurationError("model does not contain converted INT8 modules")
    np.savez(destination / "weights.npz", **arrays)
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
