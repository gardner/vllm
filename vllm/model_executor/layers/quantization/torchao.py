# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import json
import types
from importlib.util import find_spec
from typing import Any

import regex as re
import torch
import torch.nn.functional as F
from packaging import version
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _bond_method_to_cls(func, obj):
    if hasattr(func, "__self__") or not callable(func):
        # If the function is already bound to an instance, return it as is
        return func
    else:
        return types.MethodType(func, obj)


def _get_weight_attrs(param):
    # record attributes attached to the weight, so we can
    # recover later
    recorded_weight_attr = {}
    for key in param.__dict__:
        if hasattr(param, key):
            attr = getattr(param, key)
            if not callable(attr):
                recorded_weight_attr[key] = attr
            elif hasattr(attr, "__self__") and param is attr.__self__:
                # if attr is a bonded method for an instance, and
                # attr.__self__ points to the instance (param)
                # we'll record the underlying function object
                recorded_weight_attr[key] = attr.__func__
            else:
                recorded_weight_attr[key] = attr
    return recorded_weight_attr


def _restore_weight_attrs(param, recorded_weight_attr):
    for attr_name, attr in recorded_weight_attr.items():
        if not hasattr(param, attr_name):
            setattr(param, attr_name, _bond_method_to_cls(attr, param))


def torchao_version_at_least(torchao_version: str) -> bool:
    if find_spec("torchao"):
        try:
            if version.parse(importlib.metadata.version("torchao")) >= version.parse(
                torchao_version
            ):
                return True
        except (ImportError, version.InvalidVersion):
            return False
    return False


def should_skip(prefix: str, skip_modules: list[str]) -> bool:
    """
    Robust skipping logic:
    should_skip("model.model.layers.1.q_proj",
                ["model.model.layers.1.q_proj"])  # True
    should_skip("model.model.layers.10.o_proj", ["o_proj"])  -> True
    should_skip("visual.model.layers.1.q_proj", ["visual"])   -> True
    should_skip("model.model.layers.1.q_proj", ["layers.1"])  -> True
    should_skip("model.model.layers.11.q_proj", ["layers.1"]) -> False
    """
    for s in skip_modules:
        if prefix == s:
            return True
        if f".{s}." in f".{prefix}.":
            return True
    return False


if torchao_version_at_least("0.15.0"):
    from torchao.prototype.tensor_conversion.api import (
        convert_to_packed_tensor_based_on_current_hardware,
    )
else:
    convert_to_packed_tensor_based_on_current_hardware = lambda t: t


def _is_sm12x_device() -> bool:
    is_family = getattr(current_platform, "is_device_capability_family", None)
    if callable(is_family):
        result = is_family(120)
        if isinstance(result, bool):
            return result

    get_device_capability = getattr(current_platform, "get_device_capability", None)
    if callable(get_device_capability):
        capability = get_device_capability()
        major = getattr(capability, "major", None)
        if isinstance(major, int):
            return major == 12
        if isinstance(capability, tuple) and capability:
            return capability[0] == 12

    return False


def _is_torchao_fp8_activation_config(torchao_config: Any) -> bool:
    config_name = type(torchao_config).__name__
    return "Float8" in config_name and "Activation" in config_name


def _gb10_torchao_fp8_activation_unsupported_reason(
    torchao_config: Any,
) -> str | None:
    if not _is_sm12x_device() or not _is_torchao_fp8_activation_config(
        torchao_config
    ):
        return None
    config_name = type(torchao_config).__name__
    return (
        "TorchAO FP8 activation quantization is not supported on GB10/SM12x. "
        f"The torchao quantization method can use {config_name}, call "
        "torchao.quantization.quantize_, and convert weights with "
        "convert_to_packed_tensor_based_on_current_hardware today, but this is "
        "not native GB10 TorchAO FP8 activation correctness evidence. Use a "
        "validated GB10 TorchAO FP8 activation path after native SM12x "
        "correctness evidence exists, or keep FP8 activation TorchAO configs "
        "unselected."
    )


def _gb10_torchao_weight_quantization_unsupported_reason(
    torchao_config: Any,
) -> str | None:
    if not _is_sm12x_device() or _is_torchao_fp8_activation_config(torchao_config):
        return None
    config_name = type(torchao_config).__name__
    return (
        "TorchAO weight quantization is not supported on GB10/SM12x. The "
        f"torchao quantization method can use {config_name}, call "
        "torchao.quantization.quantize_, and convert weights with "
        "convert_to_packed_tensor_based_on_current_hardware today, but this is "
        "not native GB10 TorchAO weight correctness evidence. Use a validated "
        "GB10 TorchAO weight path after native SM12x correctness evidence "
        "exists, or keep TorchAO weight quantization configs unselected."
    )


def _check_torchao_fp8_activation_capability(torchao_config) -> None:
    """Check if the current GPU supports FP8 activation quantization.

    FP8 activation configs (e.g., Float8DynamicActivationFloat8WeightConfig)
    require GPU compute capability >= 8.9 (Ada Lovelace / Hopper) on NVIDIA,
    or MI300+ on AMD. This check provides a clear error message before
    torchao's internal assertion fires with a confusing message.
    """
    config_name = type(torchao_config).__name__
    if "Float8" not in config_name or "Activation" not in config_name:
        return

    if reason := _gb10_torchao_fp8_activation_unsupported_reason(torchao_config):
        raise ValueError(reason)

    if current_platform.supports_fp8():
        return

    capability = current_platform.get_device_capability()
    capability_str = (
        f" (current GPU compute capability: {capability.major}.{capability.minor})"
        if capability is not None
        else ""
    )
    raise ValueError(
        f"torchao FP8 activation quantization config '{config_name}' "
        f"requires GPU compute capability >= 8.9 (e.g., NVIDIA Ada Lovelace "
        f"/ Hopper or AMD MI300+){capability_str}. "
        f"For older GPUs, consider using a non-FP8 config such as "
        f"Int8WeightOnlyConfig or Int4WeightOnlyConfig."
    )


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao."""

    def __init__(
        self,
        torchao_config,
        skip_modules: list[str] | None = None,
        is_checkpoint_torchao_serialized: bool = False,
    ) -> None:
        super().__init__()
        if reason := _gb10_torchao_fp8_activation_unsupported_reason(torchao_config):
            raise ValueError(reason)
        if reason := _gb10_torchao_weight_quantization_unsupported_reason(
            torchao_config
        ):
            raise ValueError(reason)
        self.torchao_config = torchao_config
        self.skip_modules = skip_modules or []
        self.is_checkpoint_torchao_serialized = is_checkpoint_torchao_serialized

    def __repr__(self) -> str:
        return (
            f"TorchAOConfig({self.torchao_config=}, {self.skip_modules=}, "
            f"{self.is_checkpoint_torchao_serialized=})"
        )

    def get_name(self) -> QuantizationMethods:
        return "torchao"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        """torchao doesn't require additional config files, we use
        `config.json` from huggingface: `model_config.hf_config`
        """
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchAOConfig":
        """Create the quant config from an hf model config"""
        try:
            from torchao.core.config import config_from_dict
        except ImportError as err:
            raise ImportError(
                "Please install torchao>=0.10.0 via "
                "`pip install torchao>=0.10.0` to use torchao quantization."
            ) from err

        quant_method = cls.get_from_keys_or(config, ["quant_method"], None)
        is_checkpoint_torchao_serialized = (
            quant_method is not None and "torchao" in quant_method
        )

        hf_config = cls.get_from_keys_or(config, ["quant_type"], None)
        assert hf_config is not None, "quant_type must be specified"
        assert len(hf_config) == 1 and "default" in hf_config, (
            "Expected only one key 'default' in quant_type dictionary"
        )
        quant_type = hf_config["default"]
        ao_config = config_from_dict(quant_type)

        # Adds skipped modules defined in "modules_to_not_convert"
        skip_modules = config.get("modules_to_not_convert", []) or []

        # Adds skipped modules defined in "module_fqn_to_config"
        _data = quant_type.get("_data", {})
        if not isinstance(_data, dict):
            _data = {}

        module_fqn = _data.get("module_fqn_to_config", {})
        if not isinstance(module_fqn, dict):
            module_fqn = {}

        for layer, layer_cfg in module_fqn.items():
            if layer_cfg is None:
                skip_modules.append(layer)

        return cls(ao_config, skip_modules, is_checkpoint_torchao_serialized)

    @classmethod
    def from_config_file(cls, config_file: str) -> "TorchAOConfig":
        """Initialize class from a config file. Example:
        ```
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        fn = "torchao_config.json"

        with open(fn, "w") as f:
            f.write(json.dumps(config_to_dict(config)))
        ```
        """
        with open(config_file) as f:
            f.seek(0)
            f_read = f.read()
            config_dict = json.loads(f_read)

        hf_config = {"quant_type": {"default": config_dict}}
        return cls.from_config(hf_config)

    @classmethod
    def from_config_dict_json(cls, config_dict_json: str) -> "TorchAOConfig":
        """Initialize class from a config_dict json string, got from
        torchao_config_object = some AOBaseConfig object
        json.dumps(config_to_dict(torchao_config_object))
        """
        config_dict = json.loads(config_dict_json)
        hf_config = {"quant_type": {"default": config_dict}}
        return cls.from_config(hf_config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if not isinstance(layer, LinearBase):
            return None

        from torchao.quantization import ModuleFqnToConfig

        if should_skip(prefix, self.skip_modules):
            return UnquantizedLinearMethod()

        module_fqn = prefix
        if isinstance(self.torchao_config, ModuleFqnToConfig):
            module_fqn_to_config = self.torchao_config.module_fqn_to_config
            c = None
            if module_fqn in module_fqn_to_config:
                assert not module_fqn.startswith("re:"), (
                    "module fqn should not start with"
                    "`re:`, which is used for specifying regex"
                )
                c = module_fqn_to_config[module_fqn]
            else:
                for maybe_module_fqn_pattern in module_fqn_to_config:
                    if not maybe_module_fqn_pattern.startswith("re:"):
                        continue
                    elif re.fullmatch(maybe_module_fqn_pattern[3:], module_fqn):
                        # we'll apply the config for first fully matched pattern
                        c = module_fqn_to_config[maybe_module_fqn_pattern]
                        break
                else:
                    # fallback to use default if no module specific
                    # config is provided
                    c = module_fqn_to_config.get("_default", None)

            if c is not None:
                current_torchao_config = TorchAOConfig(
                    c, self.skip_modules, self.is_checkpoint_torchao_serialized
                )
                return TorchAOLinearMethod(current_torchao_config)
            else:
                return UnquantizedLinearMethod()

        return TorchAOLinearMethod(self)

    def get_scaled_act_names(self) -> list[str]:
        return []


def torchao_quantize_param_data(
    param: torch.Tensor, torchao_config: Any
) -> torch.nn.Parameter:
    """Quantize a Tensor with torchao quantization specified by torchao_config

    Args:
        param: weight parameter of the linear module
        torchao_config: type of quantization and their arguments we want to
            use to quantize the Tensor
    """
    from torchao.core.config import AOBaseConfig
    from torchao.quantization import quantize_

    assert isinstance(torchao_config, AOBaseConfig), f"{torchao_config}"
    _check_torchao_fp8_activation_capability(torchao_config)
    """
    Avoid real weight allocation for faster load, since we will
    end up setting it to param.
    """
    with torch.device("meta"):
        # linear can't be top level module since quantize_ is inplace
        # while some of our configs need to do module swap, and only non-top
        # level modules support module swap
        dummy_linear = torch.nn.Sequential(
            torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
        )

    dummy_linear[0].weight = param
    quantize_(dummy_linear, torchao_config)
    return dummy_linear[0].weight


class TorchAOLinearMethod(LinearMethodBase):
    """Linear method for torchao.

    Args:
        quant_config: The torchao quantization config, a string that encodes
            the type of quantization and all relevant arguments.
    """

    def __init__(self, quant_config: TorchAOConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        if self.quant_config.is_checkpoint_torchao_serialized:
            weight = torchao_quantize_param_data(
                weight, self.quant_config.torchao_config
            )

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_config.is_checkpoint_torchao_serialized:
            if not hasattr(layer, "weight"):
                return

            # record attributes attached to the weight, so we can
            # recover later
            recorded_weight_attr = _get_weight_attrs(layer.weight)

            layer.weight = Parameter(
                convert_to_packed_tensor_based_on_current_hardware(layer.weight),
                requires_grad=layer.weight.requires_grad,
            )

            _restore_weight_attrs(layer.weight, recorded_weight_attr)
            return

        # online quantize the weight if the checkpoint is not already
        # quantized by torchao
        recorded_weight_attr = _get_weight_attrs(layer.weight)

        weight = torchao_quantize_param_data(
            layer.weight, self.quant_config.torchao_config
        )
        weight = torch.nn.Parameter(
            convert_to_packed_tensor_based_on_current_hardware(weight),
            weight.requires_grad,
        )

        _restore_weight_attrs(weight, recorded_weight_attr)
        layer.register_parameter("weight", weight)
