# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)
from vllm.platforms import current_platform


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


def _gb10_experts_int8_quantization_unsupported_reason() -> str | None:
    if not _is_sm12x_device():
        return None
    return (
        "ExpertsInt8 quantization is not supported on GB10/SM12x. The "
        "backward-compatible public quantization method can reach "
        "online Int8 MoE backend selection today, but this is not native GB10 "
        "online Int8 MoE correctness evidence. Use a validated GB10 Int8 "
        "MoE path after native SM12x correctness evidence exists, or keep "
        "--quantization experts_int8 unselected."
    )


class ExpertsInt8Config(QuantizationConfig):
    """Online int8 quantization for MoE expert weights.
    Linear layers are left unquantized.

    Backward-compatible config for ``--quantization experts_int8``.
    Prefer ``--quantization int8_per_channel``
    """

    def __init__(self) -> None:
        super().__init__()
        if reason := _gb10_experts_int8_quantization_unsupported_reason():
            raise ValueError(reason)

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "experts_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExpertsInt8Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return Int8OnlineMoEMethod(layer=layer)
        return None
