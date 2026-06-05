# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
    Mxfp8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


# Online dispatch tables, keyed by the QuantSpec.weight QuantKey. The
# corresponding method class handles the activation choice via its
# `supported_activation_quant` set.
_ONLINE_LINEAR_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineLinearMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineLinearMethod,
    kMxfp8Dynamic: Mxfp8OnlineLinearMethod,
}

_ONLINE_MOE_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineMoEMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineMoEMethod,
    kMxfp8Dynamic: Mxfp8OnlineMoEMethod,
    kInt8StaticChannelSym: Int8OnlineMoEMethod,
}

_ONLINE_FP8_WEIGHT_KEYS = frozenset((kFp8StaticTensorSym, kFp8Static128BlockSym))


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


def _spec_uses_fp8_weight(spec: QuantSpec | None) -> bool:
    return spec is not None and spec.weight in _ONLINE_FP8_WEIGHT_KEYS


def _spec_uses_mxfp8_weight(spec: QuantSpec | None) -> bool:
    return spec is not None and spec.weight == kMxfp8Dynamic


def _spec_uses_mxfp4_weight(spec: QuantSpec | None) -> bool:
    return spec is not None and spec.weight == kMxfp4Dynamic


def _spec_uses_int8_moe_weight(spec: QuantSpec | None) -> bool:
    return spec is not None and spec.weight == kInt8StaticChannelSym


def _gb10_online_fp8_quantization_unsupported_reason(
    args: QuantizationConfigArgs,
) -> str | None:
    if not _is_sm12x_device():
        return None
    if not (_spec_uses_fp8_weight(args.linear) or _spec_uses_fp8_weight(args.moe)):
        return None
    return (
        "Online FP8 quantization is not supported on GB10/SM12x. The "
        "online dense path can select FP8 scaled-mm dense kernels and the "
        "online MoE path can reach generic FP8 MoE backend selection, but "
        "this is not native GB10 online FP8 correctness evidence. Use a "
        "native SM12x online FP8 dense/MoE path after correctness evidence "
        "exists, or keep online FP8 quantization unselected."
    )


def _gb10_online_mxfp8_quantization_unsupported_reason(
    args: QuantizationConfigArgs,
) -> str | None:
    if not _is_sm12x_device():
        return None
    if not (
        _spec_uses_mxfp8_weight(args.linear) or _spec_uses_mxfp8_weight(args.moe)
    ):
        return None
    return (
        "Online MXFP8 quantization is not supported on GB10/SM12x. The "
        "online dense path can select FlashInfer CUTLASS MXFP8 dense and the "
        "online MoE path can reach generic MXFP8 MoE backend selection, but "
        "this is not native GB10 online MXFP8 correctness evidence. Use a "
        "native SM12x online MXFP8 dense/MoE path after correctness evidence "
        "exists, or keep --quantization mxfp8 unselected."
    )


def _gb10_online_mxfp4_quantization_unsupported_reason(
    args: QuantizationConfigArgs,
) -> str | None:
    if not _is_sm12x_device():
        return None
    if not (
        _spec_uses_mxfp4_weight(args.linear) or _spec_uses_mxfp4_weight(args.moe)
    ):
        return None
    return (
        "Online MXFP4 quantization is not supported on GB10/SM12x. "
        "quantization_config can request weight='mxfp4' for dense or MoE "
        "online quantization, but no online MXFP4 method is wired and this is "
        "not native GB10 online MXFP4 correctness evidence. Use a native "
        "SM12x online MXFP4 path after correctness evidence exists, or keep "
        "weight='mxfp4' online quantization unselected."
    )


def _gb10_online_int8_moe_quantization_unsupported_reason(
    args: QuantizationConfigArgs,
) -> str | None:
    if not _is_sm12x_device():
        return None
    if not _spec_uses_int8_moe_weight(args.moe):
        return None
    return (
        "Online Int8 MoE quantization is not supported on GB10/SM12x. The "
        "int8_per_channel_weight_only shorthand can reach online Int8 MoE "
        "backend selection today, but this is not native GB10 online Int8 MoE "
        "correctness evidence. Use a native SM12x online Int8 MoE path after "
        "correctness evidence exists, or keep --quantization "
        "int8_per_channel_weight_only unselected."
    )


class OnlineQuantizationConfig(QuantizationConfig):
    """Model-level config for online quantization (quantize fp16/bf16 weights
    during model loading, without requiring a pre-quantized checkpoint)."""

    def __init__(
        self,
        args: QuantizationConfigArgs,
    ) -> None:
        super().__init__()
        if args.linear is None and args.moe is None:
            raise ValueError(
                "OnlineQuantizationConfig requires at least one of "
                "quantization_config.linear or quantization_config.moe "
                "to be set."
            )
        if reason := _gb10_online_fp8_quantization_unsupported_reason(args):
            raise ValueError(reason)
        if reason := _gb10_online_mxfp8_quantization_unsupported_reason(args):
            raise ValueError(reason)
        if reason := _gb10_online_mxfp4_quantization_unsupported_reason(args):
            raise ValueError(reason)
        if reason := _gb10_online_int8_moe_quantization_unsupported_reason(args):
            raise ValueError(reason)
        self.args = args
        self.ignored_layers: list[str] = args.ignore

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "online"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Note: as more online quant schemes will be added, this
        # value will become the minimum across all supported schemes.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineQuantizationConfig":
        raise NotImplementedError(
            "OnlineQuantizationConfig does not support loading from a "
            "checkpoint config. Use quantization_config or "
            "quantization='fp8_per_tensor'/'fp8_per_block' instead."
        )

    def _dispatch(
        self,
        spec: QuantSpec | None,
        table: dict[QuantKey, type],
        layer: torch.nn.Module,
    ) -> "QuantizeMethodBase | None":
        if spec is None or spec.weight is None:
            return None
        cls = table.get(spec.weight)
        if cls is None:
            raise ValueError(
                f"online quantization for {type(layer).__name__} with "
                f"weight={spec.weight} is not supported; supported weight "
                f"keys: {sorted(str(k) for k in table)}"
            )
        # Online method classes pick their own activation format internally.
        # Per-class activation overrides are not yet wired through; reject
        # explicit overrides until the relevant method class opts in.
        if spec.activation is not None:
            raise ValueError(
                f"activation override (activation={spec.activation}) is not "
                f"yet supported for online {cls.__name__}"
            )
        if isinstance(layer, RoutedExperts):
            return cls(layer=layer)
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            method = self._dispatch(self.args.linear, _ONLINE_LINEAR_METHODS, layer)
            return method if method is not None else UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            method = self._dispatch(self.args.moe, _ONLINE_MOE_METHODS, layer)
            return (
                method
                if method is not None
                else UnquantizedFusedMoEMethod(layer.moe_config)
            )
        return None
