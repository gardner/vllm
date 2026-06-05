# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import regex as re
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from torch.nn import Module

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform

_GB10_SM12X_UNSUPPORTED = "not supported on GB10/SM12x"

# Maps quantization strategy to the corresponding scale parameter type.
# Shared across compressed-tensor scheme classes (w8a16_fp8, w8a8_fp8, …).
STRATEGY_TO_PARAMETER_TYPE = {
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter,
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter,
    QuantizationStrategy.TENSOR: PerTensorScaleParameter,
}

# Maps quantization strategy to the vLLM weight-quant key used for
# kernel selection.  Shared across compressed-tensor scheme classes.
STRATEGY_TO_WEIGHT_QUANT_KEY = {
    QuantizationStrategy.BLOCK: kFp8Static128BlockSym,
    QuantizationStrategy.CHANNEL: kFp8StaticChannelSym,
    QuantizationStrategy.TENSOR: kFp8StaticTensorSym,
}


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value,
        CompressionFormat.nvfp4_pack_quantized.value,
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS


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


def _matches_quantization_type(
    quant: QuantizationArgs | None,
    expected_type: QuantizationType,
) -> bool:
    if quant is None:
        return False
    return quant.type == expected_type or quant.type == expected_type.value


def is_fp8_w4a8_quantization(
    weight_quant: QuantizationArgs | None,
    input_quant: QuantizationArgs | None,
) -> bool:
    if not _matches_quantization_type(
        weight_quant, QuantizationType.FLOAT
    ) or not _matches_quantization_type(input_quant, QuantizationType.FLOAT):
        return False

    assert weight_quant is not None
    assert input_quant is not None
    is_weight_4_bits = weight_quant.num_bits == 4
    is_activation_8_bits = input_quant.num_bits == 8
    weight_strategy = weight_quant.strategy == QuantizationStrategy.GROUP.value
    is_token = (
        weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
    )
    is_dynamic = not weight_quant.dynamic and input_quant.dynamic
    is_symmetric = weight_quant.symmetric and input_quant.symmetric
    return (
        is_weight_4_bits
        and is_activation_8_bits
        and is_token
        and is_symmetric
        and is_dynamic
    )


def gb10_compressed_tensors_w4a8_fp8_unsupported_reason(
    weight_quant: QuantizationArgs | None,
    input_quant: QuantizationArgs | None,
) -> str | None:
    if not _is_sm12x_device() or not is_fp8_w4a8_quantization(
        weight_quant, input_quant
    ):
        return None
    return (
        "CompressedTensors W4A8 FP8 checkpoint loading is "
        f"{_GB10_SM12X_UNSUPPORTED}. "
        "The available CompressedTensors W4A8 FP8 implementation "
        "uses exact-SM90 CUTLASS W4A8 kernels today, which can prove Hopper "
        "reachability but cannot satisfy native GB10 evidence. Use a native "
        "SM12x W4A8 FP8 backend after correctness evidence exists, or keep "
        "the path unselected."
    )


def should_ignore_layer(
    layer_name: str | None,
    ignore: Iterable[str] = tuple(),
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    if layer_name is None:
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj
    proj_name = layer_name.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping and layer_name not in ignore:
        shard_proj_names = fused_mapping[proj_name]

        # Convert fused_name --> [shard_names]
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.
        should_ignore_layer = None
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(
                layer_name=shard_name, targets=ignore
            )

            # If shard_idx=0, set layer ignore to match shard.
            if should_ignore_layer is None:
                should_ignore_layer = should_ignore_shard

            # If shard_idx=1+ confirm scheme matches prior shards.
            elif should_ignore_shard != should_ignore_layer:
                raise ValueError(
                    f"Found a different quantization schemes for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme."
                )

    # Unfused layers like down_proj and o_proj will match
    # the safetensors checkpoint already.
    else:
        should_ignore_layer = check_equal_or_regex_match(
            layer_name=layer_name, targets=ignore
        )

    assert should_ignore_layer is not None
    return should_ignore_layer


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    """
    return any(_is_equal_or_regex_match(layer_name, target) for target in targets)


def find_matched_target(
    layer_name: str | None,
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> str | None:
    """
    Helper function to look up which "target" in the compressed-tensors
    config that a layer corresponds to.

    Recall that a compressed-tensors configs has a concept of
    config_groups, where each layer can be quantized with a different
    scheme.

    targets in each config_group will be a list of either layer names
    (or regexes corresponding to layer names) or names of torch Modules.

    First, we try to match the layer_name with a target
    Second, we try to match the module's name with a target
    Third, we try to map the layer_name to a list of fused module names.
        *All* component module names must match in order for a match to be
        successful. A successful match returns the first component target

    :param layer_name: layer name
    :param module: torch.nn.Module
    :param targets: list of targets to match the layer against
    :param fused_mapping: map from fused layer names to its components
    :param fused_strategy: either "all" or "any". If using "all", fused
        layers match if "all" of its components match
    """

    if layer_name is None:
        layer_name = ""

    matched_target = (
        _find_first_match(layer_name, targets)
        or _find_first_match(module.__class__.__name__, targets, True)
        or _match_fused_layer(layer_name, targets, fused_mapping)
    )

    return matched_target


def _find_first_match(
    value: str, targets: Iterable[str], check_contains: bool = False
) -> str | None:
    """
    Returns first element of target that matches value either
    exactly or as a regex after 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.

    :param value: string to compare the list of targets against
    :param targets: list of targets to match the layer against
    :param check_contains: whether or not to do a substring match
    """

    for target in targets:
        if _is_equal_or_regex_match(value, target, check_contains=check_contains):
            return target
    return None


def _is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


def _match_fused_layer(
    layer_name: str,
    target_layers: Iterable[str],
    fused_mapping: Mapping[str, list[str]],
) -> str | None:
    """
    Match a fused layer name to its corresponding individual layer in
    target_layers. Returns first value in fused_mapping which matches targets

    Implements an "all" matching strategy where a fused layer matches iff
    "all" of its components match

    :param layer_name: layer name
    :param target_layers: list of targets to match the layer against
    :param fused_mapping: map from fused layer names to its components

    Examples:
        layer_name = "model.layers.0.self_attn.qkv_proj"
        target_layers = ["model.layers.0.self_attn.q_proj",
                        "model.layers.0.self_attn.k_proj",
                        "model.layers.0.self_attn.v_proj"]
    """
    # find layer_name in mapping
    fused = next((key for key in fused_mapping if layer_name.endswith(key)), None)
    if fused is None:
        return None

    # expand path of unfused components
    unfused_paths = [
        layer_name.replace(fused, unfused) for unfused in fused_mapping[fused]
    ]

    # for each unfused component, find a match in targets
    unfused_matches: list[str | None] = []
    for unfused in unfused_paths:
        for target in target_layers:
            if _is_equal_or_regex_match(unfused, target):
                unfused_matches.append(target)
                break
        else:
            unfused_matches.append(None)

    return unfused_matches[0] if all(unfused_matches) else None
