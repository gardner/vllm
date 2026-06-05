# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int8_w8a8_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Int8MoeBackend(Enum):
    TRITON = "TRITON"


_GB10_SM12X_UNSUPPORTED = "not supported on GB10/SM12x"


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


def _gb10_int8_moe_triton_unsupported_reason(
    backend: Int8MoeBackend,
) -> str | None:
    if backend != Int8MoeBackend.TRITON or not _is_sm12x_device():
        return None
    return (
        f"Int8 MoE Triton fallback backend '{backend.value}' is "
        f"{_GB10_SM12X_UNSUPPORTED}. "
        "The generic Triton Int8 MoE path can prove reachability, "
        "but it is not native GB10 Int8 MoE evidence. Use a native SM12x "
        "Int8 MoE backend after correctness evidence exists, or keep the path "
        "unselected."
    )


def _get_priority_backends(
    moe_config: FusedMoEConfig,
) -> list[Int8MoeBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    return [Int8MoeBackend.TRITON]


def backend_to_kernel_cls(
    backend: Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Int8MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    else:
        raise ValueError(f"Unknown Int8 MoE backend: {backend.value}")


def map_int8_backend(runner_backend: MoEBackend) -> Int8MoeBackend:
    """Map user's MoEBackend to Int8MoeBackend."""
    mapping = {
        "triton": Int8MoeBackend.TRITON,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for Int8 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt8StaticChannelSym,
    activation_key: QuantKey | None = kInt8DynamicTokenSym,
) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
    """
    Select the primary Int8 MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    AVAILABLE_BACKENDS = _get_priority_backends(config)

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Int8MoeBackend) -> str:
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Int8 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Int8MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"Int8 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"Int8 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _append_unique_reason(reasons: list[str], reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    def _unsupported_suffix(reasons: list[str]) -> str:
        if not reasons:
            return ""
        return " " + " ".join(reasons)

    def _return_or_raise(
        backend: Int8MoeBackend,
    ) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
        if reason := _gb10_int8_moe_triton_unsupported_reason(backend):
            raise ValueError(_make_log_unsupported(backend, reason))

        reason: str | None = None
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_int8_backend(runner_backend)
        return _return_or_raise(requested_backend)

    # Select kernels in order of backend.
    unavailable_gb10_reasons: list[str] = []
    for backend in AVAILABLE_BACKENDS:
        if reason := _gb10_int8_moe_triton_unsupported_reason(backend):
            _append_unique_reason(unavailable_gb10_reasons, reason)
            logger.debug_once(_make_log_unsupported(backend, reason))
            continue

        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No Int8 MoE backend supports the deployment configuration."
        + _unsupported_suffix(unavailable_gb10_reasons)
    )


def make_int8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
) -> FusedMoEQuantConfig:
    assert (a1_scale is None and a2_scale is None) or (
        a1_scale is not None and a2_scale is not None
    ), "a1_scale and a2_scale must both be provided or both be None"

    if a1_scale is None or a2_scale is None:
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=None,
            w2_zp=None,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    return int8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
    )


def make_int8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=issubclass(experts_cls, mk.FusedMoEExpertsMonolithic),
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )

    return kernel
