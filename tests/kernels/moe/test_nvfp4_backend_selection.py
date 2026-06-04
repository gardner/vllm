# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as nvfp4_oracle
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms.interface import DeviceCapability


class _SupportedExperts:
    @staticmethod
    def is_supported_config(*_args, **_kwargs):
        return True, None


class _UnsupportedExperts:
    @staticmethod
    def is_supported_config(*_args, **_kwargs):
        return False, "mock unsupported"


def _make_nvfp4_moe_config(
    *,
    moe_backend: str = "auto",
    swiglu_limit: float | None = None,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size_per_partition=256,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        moe_backend=moe_backend,
        swiglu_limit=swiglu_limit,
    )


def _mock_sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nvfp4_oracle.current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(12, 1),
    )


def _mock_backend_support(
    monkeypatch: pytest.MonkeyPatch,
    supported_backends: set[NvFp4MoeBackend],
) -> dict[NvFp4MoeBackend, type]:
    def _experts_base(backend: NvFp4MoeBackend) -> type:
        if backend in supported_backends:
            return _SupportedExperts
        return _UnsupportedExperts

    kernel_by_backend = {
        backend: type(
            f"{backend.value}Experts",
            (_experts_base(backend),),
            {},
        )
        for backend in NvFp4MoeBackend
    }

    monkeypatch.setattr(
        nvfp4_oracle,
        "backend_to_kernel_cls",
        lambda backend: [kernel_by_backend[backend]],
    )
    return kernel_by_backend


def test_gb10_auto_nvfp4_moe_skips_trtllm_gen_when_cutlass_supported(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    kernel_by_backend = _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, experts_cls = select_nvfp4_moe_backend(
        _make_nvfp4_moe_config(),
        weight_key=kNvfp4Static,
        activation_key=kNvfp4Dynamic,
    )

    assert backend == NvFp4MoeBackend.FLASHINFER_CUTLASS
    assert experts_cls is kernel_by_backend[NvFp4MoeBackend.FLASHINFER_CUTLASS]


def test_gb10_swiglu_limit_nvfp4_moe_skips_trtllm_gen_for_cutlass(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    kernel_by_backend = _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, experts_cls = select_nvfp4_moe_backend(
        _make_nvfp4_moe_config(swiglu_limit=7.0),
        weight_key=kNvfp4Static,
        activation_key=kNvfp4Dynamic,
    )

    assert backend == NvFp4MoeBackend.FLASHINFER_CUTLASS
    assert experts_cls is kernel_by_backend[NvFp4MoeBackend.FLASHINFER_CUTLASS]


def test_gb10_explicit_b12x_with_swiglu_limit_recommends_cutlass_only(
    monkeypatch,
):
    _mock_sm12x_platform(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {NvFp4MoeBackend.FLASHINFER_B12X},
    )

    with pytest.raises(ValueError) as exc_info:
        select_nvfp4_moe_backend(
            _make_nvfp4_moe_config(
                moe_backend="flashinfer_b12x",
                swiglu_limit=7.0,
            ),
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )

    message = str(exc_info.value)
    assert "does not apply the SwiGLU clamp" in message
    assert "flashinfer_cutlass" in message
    assert "flashinfer_trtllm" not in message


def test_gb10_explicit_trtllm_nvfp4_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {NvFp4MoeBackend.FLASHINFER_TRTLLM},
    )

    with pytest.raises(ValueError, match="TRTLLM Gen MoE is not supported"):
        select_nvfp4_moe_backend(
            _make_nvfp4_moe_config(moe_backend="flashinfer_trtllm"),
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )


@pytest.mark.parametrize("moe_backend", ["marlin", "emulation"])
def test_gb10_explicit_fallback_nvfp4_moe_rejected(monkeypatch, moe_backend):
    _mock_sm12x_platform(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {NvFp4MoeBackend.MARLIN, NvFp4MoeBackend.EMULATION},
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        select_nvfp4_moe_backend(
            _make_nvfp4_moe_config(moe_backend=moe_backend),
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )


def test_gb10_auto_nvfp4_moe_rejects_fallback_when_no_native_backend(
    monkeypatch,
):
    _mock_sm12x_platform(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {NvFp4MoeBackend.MARLIN, NvFp4MoeBackend.EMULATION},
    )

    with pytest.raises(NotImplementedError, match="fallback.*not supported"):
        select_nvfp4_moe_backend(
            _make_nvfp4_moe_config(),
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )


def test_gb10_flashinfer_env_auto_nvfp4_moe_skips_trtllm_gen(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    kernel_by_backend = _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, experts_cls = select_nvfp4_moe_backend(
        _make_nvfp4_moe_config(),
        weight_key=kNvfp4Static,
        activation_key=kNvfp4Dynamic,
    )

    assert backend == NvFp4MoeBackend.FLASHINFER_CUTLASS
    assert experts_cls is kernel_by_backend[NvFp4MoeBackend.FLASHINFER_CUTLASS]


def test_gb10_flashinfer_env_auto_nvfp4_moe_prefers_b12x(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    kernel_by_backend = _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.FLASHINFER_B12X,
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, experts_cls = select_nvfp4_moe_backend(
        _make_nvfp4_moe_config(),
        weight_key=kNvfp4Static,
        activation_key=kNvfp4Dynamic,
    )

    assert backend == NvFp4MoeBackend.FLASHINFER_B12X
    assert experts_cls is kernel_by_backend[NvFp4MoeBackend.FLASHINFER_B12X]


def test_gb10_flashinfer_env_explicit_trtllm_nvfp4_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    monkeypatch.setattr(
        nvfp4_oracle,
        "get_flashinfer_moe_backend",
        lambda: FlashinferMoeBackend.TENSORRT_LLM,
    )
    _mock_backend_support(
        monkeypatch,
        {NvFp4MoeBackend.FLASHINFER_TRTLLM},
    )

    with pytest.raises(ValueError, match="TRTLLM Gen MoE is not supported"):
        select_nvfp4_moe_backend(
            _make_nvfp4_moe_config(),
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )


def test_gb10_auto_nvfp4_moe_reports_trtllm_gen_rejection(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {NvFp4MoeBackend.FLASHINFER_TRTLLM},
    )

    with pytest.raises(NotImplementedError, match="TRTLLM Gen MoE is not supported"):
        select_nvfp4_moe_backend(
            _make_nvfp4_moe_config(),
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )
