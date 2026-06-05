# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)


class _SupportedExperts:
    @staticmethod
    def is_supported_config(*_args, **_kwargs):
        return True, None


class _UnsupportedExperts:
    @staticmethod
    def is_supported_config(*_args, **_kwargs):
        return False, "mock unsupported"


class _Sm12xPlatform:
    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def is_rocm() -> bool:
        return False

    @staticmethod
    def is_xpu() -> bool:
        return False

    @staticmethod
    def is_cpu() -> bool:
        return False

    @staticmethod
    def is_device_capability(capability: int) -> bool:
        return capability in {120, 121}

    @staticmethod
    def is_device_capability_family(capability: int) -> bool:
        return capability in {100, 120}


def _make_mxfp4_moe_config(
    *,
    moe_backend: str = "auto",
    use_batched_activation_format: bool = False,
) -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig.make_no_parallel()
    if use_batched_activation_format:
        parallel_config.use_ep = True
        parallel_config.dp_size = 2
        parallel_config.all2all_backend = "nixl_ep"

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
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        moe_backend=moe_backend,
    )


def _mock_sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mxfp4_oracle, "current_platform", _Sm12xPlatform())


def _mock_vllm_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mxfp4_oracle,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(quantization_config=None),
        ),
    )


def _mock_backend_support(
    monkeypatch: pytest.MonkeyPatch,
    supported_backends: set[Mxfp4MoeBackend],
) -> dict[Mxfp4MoeBackend, type]:
    def _experts_base(backend: Mxfp4MoeBackend) -> type:
        if backend in supported_backends:
            return _SupportedExperts
        return _UnsupportedExperts

    kernel_by_backend = {
        backend: type(
            f"{backend.value}Experts",
            (_experts_base(backend),),
            {},
        )
        for backend in Mxfp4MoeBackend
    }

    monkeypatch.setattr(
        mxfp4_oracle,
        "backend_to_kernel_cls",
        lambda backend: [kernel_by_backend[backend]],
    )
    return kernel_by_backend


@pytest.mark.parametrize("moe_backend", ["marlin", "emulation", "cpu"])
def test_gb10_explicit_fallback_mxfp4_moe_rejected(monkeypatch, moe_backend):
    _mock_sm12x_platform(monkeypatch)
    _mock_vllm_config(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {
            Mxfp4MoeBackend.MARLIN,
            Mxfp4MoeBackend.EMULATION,
            Mxfp4MoeBackend.CPU,
        },
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        select_mxfp4_moe_backend(
            _make_mxfp4_moe_config(moe_backend=moe_backend),
        )


def test_gb10_explicit_batched_marlin_mxfp4_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_vllm_config(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {Mxfp4MoeBackend.BATCHED_MARLIN},
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        select_mxfp4_moe_backend(
            _make_mxfp4_moe_config(
                moe_backend="marlin",
                use_batched_activation_format=True,
            ),
        )


def test_gb10_auto_mxfp4_moe_rejects_fallback_when_no_native_backend(
    monkeypatch,
):
    _mock_sm12x_platform(monkeypatch)
    _mock_vllm_config(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {
            Mxfp4MoeBackend.MARLIN,
            Mxfp4MoeBackend.BATCHED_MARLIN,
            Mxfp4MoeBackend.EMULATION,
            Mxfp4MoeBackend.CPU,
        },
    )

    with pytest.raises(NotImplementedError, match="fallback.*not supported"):
        select_mxfp4_moe_backend(_make_mxfp4_moe_config())


def test_gb10_auto_mxfp4_moe_keeps_flashinfer_cutlass_when_supported(
    monkeypatch,
):
    _mock_sm12x_platform(monkeypatch)
    _mock_vllm_config(monkeypatch)
    kernel_by_backend = _mock_backend_support(
        monkeypatch,
        {
            Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
            Mxfp4MoeBackend.MARLIN,
        },
    )

    backend, experts_cls = select_mxfp4_moe_backend(_make_mxfp4_moe_config())

    assert backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16
    assert experts_cls is kernel_by_backend[
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16
    ]


def test_gb10_env_forced_marlin_mxfp4_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_vllm_config(monkeypatch)
    monkeypatch.setenv("VLLM_MXFP4_USE_MARLIN", "1")
    _mock_backend_support(
        monkeypatch,
        {Mxfp4MoeBackend.MARLIN},
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        select_mxfp4_moe_backend(_make_mxfp4_moe_config())


@pytest.mark.parametrize("moe_backend", ["marlin", "emulation", "cpu"])
def test_gb10_explicit_deepseek_v4_mxfp4_moe_fallback_rejected(
    monkeypatch,
    moe_backend,
):
    _mock_sm12x_platform(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {
            Mxfp4MoeBackend.MARLIN,
            Mxfp4MoeBackend.EMULATION,
            Mxfp4MoeBackend.CPU,
        },
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        select_deepseek_v4_mxfp4_moe_backend(
            _make_mxfp4_moe_config(moe_backend=moe_backend),
        )


def test_gb10_auto_deepseek_v4_mxfp4_moe_rejects_fallback_when_no_native_backend(
    monkeypatch,
):
    _mock_sm12x_platform(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {
            Mxfp4MoeBackend.MARLIN,
            Mxfp4MoeBackend.BATCHED_MARLIN,
            Mxfp4MoeBackend.EMULATION,
            Mxfp4MoeBackend.CPU,
        },
    )

    with pytest.raises(NotImplementedError, match="fallback.*not supported"):
        select_deepseek_v4_mxfp4_moe_backend(_make_mxfp4_moe_config())
