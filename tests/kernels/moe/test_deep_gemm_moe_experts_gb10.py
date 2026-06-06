# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.layers.fused_moe.config as fused_moe_config
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.config import (
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts import (
    batched_deep_gemm_moe,
    deep_gemm_moe,
    triton_deep_gemm_moe,
)
from vllm.platforms import PlatformEnum


class _Sm12xCudaPlatform:
    _enum = PlatformEnum.CUDA

    def fp8_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def get_device_capability(self) -> tuple[int, int]:
        return (12, 1)

    def is_cuda(self) -> bool:
        return True

    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


def _patch_sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _Sm12xCudaPlatform()
    monkeypatch.setattr(fused_moe_config, "current_platform", platform)
    monkeypatch.setattr(deep_gemm_moe, "current_platform", platform)
    monkeypatch.setattr(batched_deep_gemm_moe, "current_platform", platform)
    monkeypatch.setattr(triton_deep_gemm_moe, "current_platform", platform)


def _make_fp8_quant_config():
    ones = torch.ones(1, dtype=torch.float32)
    return fp8_w8a8_moe_quant_config(
        w1_scale=ones,
        w2_scale=ones,
        a1_scale=ones,
        a2_scale=ones,
        block_shape=[128, 128],
    )


def test_sm12x_deep_gemm_experts_constructor_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    def _should_not_be_called():
        raise AssertionError("GB10 should reject DeepGEMM MoE before setup")

    monkeypatch.setattr(
        deep_gemm_moe,
        "get_mk_alignment_for_contiguous_layout",
        _should_not_be_called,
    )

    with pytest.raises(ValueError, match="DeepGEMM MoE.*GB10/SM12x"):
        deep_gemm_moe.DeepGemmExperts(
            moe_config=make_dummy_moe_config(),
            quant_config=_make_fp8_quant_config(),
        )


def test_sm12x_deep_gemm_fp4_experts_constructor_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    with pytest.raises(ValueError, match="DeepGEMM MoE.*GB10/SM12x"):
        deep_gemm_moe.DeepGemmFP4Experts(
            moe_config=make_dummy_moe_config(),
            quant_config=_make_fp8_quant_config(),
        )


def test_sm12x_batched_deep_gemm_experts_constructor_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    def _should_not_be_called():
        raise AssertionError("GB10 should reject batched DeepGEMM MoE before setup")

    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "get_mk_alignment_for_contiguous_layout",
        _should_not_be_called,
    )

    with pytest.raises(ValueError, match="DeepGEMM MoE.*GB10/SM12x"):
        batched_deep_gemm_moe.BatchedDeepGemmExperts(
            moe_config=make_dummy_moe_config(),
            quant_config=_make_fp8_quant_config(),
            max_num_tokens=16,
            num_dispatchers=1,
        )


def test_sm12x_triton_or_deep_gemm_experts_constructor_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError(
            "GB10 should reject TritonOrDeepGemm MoE before building experts"
        )

    monkeypatch.setattr(
        triton_deep_gemm_moe,
        "DeepGemmExperts",
        _should_not_be_called,
    )
    monkeypatch.setattr(
        triton_deep_gemm_moe,
        "TritonExperts",
        _should_not_be_called,
    )

    with pytest.raises(ValueError, match="DeepGEMM MoE.*GB10/SM12x"):
        triton_deep_gemm_moe.TritonOrDeepGemmExperts(
            moe_config=make_dummy_moe_config(),
            quant_config=_make_fp8_quant_config(),
        )
