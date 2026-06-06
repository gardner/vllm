# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.layers.fused_moe.config as fused_moe_config
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    int4_w4a16_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts import (
    trtllm_bf16_moe,
    trtllm_fp8_moe,
    trtllm_mxfp4_moe,
    trtllm_mxint4_moe,
    trtllm_nvfp4_moe,
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
    monkeypatch.setattr(trtllm_bf16_moe, "current_platform", platform)
    monkeypatch.setattr(trtllm_fp8_moe, "current_platform", platform)
    monkeypatch.setattr(trtllm_mxint4_moe, "current_platform", platform)
    monkeypatch.setattr(trtllm_mxfp4_moe, "current_platform", platform)
    monkeypatch.setattr(trtllm_nvfp4_moe, "current_platform", platform)


def _make_fp8_quant_config():
    ones = torch.ones(1, dtype=torch.float32)
    return fp8_w8a8_moe_quant_config(
        w1_scale=ones,
        w2_scale=ones,
        a1_scale=ones,
        a2_scale=ones,
    )


def _make_mxfp4_quant_config():
    ones = torch.ones(1, dtype=torch.float32)
    return mxfp4_w4a16_moe_quant_config(
        w1_scale=ones,
        w2_scale=ones,
        gemm1_alpha=1.0,
        gemm1_beta=0.0,
        gemm1_clamp_limit=2.0,
    )


def _make_nvfp4_quant_config():
    ones = torch.ones(1, dtype=torch.float32)
    return nvfp4_moe_quant_config(
        g1_alphas=ones,
        g2_alphas=ones,
        a1_gscale=ones,
        a2_gscale=ones,
        w1_scale=ones,
        w2_scale=ones,
        gemm1_clamp_limit=2.0,
    )


def _make_mxint4_quant_config():
    ones = torch.ones(1, dtype=torch.float32)
    return int4_w4a16_moe_quant_config(
        w1_scale=ones,
        w2_scale=ones,
        block_shape=[0, 32],
    )


def test_sm12x_trtllm_bf16_experts_constructor_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        trtllm_bf16_moe.TrtLlmBf16Experts(
            moe_config=make_dummy_moe_config(),
            quant_config=FusedMoEQuantConfig.make(),
        )


@pytest.mark.parametrize(
    "experts_cls",
    [
        trtllm_fp8_moe.TrtLlmFp8ExpertsMonolithic,
        trtllm_fp8_moe.TrtLlmFp8ExpertsModular,
    ],
)
def test_sm12x_trtllm_fp8_experts_constructors_reject(
    monkeypatch: pytest.MonkeyPatch,
    experts_cls,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("GB10 should reject TRTLLM FP8 MoE before setup")

    monkeypatch.setattr(trtllm_fp8_moe.torch, "ones_like", _should_not_be_called)

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        experts_cls(
            moe_config=make_dummy_moe_config(),
            quant_config=_make_fp8_quant_config(),
        )


@pytest.mark.parametrize(
    "experts_cls",
    [
        trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsMonolithic,
        trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsModular,
    ],
)
def test_sm12x_trtllm_mxfp4_experts_constructors_reject(
    monkeypatch: pytest.MonkeyPatch,
    experts_cls,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    def _should_not_be_called() -> None:
        raise AssertionError("GB10 should reject TRTLLM MXFP4 MoE before setup")

    monkeypatch.setattr(
        trtllm_mxfp4_moe.torch.accelerator,
        "current_device_index",
        _should_not_be_called,
    )

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        experts_cls(
            moe_config=make_dummy_moe_config(),
            quant_config=_make_mxfp4_quant_config(),
        )


@pytest.mark.parametrize(
    "experts_cls",
    [
        trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsMonolithic,
        trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsModular,
    ],
)
def test_sm12x_trtllm_nvfp4_experts_constructors_reject(
    monkeypatch: pytest.MonkeyPatch,
    experts_cls,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    def _should_not_be_called() -> None:
        raise AssertionError("GB10 should reject TRTLLM NVFP4 MoE before setup")

    monkeypatch.setattr(
        trtllm_nvfp4_moe.torch.accelerator,
        "current_device_index",
        _should_not_be_called,
    )

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        experts_cls(
            moe_config=make_dummy_moe_config(),
            quant_config=_make_nvfp4_quant_config(),
        )


def test_sm12x_trtllm_mxint4_experts_constructor_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        trtllm_mxint4_moe.TrtLlmMxint4ExpertsMonolithic(
            moe_config=make_dummy_moe_config(),
            quant_config=_make_mxint4_quant_config(),
        )
