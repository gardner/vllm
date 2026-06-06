# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts import (
    flashinfer_cutedsl_batched_moe,
    flashinfer_cutedsl_moe,
)
from vllm.platforms.interface import DeviceCapability


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        return DeviceCapability(12, 1)


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = Sm12xPlatform()
    monkeypatch.setattr(
        flashinfer_cutedsl_moe,
        "current_platform",
        platform,
        raising=False,
    )
    monkeypatch.setattr(
        flashinfer_cutedsl_batched_moe,
        "current_platform",
        platform,
        raising=False,
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
    )


@pytest.mark.parametrize(
    "experts_cls,module,probe_name,constructor_kwargs",
    [
        pytest.param(
            flashinfer_cutedsl_moe.FlashInferCuteDSLExperts,
            flashinfer_cutedsl_moe,
            "has_flashinfer_cutedsl_moe_nvfp4",
            {},
            id="standard",
        ),
        pytest.param(
            flashinfer_cutedsl_batched_moe.FlashInferCuteDSLBatchedExperts,
            flashinfer_cutedsl_batched_moe,
            "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
            {"max_num_tokens": 64, "num_dispatchers": 1},
            id="batched",
        ),
    ],
)
def test_flashinfer_cutedsl_nvfp4_moe_constructors_reject_on_sm12x(
    sm12x_platform,
    monkeypatch: pytest.MonkeyPatch,
    experts_cls,
    module,
    probe_name,
    constructor_kwargs,
) -> None:
    probe = MagicMock()
    monkeypatch.setattr(module, probe_name, probe, raising=False)

    with pytest.raises(
        ValueError,
        match="FlashInfer CuteDSL NVFP4 MoE is not supported on GB10/SM12x",
    ):
        experts_cls(
            moe_config=make_dummy_moe_config(
                num_experts=8,
                experts_per_token=2,
                hidden_dim=128,
                intermediate_size_per_partition=256,
            ),
            quant_config=_make_nvfp4_quant_config(),
            **constructor_kwargs,
        )

    probe.assert_not_called()
