# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

import vllm.model_executor.layers.attention.mm_encoder_attention as mm_attn_mod
import vllm.utils.flashinfer as flashinfer_utils
from vllm.config.multimodal import MultiModalConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class _Sm12xCudaPlatform:

    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def has_device_capability(capability: int) -> bool:
        return capability <= 120

    @staticmethod
    def is_device_capability_family(capability: int) -> bool:
        return capability == 120

    @staticmethod
    def get_supported_vit_attn_backends():
        return {AttentionBackendEnum.FLASHINFER}


def test_flashinfer_cudnn_fp8_prefill_attn_reports_unsupported_on_sm12x(
    monkeypatch,
):
    flashinfer_utils.is_flashinfer_cudnn_fp8_prefill_attn_supported.cache_clear()
    monkeypatch.setattr(
        flashinfer_utils,
        "current_platform",
        _Sm12xCudaPlatform(),
    )
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 99999)

    try:
        assert not flashinfer_utils.is_flashinfer_cudnn_fp8_prefill_attn_supported()
    finally:
        flashinfer_utils.is_flashinfer_cudnn_fp8_prefill_attn_supported.cache_clear()


def test_mm_encoder_fp8_attention_rejects_on_sm12x(
    default_vllm_config,
    monkeypatch,
):
    monkeypatch.setattr(
        mm_attn_mod,
        "current_platform",
        _Sm12xCudaPlatform(),
        raising=False,
    )
    monkeypatch.setattr(
        mm_attn_mod,
        "get_multimodal_config",
        lambda: MultiModalConfig(mm_encoder_attn_dtype="fp8"),
    )
    monkeypatch.setattr(
        mm_attn_mod,
        "get_vit_attn_backend",
        lambda *, head_size, dtype: AttentionBackendEnum.FLASHINFER,
    )
    monkeypatch.setattr(
        mm_attn_mod,
        "_get_flashinfer_workspace_buffer",
        lambda: torch.empty(0, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        mm_attn_mod,
        "is_flashinfer_cudnn_fp8_prefill_attn_supported",
        lambda: True,
    )

    with (
        patch.object(torch, "get_default_dtype", return_value=torch.bfloat16),
        pytest.raises(ValueError, match="MM encoder FP8 attention.*GB10/SM12x"),
    ):
        mm_attn_mod.MMEncoderAttention(
            num_heads=16,
            head_size=72,
            prefix="visual.blocks.0.attn.attn",
        )
