# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _mock_sm12x_cuda_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls: DeviceCapability(major=12, minor=1)),
    )
    monkeypatch.setattr(
        CudaPlatform,
        "has_device_capability",
        classmethod(lambda cls, capability: capability <= 120),
    )


@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    ],
)
def test_sm12x_rejects_unvalidated_mm_encoder_attention_backend_overrides(
    monkeypatch,
    backend: AttentionBackendEnum,
):
    _mock_sm12x_cuda_platform(monkeypatch)

    with pytest.raises(ValueError, match="MM encoder attention.*GB10/SM12x"):
        CudaPlatform.get_vit_attn_backend(
            head_size=72,
            dtype=torch.bfloat16,
            backend=backend,
        )


def test_sm12x_allows_mm_encoder_flashinfer_backend_override(monkeypatch):
    _mock_sm12x_cuda_platform(monkeypatch)

    assert (
        CudaPlatform.get_vit_attn_backend(
            head_size=72,
            dtype=torch.bfloat16,
            backend=AttentionBackendEnum.FLASHINFER,
        )
        == AttentionBackendEnum.FLASHINFER
    )


def test_sm12x_auto_selects_flashinfer_mm_encoder(monkeypatch):
    # With no explicit override, media serving on SM12x auto-selects the native
    # FlashInfer ViT backend (the mandated GB10 MM-encoder attention) rather
    # than falling through the supports_head_size gate to the guarded Triton
    # fallback. Validated end-to-end on RedHatAI/Qwen3.6-35B-A3B-NVFP4.
    _mock_sm12x_cuda_platform(monkeypatch)

    assert (
        CudaPlatform.get_vit_attn_backend(
            head_size=72,
            dtype=torch.bfloat16,
        )
        == AttentionBackendEnum.FLASHINFER
    )


def test_sm12x_rejects_mm_encoder_attention_fallback_without_flashinfer(
    monkeypatch,
):
    _mock_sm12x_cuda_platform(monkeypatch)
    monkeypatch.setattr(
        CudaPlatform,
        "get_supported_vit_attn_backends",
        classmethod(
            lambda cls: [
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            ]
        ),
    )

    with pytest.raises(ValueError, match="requires FlashInfer.*GB10/SM12x"):
        CudaPlatform.get_vit_attn_backend(
            head_size=72,
            dtype=torch.bfloat16,
        )
