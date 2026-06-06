# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashattn_mla import FlashAttnMLAImpl
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)


def _make_gb10_platform() -> SimpleNamespace:
    return SimpleNamespace(
        get_device_capability=lambda: DeviceCapability(major=12, minor=1),
    )


def test_public_flashattention_mla_impl_rejects_gb10_sm12x():
    with patch(
        "vllm.v1.attention.backends.mla.flashattn_mla.current_platform",
        _make_gb10_platform(),
    ), pytest.raises(ValueError, match="public FlashAttention MLA.*GB10/SM12x"):
        FlashAttnMLAImpl(
            num_heads=8,
            head_size=192,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )


def test_public_flashattention_mla_prefill_backend_rejects_gb10_sm12x():
    with patch(
        "vllm.v1.attention.backends.mla.prefill.flash_attn.current_platform",
        _make_gb10_platform(),
    ), pytest.raises(ValueError, match="public FlashAttention MLA.*GB10/SM12x"):
        FlashAttnPrefillBackend(
            num_heads=8,
            scale=1.0,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            vllm_config=SimpleNamespace(),
        )
