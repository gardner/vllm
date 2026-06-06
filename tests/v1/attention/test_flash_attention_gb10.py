# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.flash_attn_diffkv import FlashAttentionDiffKVImpl


def _make_gb10_platform() -> SimpleNamespace:
    return SimpleNamespace(
        get_device_capability=lambda: DeviceCapability(major=12, minor=1),
    )


@pytest.mark.parametrize("impl_cls", [FlashAttentionImpl, FlashAttentionDiffKVImpl])
def test_public_flashattention_impl_rejects_gb10_sm12x(impl_cls):
    with patch(
        "vllm.v1.attention.backends.flash_attn.current_platform",
        _make_gb10_platform(),
    ), pytest.raises(ValueError, match="public FlashAttention.*GB10/SM12x"):
        impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )
