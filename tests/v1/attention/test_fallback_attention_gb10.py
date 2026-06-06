# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flex_attention import FlexAttentionImpl
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl


def _make_gb10_platform() -> SimpleNamespace:
    return SimpleNamespace(
        get_device_capability=lambda: DeviceCapability(major=12, minor=1),
    )


@pytest.mark.parametrize(
    ("patch_target", "impl_cls", "kwargs", "match"),
    [
        (
            "vllm.v1.attention.backends.triton_attn.current_platform",
            TritonAttentionImpl,
            {
                "num_heads": 8,
                "head_size": 128,
                "scale": 1.0,
                "num_kv_heads": 8,
                "alibi_slopes": None,
                "sliding_window": None,
                "kv_cache_dtype": "auto",
            },
            "Triton attention backend.*GB10/SM12x",
        ),
        (
            "vllm.v1.attention.backends.flex_attention.current_platform",
            FlexAttentionImpl,
            {
                "num_heads": 8,
                "head_size": 128,
                "scale": 1.0,
                "num_kv_heads": 8,
                "alibi_slopes": None,
                "sliding_window": None,
                "kv_cache_dtype": "auto",
            },
            "FlexAttention backend.*GB10/SM12x",
        ),
        (
            "vllm.v1.attention.backends.turboquant_attn.current_platform",
            TurboQuantAttentionImpl,
            {
                "num_heads": 8,
                "head_size": 128,
                "scale": 1.0,
                "num_kv_heads": 8,
                "kv_cache_dtype": "turboquant_4bit_nc",
            },
            "TurboQuant attention backend.*GB10/SM12x",
        ),
    ],
)
def test_fallback_attention_impls_reject_gb10_sm12x(
    patch_target,
    impl_cls,
    kwargs,
    match,
):
    with patch(patch_target, _make_gb10_platform()), pytest.raises(
        ValueError,
        match=match,
    ):
        impl_cls(**kwargs)
