# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.flex_attention import FlexAttentionBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend


def _validate_attention_backend(
    backend: type[AttentionBackend],
    *,
    capability: DeviceCapability,
    kv_cache_dtype: CacheDType | None = "auto",
) -> list[str]:
    return backend.validate_configuration(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=capability,
        attn_type=AttentionType.DECODER,
        use_non_causal=False,
        use_batch_invariant=False,
        use_kv_connector=False,
    )


@pytest.mark.parametrize(
    ("backend", "kv_cache_dtype", "expected_phrase"),
    [
        (
            TritonAttentionBackend,
            "auto",
            "Triton attention backend is not supported on GB10/SM12x",
        ),
        (
            FlexAttentionBackend,
            "auto",
            "FlexAttention backend is not supported on GB10/SM12x",
        ),
        (
            TurboQuantAttentionBackend,
            "turboquant_k8v4",
            "TurboQuant attention backend is not supported on GB10/SM12x",
        ),
    ],
)
def test_gb10_unvalidated_attention_fallbacks_reject_sm12x(
    backend: type[AttentionBackend],
    kv_cache_dtype: CacheDType,
    expected_phrase: str,
) -> None:
    reasons = _validate_attention_backend(
        backend,
        capability=DeviceCapability(12, 1),
        kv_cache_dtype=kv_cache_dtype,
    )

    assert any(expected_phrase in reason for reason in reasons)


@pytest.mark.parametrize(
    ("backend", "kv_cache_dtype"),
    [
        (TritonAttentionBackend, "auto"),
        (FlexAttentionBackend, "auto"),
        (TurboQuantAttentionBackend, "turboquant_k8v4"),
    ],
)
def test_unvalidated_attention_fallbacks_do_not_emit_gb10_reason_off_sm12x(
    backend: type[AttentionBackend],
    kv_cache_dtype: CacheDType,
) -> None:
    reasons = _validate_attention_backend(
        backend,
        capability=DeviceCapability(10, 0),
        kv_cache_dtype=kv_cache_dtype,
    )

    assert not any("GB10/SM12x" in reason for reason in reasons)
