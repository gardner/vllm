# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.platforms.cuda as cuda_platform
from vllm.config.cache import CacheDType
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.mla.flashmla import FlashMLABackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig


def _validate_mla_backend(
    backend: type[AttentionBackend],
    *,
    capability: DeviceCapability,
    head_size: int = 576,
    kv_cache_dtype: CacheDType | None = "auto",
    use_sparse: bool = False,
) -> list[str]:
    return backend.validate_configuration(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=use_sparse,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=capability,
        attn_type=AttentionType.DECODER,
        use_non_causal=False,
        use_batch_invariant=False,
        use_kv_connector=False,
    )


def _mla_selector_config(*, use_sparse: bool = False) -> AttentionSelectorConfig:
    return AttentionSelectorConfig(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=use_sparse,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        attn_type=AttentionType.DECODER,
        use_non_causal=False,
        use_batch_invariant=False,
        use_kv_connector=False,
    )


@pytest.mark.parametrize(
    "backend",
    [
        FlashMLABackend,
        FlashMLASparseBackend,
    ],
)
def test_gb10_flashmla_backends_accept_sm12x_when_extension_available(
    monkeypatch: pytest.MonkeyPatch,
    backend: type[AttentionBackend],
) -> None:
    monkeypatch.setattr(
        "vllm.v1.attention.ops.flashmla.is_flashmla_dense_supported",
        lambda: (True, None),
    )
    monkeypatch.setattr(
        "vllm.v1.attention.ops.flashmla.is_flashmla_sparse_supported",
        lambda: (True, None),
    )

    reasons = _validate_mla_backend(
        backend,
        capability=DeviceCapability(12, 1),
        use_sparse=backend.is_sparse(),
    )

    assert reasons == []


def test_gb10_triton_mla_fallback_rejects_sm12x() -> None:
    reasons = _validate_mla_backend(
        TritonMLABackend,
        capability=DeviceCapability(12, 1),
    )

    assert any(
        "Triton MLA backend is not supported on GB10/SM12x" in r for r in reasons
    )


def test_sm12x_mla_priorities_only_try_native_flashmla_backends() -> None:
    cuda_platform._get_backend_priorities.cache_clear()

    assert cuda_platform._get_backend_priorities(
        use_mla=True,
        device_capability=DeviceCapability(major=12, minor=1),
    ) == [
        AttentionBackendEnum.FLASHMLA,
        AttentionBackendEnum.FLASHMLA_SPARSE,
    ]


@pytest.mark.parametrize(
    ("backend", "expected_name"),
    [
        (AttentionBackendEnum.TRITON_MLA, "Triton MLA"),
        (AttentionBackendEnum.FLASHINFER_MLA, "FlashInfer TRT-LLM MLA"),
        (AttentionBackendEnum.FLASHINFER_MLA_SPARSE, "FlashInfer TRT-LLM Sparse MLA"),
        (AttentionBackendEnum.FLASH_ATTN_MLA, "public FlashAttention MLA"),
        (AttentionBackendEnum.CUTLASS_MLA, "SM100 CUTLASS MLA"),
        (AttentionBackendEnum.TOKENSPEED_MLA, "TokenSpeed CuTe DSL MLA"),
    ],
)
def test_sm12x_rejects_unvalidated_mla_backend_overrides_before_import(
    monkeypatch: pytest.MonkeyPatch,
    backend: AttentionBackendEnum,
    expected_name: str,
) -> None:
    monkeypatch.setattr(
        CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls: DeviceCapability(major=12, minor=1)),
    )

    with pytest.raises(ValueError, match=f"{expected_name}.*GB10/SM12x"):
        CudaPlatform.get_attn_backend_cls(
            selected_backend=backend,
            attn_selector_config=_mla_selector_config(),
        )


def test_sm12x_mla_selector_prefers_flashmla_over_triton_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls: DeviceCapability(major=12, minor=1)),
    )
    monkeypatch.setattr(
        "vllm.platforms.cuda._get_backend_priorities",
        lambda *args, **kwargs: [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.TRITON_MLA,
        ],
    )
    monkeypatch.setattr(
        "vllm.v1.attention.ops.flashmla.is_flashmla_dense_supported",
        lambda: (True, None),
    )

    assert CudaPlatform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=_mla_selector_config(),
    ) == AttentionBackendEnum.FLASHMLA.get_path()


def test_sm12x_mla_selector_rejects_triton_when_flashmla_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls: DeviceCapability(major=12, minor=1)),
    )
    monkeypatch.setattr(
        "vllm.platforms.cuda._get_backend_priorities",
        lambda *args, **kwargs: [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.TRITON_MLA,
        ],
    )
    monkeypatch.setattr(
        "vllm.v1.attention.ops.flashmla.is_flashmla_dense_supported",
        lambda: (False, "FlashMLA is unavailable in this test"),
    )

    with pytest.raises(ValueError, match="Triton MLA backend.*GB10/SM12x"):
        CudaPlatform.get_attn_backend_cls(
            selected_backend=None,
            attn_selector_config=_mla_selector_config(),
        )
