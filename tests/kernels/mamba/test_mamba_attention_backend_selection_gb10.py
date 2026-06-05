# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.platforms as platforms
from vllm.v1.attention import selector
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


@pytest.fixture(autouse=True)
def clear_mamba_backend_cache():
    selector._cached_get_mamba_attn_backend.cache_clear()
    yield
    selector._cached_get_mamba_attn_backend.cache_clear()


def _mock_sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda family: family == 120,
        ),
        raising=False,
    )


@pytest.mark.parametrize(
    "mamba_type",
    [
        MambaAttentionBackendEnum.MAMBA1,
        MambaAttentionBackendEnum.MAMBA2,
        MambaAttentionBackendEnum.SHORT_CONV,
        MambaAttentionBackendEnum.LINEAR,
    ],
)
def test_sm12x_rejects_unvalidated_mamba_attention_backends(
    monkeypatch: pytest.MonkeyPatch,
    mamba_type: MambaAttentionBackendEnum,
):
    _mock_sm12x_platform(monkeypatch)

    with pytest.raises(ValueError, match="Mamba.*GB10/SM12x"):
        selector.get_mamba_attn_backend(mamba_type)


def test_sm12x_allows_gdn_attention_backend(monkeypatch: pytest.MonkeyPatch):
    _mock_sm12x_platform(monkeypatch)

    backend = selector.get_mamba_attn_backend(MambaAttentionBackendEnum.GDN_ATTN)

    assert backend.get_name() == "GDN_ATTN"
