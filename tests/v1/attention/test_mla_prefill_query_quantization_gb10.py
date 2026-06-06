# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.attention.mla_attention as mla_attention
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
)


class _Sm12xCudaPlatform:
    def is_cuda(self) -> bool:
        return True

    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


def _make_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        attention_config=SimpleNamespace(use_prefill_query_quantization=True),
    )


def test_sm12x_prefill_query_quantization_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mla_attention, "current_platform", _Sm12xCudaPlatform())

    with pytest.raises(ValueError, match="GB10/SM12x"):
        MLACommonMetadataBuilder.determine_prefill_query_data_type(
            _make_vllm_config(),
            torch.bfloat16,
        )
