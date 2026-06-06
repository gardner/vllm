# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _FakeGGMLQuantizationType:
    F32 = "F32"
    F16 = "F16"
    BF16 = "BF16"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q8_0 = "Q8_0"
    Q8_1 = "Q8_1"
    Q2_K = "Q2_K"
    Q3_K = "Q3_K"
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    IQ1_M = "IQ1_M"
    IQ1_S = "IQ1_S"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    IQ3_XXS = "IQ3_XXS"
    IQ3_S = "IQ3_S"
    IQ4_XS = "IQ4_XS"
    IQ4_NL = "IQ4_NL"


fake_gguf = ModuleType("gguf")
fake_gguf.GGMLQuantizationType = _FakeGGMLQuantizationType
sys.modules["gguf"] = fake_gguf

gguf = import_module("vllm.model_executor.layers.quantization.gguf")


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf, "current_platform", Sm12xPlatform(), raising=False)


def test_gguf_config_rejects_on_sm12x(sm12x_platform) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        gguf.GGUFConfig()


def test_gguf_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        gguf.GGUFLinearMethod(SimpleNamespace())


def test_gguf_embedding_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        gguf.GGUFEmbeddingMethod(SimpleNamespace())


def test_gguf_moe_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        gguf.GGUFMoEMethod(SimpleNamespace(), MagicMock())
