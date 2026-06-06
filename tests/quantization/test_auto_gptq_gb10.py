# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.auto_gptq as auto_gptq


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auto_gptq,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def test_auto_gptq_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    verify_supported = MagicMock()
    monkeypatch.setattr(
        auto_gptq,
        "verify_marlin_supported",
        verify_supported,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        auto_gptq.AutoGPTQLinearMethod(SimpleNamespace())

    verify_supported.assert_not_called()


def test_auto_gptq_moe_method_constructor_rejects_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(
        auto_gptq,
        "select_wna16_moe_backend",
        selector,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        auto_gptq.AutoGPTQMoEMethod(SimpleNamespace(), MagicMock())

    selector.assert_not_called()
