# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.humming as humming


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        humming,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def test_humming_config_rejects_on_sm12x(sm12x_platform) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        humming.HummingConfig()


def test_humming_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        humming.HummingLinearMethod(SimpleNamespace())


def test_humming_moe_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        humming.HummingMoEMethod(SimpleNamespace(), MagicMock())
