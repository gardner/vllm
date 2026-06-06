# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.awq as awq
import vllm.model_executor.layers.quantization.awq_marlin as awq_marlin


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(awq, "current_platform", Sm12xPlatform(), raising=False)
    monkeypatch.setattr(
        awq_marlin,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def test_awq_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        awq.AWQLinearMethod(SimpleNamespace())


def test_awq_marlin_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    verify_supported = MagicMock()
    monkeypatch.setattr(
        awq_marlin,
        "verify_marlin_supported",
        verify_supported,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        awq_marlin.AWQMarlinLinearMethod(SimpleNamespace())

    verify_supported.assert_not_called()


def test_awq_marlin_moe_method_constructor_rejects_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(
        awq_marlin,
        "select_wna16_moe_backend",
        selector,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        awq_marlin.AWQMarlinMoEMethod(SimpleNamespace(), MagicMock())

    selector.assert_not_called()
