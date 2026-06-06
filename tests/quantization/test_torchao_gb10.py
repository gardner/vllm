# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.model_executor.layers.quantization.torchao as torchao_quant


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


class NonSm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return False


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torchao_quant,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def test_torchao_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    quant_config = SimpleNamespace(
        torchao_config=SimpleNamespace(),
        is_checkpoint_torchao_serialized=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        torchao_quant.TorchAOLinearMethod(quant_config)


def test_torchao_linear_method_constructor_allows_non_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torchao_quant,
        "current_platform",
        NonSm12xPlatform(),
        raising=False,
    )
    quant_config = SimpleNamespace(
        torchao_config=SimpleNamespace(),
        is_checkpoint_torchao_serialized=False,
    )

    method = torchao_quant.TorchAOLinearMethod(quant_config)

    assert method.quant_config is quant_config
