# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.model_executor.layers.quantization.fp_quant as fp_quant


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fp_quant, "current_platform", Sm12xPlatform(), raising=False)


def test_fp_quant_config_rejects_on_sm12x(sm12x_platform) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        fp_quant.FPQuantConfig()


def test_fp_quant_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        fp_quant.FPQuantLinearMethod(SimpleNamespace())
