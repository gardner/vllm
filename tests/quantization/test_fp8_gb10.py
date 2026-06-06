# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.model_executor.layers.quantization.fp8 as fp8


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fp8, "current_platform", Sm12xPlatform(), raising=False)


def test_public_fp8_config_rejects_on_sm12x(sm12x_platform) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        fp8.Fp8Config()

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        fp8.Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
            }
        )


@pytest.mark.parametrize(
    ("ctor", "args"),
    [
        (fp8.Fp8LinearMethod, (SimpleNamespace(),)),
        (fp8.Fp8OnlineLinearMethod, (SimpleNamespace(),)),
        (fp8.Fp8MoEMethod, (SimpleNamespace(), SimpleNamespace())),
        (fp8.Fp8OnlineMoEMethod, (SimpleNamespace(), SimpleNamespace())),
        (fp8.Fp8KVCacheMethod, (SimpleNamespace(),)),
    ],
)
def test_public_fp8_method_constructors_reject_on_sm12x(
    sm12x_platform,
    ctor,
    args,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ctor(*args)
