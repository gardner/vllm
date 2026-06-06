# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.model_executor.layers.quantization.fbgemm_fp8 as fbgemm_fp8


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fbgemm_fp8,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def test_fbgemm_fp8_config_rejects_on_sm12x(sm12x_platform) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        fbgemm_fp8.FBGEMMFp8Config(ignore_list=[], input_scale_ub=1.0)


def test_fbgemm_fp8_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        fbgemm_fp8.FBGEMMFp8LinearMethod(SimpleNamespace())
