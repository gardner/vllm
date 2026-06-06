# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.mxfp4 as mxfp4


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mxfp4, "current_platform", Sm12xPlatform(), raising=False)


def test_mxfp4_config_rejects_on_sm12x(sm12x_platform) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        mxfp4.Mxfp4Config()


def test_gpt_oss_mxfp4_moe_method_rejects_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(
        mxfp4,
        "select_mxfp4_moe_backend",
        selector,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        mxfp4.GptOssMxfp4MoEMethod(MagicMock())

    selector.assert_not_called()


def test_mxfp4_moe_method_rejects_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(
        mxfp4,
        "select_deepseek_v4_mxfp4_moe_backend",
        selector,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        mxfp4.Mxfp4MoEMethod(MagicMock())

    selector.assert_not_called()
