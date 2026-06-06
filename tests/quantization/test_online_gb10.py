# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.online.fp8 as online_fp8
import vllm.model_executor.layers.quantization.online.int8 as online_int8
import vllm.model_executor.layers.quantization.online.moe_base as online_moe_base
import vllm.model_executor.layers.quantization.online.mxfp8 as online_mxfp8


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (online_fp8, online_moe_base, online_mxfp8, online_int8):
        monkeypatch.setattr(
            module,
            "current_platform",
            Sm12xPlatform(),
            raising=False,
        )


def test_online_fp8_linear_constructors_reject_on_sm12x(sm12x_platform) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        online_fp8.Fp8PerTensorOnlineLinearMethod()

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        online_fp8.Fp8PerBlockOnlineLinearMethod()


def test_online_mxfp8_linear_constructor_rejects_on_sm12x(
    sm12x_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_mxfp8_kernel = MagicMock()
    monkeypatch.setattr(
        online_mxfp8,
        "init_mxfp8_linear_kernel",
        init_mxfp8_kernel,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        online_mxfp8.Mxfp8OnlineLinearMethod()

    init_mxfp8_kernel.assert_not_called()


@pytest.mark.parametrize(
    "method_cls, selector_module, selector_name",
    [
        (
            online_fp8.Fp8PerTensorOnlineMoEMethod,
            online_fp8,
            "select_fp8_moe_backend",
        ),
        (
            online_fp8.Fp8PerBlockOnlineMoEMethod,
            online_fp8,
            "select_fp8_moe_backend",
        ),
        (
            online_mxfp8.Mxfp8OnlineMoEMethod,
            online_mxfp8,
            "select_mxfp8_moe_backend",
        ),
        (
            online_int8.Int8OnlineMoEMethod,
            online_int8,
            "select_int8_moe_backend",
        ),
    ],
)
def test_online_moe_constructors_reject_on_sm12x(
    sm12x_platform,
    monkeypatch: pytest.MonkeyPatch,
    method_cls,
    selector_module,
    selector_name,
) -> None:
    selector = MagicMock()
    monkeypatch.setattr(selector_module, selector_name, selector, raising=False)

    layer = SimpleNamespace(moe_config=MagicMock())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        method_cls(layer=layer)

    selector.assert_not_called()
