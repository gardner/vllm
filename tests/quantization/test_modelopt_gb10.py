# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

modelopt = import_module("vllm.model_executor.layers.quantization.modelopt")


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        modelopt,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


@pytest.mark.parametrize(
    "constructor",
    [
        modelopt.ModelOptFp8LinearMethod,
        modelopt.ModelOptFp8PcPtLinearMethod,
        modelopt.ModelOptFp8PbWoLinearMethod,
    ],
)
def test_modelopt_fp8_linear_constructors_reject_on_sm12x(
    sm12x_platform, constructor
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        constructor(SimpleNamespace(quant_method="FP8"))


def test_modelopt_fp8_moe_constructor_rejects_on_sm12x_before_backend_selection(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(modelopt, "select_fp8_moe_backend", selector)

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptFp8MoEMethod(
            SimpleNamespace(
                quant_method="FP8",
                is_checkpoint_fp8_serialized=True,
            ),
            MagicMock(),
        )

    selector.assert_not_called()


def test_modelopt_mxfp8_linear_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptMxFp8LinearMethod(
            SimpleNamespace(is_checkpoint_mxfp8_serialized=True)
        )


def test_modelopt_mxfp8_moe_constructor_rejects_on_sm12x_before_backend_selection(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(modelopt, "select_mxfp8_moe_backend", selector)

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptMxFp8FusedMoE(
            SimpleNamespace(is_checkpoint_mxfp8_serialized=True),
            MagicMock(),
        )

    selector.assert_not_called()
