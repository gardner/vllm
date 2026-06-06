# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.quark.utils as quark_utils
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkW8A8Fp8MoEMethod,
    QuarkW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
    QuarkW8A8Fp8,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_int8 import (
    QuarkW8A8Int8,
)


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        quark_utils,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def _w8a8_fp8_weight_quant() -> dict[str, object]:
    return {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }


def _w8a8_fp8_input_quant() -> dict[str, object]:
    return {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }


def _w8a8_int8_weight_quant() -> dict[str, object]:
    return {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "symmetric": True,
    }


def _w8a8_int8_input_quant() -> dict[str, object]:
    return {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }


def test_quark_w8a8_fp8_checkpoint_loading_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        QuarkW8A8Fp8(
            weight_config=_w8a8_fp8_weight_quant(),
            input_config=_w8a8_fp8_input_quant(),
        )


def test_quark_w8a8_fp8_moe_checkpoint_loading_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        QuarkW8A8Fp8MoEMethod(
            weight_config=_w8a8_fp8_weight_quant(),
            input_config=_w8a8_fp8_input_quant(),
            moe=MagicMock(),
        )


def test_quark_w8a8_int8_checkpoint_loading_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        QuarkW8A8Int8(
            qscheme="per_tensor",
            is_static_input_scheme=True,
            input_symmetric=True,
        )


def test_quark_w8a8_int8_moe_checkpoint_loading_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        QuarkW8A8Int8MoEMethod(
            weight_config=_w8a8_int8_weight_quant(),
            input_config=_w8a8_int8_input_quant(),
            moe=MagicMock(),
        )
