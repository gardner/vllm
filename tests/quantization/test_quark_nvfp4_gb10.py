# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.quark.utils as quark_utils
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkNvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_nvfp4 import (
    QuarkNVFP4,
)


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch):
    monkeypatch.setattr(
        quark_utils,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def test_quark_nvfp4_dense_rejects_checkpoint_loading_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        QuarkNVFP4()


def test_quark_nvfp4_moe_rejects_checkpoint_loading_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        QuarkNvfp4MoEMethod(
            weight_config={},
            input_config={},
            moe=MagicMock(),
            quant_config=MagicMock(),
        )
