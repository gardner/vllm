# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module

import pytest
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

ct_dense_fp8 = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w8a8_fp8"
)
ct_dense_wna16 = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_wNa16"
)
ct_utils = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.utils"
)


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch):
    monkeypatch.setattr(
        ct_utils,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def _w8a8_fp8_weight_quant() -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )


def test_compressed_tensors_w8a8_fp8_dense_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_dense_fp8.CompressedTensorsW8A8Fp8(
            weight_quant=_w8a8_fp8_weight_quant(),
            is_static_input_scheme=True,
        )


def test_compressed_tensors_wna16_dense_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_dense_wna16.CompressedTensorsWNA16(
            strategy="channel",
            num_bits=4,
        )
