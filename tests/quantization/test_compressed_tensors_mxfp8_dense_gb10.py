# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

ct_dense_mxfp8 = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w8a8_mxfp8"
)
ct_config_module = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
)
ct_utils = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.utils"
)
CompressedTensorsConfig = ct_config_module.CompressedTensorsConfig


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch):
    monkeypatch.setattr(
        ct_dense_mxfp8,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )
    monkeypatch.setattr(
        ct_utils,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def _mxfp8_weight_quant() -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
        scale_dtype=torch.uint8,
    )


def test_compressed_tensors_w8a8_mxfp8_dense_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_dense_mxfp8.CompressedTensorsW8A8Mxfp8()


def test_compressed_tensors_w8a8_mxfp8_config_dispatch_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format="mxfp8-pack-quantized",
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        config._get_scheme_from_parts(
            weight_quant=_mxfp8_weight_quant(),
            input_quant=None,
        )
