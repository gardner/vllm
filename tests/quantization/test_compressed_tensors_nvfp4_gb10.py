# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module

import pytest
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A16Fp4,
)

ct_w4a16 = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w4a16_nvfp4"
)
ct_qutlass = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.transform."
    "schemes.linear_qutlass_nvfp4"
)
ct_config_module = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
)
CompressedTensorsConfig = ct_config_module.CompressedTensorsConfig


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch):
    monkeypatch.setattr(
        ct_w4a16,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def _nvfp4_weight_quant() -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        group_size=16,
        dynamic=False,
    )


def test_compressed_tensors_w4a16_nvfp4_rejects_marlin_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        CompressedTensorsW4A16Fp4()


def test_compressed_tensors_w4a16_nvfp4_config_dispatch_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format="nvfp4-pack-quantized",
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        config._get_scheme_from_parts(
            weight_quant=_nvfp4_weight_quant(),
            input_quant=None,
        )


def test_compressed_tensors_qutlass_nvfp4_transform_constructor_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_qutlass,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_qutlass.QutlassNvFP4LinearMethod(
            quant_method=object(),
            input_tfms={},
            output_tfms={},
        )
