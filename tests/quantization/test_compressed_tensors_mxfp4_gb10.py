# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from unittest.mock import MagicMock

import pytest
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

ct_dense_mxfp4 = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w4a4_mxfp4"
)
ct_moe_mxfp4 = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4"
)
ct_config_module = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
)
CompressedTensorsW4A4Mxfp4 = ct_dense_mxfp4.CompressedTensorsW4A4Mxfp4
CompressedTensorsW4A4Mxfp4MoEMethod = (
    ct_moe_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod
)
CompressedTensorsConfig = ct_config_module.CompressedTensorsConfig


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch):
    monkeypatch.setattr(
        ct_dense_mxfp4,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )
    monkeypatch.setattr(
        ct_moe_mxfp4,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def _mxfp4_weight_quant() -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        group_size=32,
        dynamic=False,
    )


def test_compressed_tensors_w4a4_mxfp4_dense_rejects_on_sm12x(
    sm12x_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_dense_mxfp4,
        "init_mxfp4_linear_kernel",
        lambda: MagicMock(),
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        CompressedTensorsW4A4Mxfp4()


def test_compressed_tensors_w4a4_mxfp4_config_dispatch_rejects_on_sm12x(
    sm12x_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_dense_mxfp4,
        "init_mxfp4_linear_kernel",
        lambda: MagicMock(),
    )
    config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format="mxfp4-pack-quantized",
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        config._get_scheme_from_parts(
            weight_quant=_mxfp4_weight_quant(),
            input_quant=None,
        )


def test_compressed_tensors_w4a4_mxfp4_moe_rejects_marlin_fallback_on_sm12x(
    sm12x_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_moe_mxfp4.CutlassExpertsMxfp4,
        "_supports_current_device",
        staticmethod(lambda: False),
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        CompressedTensorsW4A4Mxfp4MoEMethod(MagicMock())


def test_compressed_tensors_w4a4_mxfp4_moe_keeps_native_cutlass_on_sm12x(
    sm12x_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_moe_mxfp4.CutlassExpertsMxfp4,
        "_supports_current_device",
        staticmethod(lambda: True),
    )

    method = CompressedTensorsW4A4Mxfp4MoEMethod(MagicMock())

    assert method.use_cutlass_mxfp4
    assert method.experts_cls is ct_moe_mxfp4.CutlassExpertsMxfp4
