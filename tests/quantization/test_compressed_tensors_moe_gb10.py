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

ct_utils = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.utils"
)
ct_w4a8_fp8_moe = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w4a8_fp8"
)
ct_w8a8_fp8_moe = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8"
)
ct_w8a8_int8_moe = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)
ct_wna16_moe = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_wna16"
)
ct_wna16_marlin_moe = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_wna16_marlin"
)
ct_moe = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)


def _w4a8_fp8_weight_quant() -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )


def _w4a8_fp8_input_quant() -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )


def _wna16_weight_quant() -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=64,
    )


def test_compressed_tensors_w4a8_fp8_moe_constructor_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_utils,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_w4a8_fp8_moe.CompressedTensorsW4A8Fp8MoEMethod(
            _w4a8_fp8_weight_quant(),
            _w4a8_fp8_input_quant(),
            MagicMock(),
        )


def test_compressed_tensors_w8a8_fp8_moe_constructor_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_w8a8_fp8_moe,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_w8a8_fp8_moe.CompressedTensorsW8A8Fp8MoEMethod(
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )


def test_compressed_tensors_w8a8_int8_moe_constructor_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_w8a8_int8_moe,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_w8a8_int8_moe.CompressedTensorsW8A8Int8MoEMethod(
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )


def test_compressed_tensors_wna16_moe_constructor_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ct_moe, "_is_sm12x_device", lambda: True, raising=False)

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_wna16_moe.CompressedTensorsWNA16MoEMethod(
            _wna16_weight_quant(),
            None,
            MagicMock(),
        )


def test_compressed_tensors_wna16_marlin_moe_constructor_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ct_moe, "_is_sm12x_device", lambda: True, raising=False)

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_wna16_marlin_moe.CompressedTensorsWNA16MarlinMoEMethod(
            _wna16_weight_quant(),
            None,
            MagicMock(),
        )
