# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module

import pytest
from compressed_tensors.quantization import QuantizationStrategy

ct_w4a8_int = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w4a8_int"
)
ct_w8a8_int8 = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w8a8_int8"
)


def test_compressed_tensors_w4a8_int_dense_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_w4a8_int,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        ct_w4a8_int.CompressedTensorsW4A8Int(
            strategy=QuantizationStrategy.CHANNEL,
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W4A8 Int dense loading" in reason
    assert "generic mixed-precision" in reason
    assert "not native GB10 W4A8 Int dense evidence" in reason


def test_compressed_tensors_w8a8_int8_dense_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_w8a8_int8,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        ct_w8a8_int8.CompressedTensorsW8A8Int8(
            strategy=QuantizationStrategy.CHANNEL,
            is_static_input_scheme=False,
            input_symmetric=True,
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W8A8 Int dense loading" in reason
    assert "Cutlass/Triton W8A8 Int8 scaled-mm kernels" in reason
    assert "Use a native SM12x W8A8 Int8 dense backend" in reason
