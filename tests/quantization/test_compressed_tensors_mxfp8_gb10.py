# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from unittest.mock import MagicMock

import pytest

ct_mxfp8_moe = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8"
)
ct_moe = import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)


def test_compressed_tensors_w8a8_mxfp8_moe_constructor_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ct_moe,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        ct_mxfp8_moe.CompressedTensorsW8A8Mxfp8MoEMethod(MagicMock())
