# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.moe_wna16 as moe_wna16


def test_moe_wna16_method_constructor_rejects_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        moe_wna16,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        moe_wna16.MoeWNA16Method(MagicMock(), MagicMock())
