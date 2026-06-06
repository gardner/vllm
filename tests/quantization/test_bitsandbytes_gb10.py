# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.quantization.bitsandbytes as bitsandbytes


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bitsandbytes,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


def test_bitsandbytes_linear_method_constructor_rejects_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_version = MagicMock()
    monkeypatch.setattr(
        bitsandbytes,
        "_check_bitsandbytes_version",
        check_version,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        bitsandbytes.BitsAndBytesLinearMethod(SimpleNamespace())

    check_version.assert_not_called()


def test_bitsandbytes_moe_method_constructor_rejects_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_version = MagicMock()
    monkeypatch.setattr(
        bitsandbytes,
        "_check_bitsandbytes_version",
        check_version,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        bitsandbytes.BitsAndBytesMoEMethod(SimpleNamespace(), MagicMock())

    check_version.assert_not_called()
