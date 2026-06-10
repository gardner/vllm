# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

modelopt = import_module("vllm.model_executor.layers.quantization.modelopt")


class Sm12xPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


@pytest.fixture
def sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        modelopt,
        "current_platform",
        Sm12xPlatform(),
        raising=False,
    )


@pytest.mark.parametrize(
    "constructor",
    [
        modelopt.ModelOptFp8PcPtLinearMethod,
        modelopt.ModelOptFp8PbWoLinearMethod,
    ],
)
def test_modelopt_fp8_unvalidated_linear_constructors_reject_on_sm12x(
    sm12x_platform, constructor
) -> None:
    # FP8 PerChannelPerToken and PB-WO dense linear remain rejected on GB10
    # until validated. (FP8 standard W8A8 dense is native — see below.)
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        constructor(SimpleNamespace(quant_method="FP8"))


def test_modelopt_fp8_dense_linear_is_native_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FP8 W8A8 standard dense linear is validated native on GB10 (selects
    # FlashInferFP8ScaledMMLinearKernel) — its constructor no longer raises the
    # GB10 guard. Proven end-to-end in the nvidia/Qwen3.6-35B-A3B-NVFP4
    # mixed-precision smoke (FP8 attention + Qwen3-Next linear_attn).
    import torch

    monkeypatch.setattr(
        modelopt,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16)
        ),
        raising=False,
    )
    method = modelopt.ModelOptFp8LinearMethod(SimpleNamespace(quant_method="FP8"))
    assert isinstance(method, modelopt.ModelOptFp8LinearMethod)


def test_modelopt_fp8_moe_constructor_rejects_on_sm12x_before_backend_selection(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(modelopt, "select_fp8_moe_backend", selector)

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptFp8MoEMethod(
            SimpleNamespace(
                quant_method="FP8",
                is_checkpoint_fp8_serialized=True,
            ),
            MagicMock(),
        )

    selector.assert_not_called()


def test_modelopt_kv_cache_method_rejects_nvfp4_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptKVCacheMethod(
            SimpleNamespace(kv_cache_quant_algo="NVFP4")
        )


def test_modelopt_kv_cache_method_allows_fp8_on_sm12x(
    sm12x_platform,
) -> None:
    modelopt.ModelOptKVCacheMethod(
        SimpleNamespace(kv_cache_quant_method="FP8")
    )


def test_modelopt_mxfp8_linear_constructor_rejects_on_sm12x(
    sm12x_platform,
) -> None:
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptMxFp8LinearMethod(
            SimpleNamespace(is_checkpoint_mxfp8_serialized=True)
        )


def test_modelopt_mxfp8_moe_constructor_rejects_on_sm12x_before_backend_selection(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(modelopt, "select_mxfp8_moe_backend", selector)

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptMxFp8FusedMoE(
            SimpleNamespace(is_checkpoint_mxfp8_serialized=True),
            MagicMock(),
        )

    selector.assert_not_called()


def test_modelopt_w4a16_nvfp4_linear_constructor_uses_native_dequant_on_sm12x(
    sm12x_platform,
) -> None:
    # GB10 serves W4A16 NVFP4 dense natively (dequant FP4->bf16 at load); the
    # constructor pins that path instead of rejecting / selecting Marlin.
    method = modelopt.ModelOptNvFp4W4A16LinearMethod(
        SimpleNamespace(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
        )
    )
    assert method._gb10_w4a16_dequant is True
    assert method.kernel is None


def test_modelopt_w4a16_nvfp4_moe_constructor_selects_native_b12x_on_sm12x(
    sm12x_platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    captured = {}

    def fake_select(config, weight_key, activation_key):
        captured["activation_key"] = activation_key
        return NvFp4MoeBackend.FLASHINFER_B12X_W4A16, MagicMock()

    monkeypatch.setattr(modelopt, "select_nvfp4_moe_backend", fake_select)

    method = modelopt.ModelOptNvFp4FusedMoE(
        SimpleNamespace(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
        ),
        MagicMock(),
    )

    # W4A16 -> activation_key None; native b12x W4A16 backend is selected.
    assert captured["activation_key"] is None
    assert method.nvfp4_backend == NvFp4MoeBackend.FLASHINFER_B12X_W4A16
