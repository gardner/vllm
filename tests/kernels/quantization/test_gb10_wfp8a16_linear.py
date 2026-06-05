# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

try:
    import vllm._C  # noqa: F401
except ModuleNotFoundError:
    import vllm.platforms as vllm_platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    class _TestPlatform(UnspecifiedPlatform):
        simple_compile_backend = "eager"

        def fp8_dtype(self) -> torch.dtype:
            return torch.float8_e4m3fn

        def is_rocm(self) -> bool:
            return False

        def has_device_capability(self, capability: int) -> bool:
            return False

        def is_device_capability_family(self, capability: int) -> bool:
            return False

        def import_kernels(self) -> None:
            return None

    vllm_platforms.current_platform = _TestPlatform()

import vllm.model_executor.kernels.linear as linear_kernels  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum  # noqa: E402


class _Sm12xCudaPlatform:
    _enum = PlatformEnum.CUDA

    def get_device_capability(self) -> tuple[int, int]:
        return (12, 1)

    def has_device_capability(self, capability: int) -> bool:
        return capability <= 121

    def is_cuda(self) -> bool:
        return True

    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


def _make_wfp8a16_marlin_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        linear_kernels.MarlinFP8ScaledMMLinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        linear_kernels.MarlinFP8ScaledMMLinearKernel,
        "can_implement",
        classmethod(lambda cls, config: (True, None)),
    )


@pytest.mark.parametrize("backend", ["auto", "marlin"])
def test_sm12x_wfp8a16_marlin_fallback_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    monkeypatch.setattr(linear_kernels, "current_platform", _Sm12xCudaPlatform())
    monkeypatch.setattr(linear_kernels, "_get_linear_backend", lambda: backend)
    _make_wfp8a16_marlin_supported(monkeypatch)

    config = linear_kernels.FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8DynamicTensorSym,
        weight_shape=(128, 128),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        linear_kernels.choose_scaled_mm_linear_kernel(
            config,
            linear_kernels._POSSIBLE_WFP8A16_KERNELS,
        )


def test_sm12x_forced_wfp8a16_marlin_fallback_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(linear_kernels, "current_platform", _Sm12xCudaPlatform())
    _make_wfp8a16_marlin_supported(monkeypatch)

    config = linear_kernels.FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8DynamicTensorSym,
        weight_shape=(128, 128),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        linear_kernels.choose_scaled_mm_linear_kernel(
            config,
            linear_kernels._POSSIBLE_WFP8A16_KERNELS,
            force_kernel=linear_kernels.MarlinFP8ScaledMMLinearKernel,
        )
