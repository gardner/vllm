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

import vllm.model_executor.layers.fused_moe.config as fused_moe_config  # noqa: E402
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as marlin_moe  # noqa: E402
from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEQuantConfig,
    fp8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (  # noqa: E402
    BatchedMarlinExperts,
    MarlinExperts,
)
from vllm.platforms import PlatformEnum  # noqa: E402


class _Sm12xCudaPlatform:
    _enum = PlatformEnum.CUDA

    def fp8_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def get_device_capability(self) -> tuple[int, int]:
        return (12, 1)

    def has_device_capability(self, capability: int) -> bool:
        return capability <= 121

    def is_cuda(self) -> bool:
        return True

    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120

    def is_rocm(self) -> bool:
        return False


def _patch_sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _Sm12xCudaPlatform()
    monkeypatch.setattr(fused_moe_config, "current_platform", platform)
    monkeypatch.setattr(marlin_moe, "current_platform", platform)


def _make_fp8_w8a16_quant_config() -> FusedMoEQuantConfig:
    return fp8_w8a16_moe_quant_config(
        w1_scale=torch.ones(1, dtype=torch.float32),
        w2_scale=torch.ones(1, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("experts_cls", "ctor_kwargs"),
    [
        (MarlinExperts, {}),
        (BatchedMarlinExperts, {"max_num_tokens": 64, "num_dispatchers": 1}),
    ],
)
def test_sm12x_fp8_w8a16_marlin_moe_constructor_rejects(
    monkeypatch: pytest.MonkeyPatch,
    experts_cls,
    ctor_kwargs,
) -> None:
    _patch_sm12x_platform(monkeypatch)

    def _should_not_be_called() -> None:
        raise AssertionError("GB10 should reject FP8 W8A16 Marlin before setup")

    monkeypatch.setattr(marlin_moe, "get_marlin_input_dtype", _should_not_be_called)

    quant_config = _make_fp8_w8a16_quant_config()

    with pytest.raises(ValueError, match="FP8 W8A16 Marlin fallback.*GB10/SM12x"):
        experts_cls(
            moe_config=make_dummy_moe_config(),
            quant_config=quant_config,
            **ctor_kwargs,
        )
