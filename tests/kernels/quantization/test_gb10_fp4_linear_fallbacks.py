# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

try:
    import vllm._C  # noqa: F401
except (ImportError, ModuleNotFoundError):
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

import vllm.model_executor.kernels.linear as linear_kernels


@pytest.mark.parametrize(
    "kernel_cls, reason_fragment",
    [
        (
            linear_kernels.FbgemmNvFp4LinearKernel,
            "FBGEMM NVFP4 dense backend",
        ),
        (
            linear_kernels.EmulationNvFp4LinearKernel,
            "NVFP4 emulation fallback",
        ),
        (
            linear_kernels.MarlinNvFp4LinearKernel,
            "Marlin NVFP4 dense fallback",
        ),
        (
            linear_kernels.MarlinMxFp4LinearKernel,
            "Marlin MXFP4 dense fallback",
        ),
    ],
)
def test_sm12x_fp4_fallback_kernels_report_unsupported(
    kernel_cls,
    reason_fragment: str,
) -> None:
    supported, reason = kernel_cls.is_supported(compute_capability=121)

    assert not supported
    assert reason is not None
    assert "GB10/SM12x" in reason
    assert reason_fragment in reason
