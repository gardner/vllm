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
from vllm.platforms import PlatformEnum  # noqa: E402


class _Sm12xCudaPlatform:
    _enum = PlatformEnum.CUDA

    def has_device_capability(self, capability: int) -> bool:
        return capability <= 121

    def is_device_capability_family(self, capability: int) -> bool:
        return capability == 120


def _make_mxfp8_fallbacks_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    for kernel_cls in (
        linear_kernels.MarlinMxfp8LinearKernel,
        linear_kernels.EmulationMxfp8LinearKernel,
    ):
        monkeypatch.setattr(
            kernel_cls,
            "is_supported",
            classmethod(lambda cls: (True, None)),
        )
        monkeypatch.setattr(
            kernel_cls,
            "can_implement",
            classmethod(lambda cls, config: (True, None)),
        )


@pytest.mark.parametrize(
    "kernel_cls, reason_fragment",
    [
        (
            linear_kernels.FlashInferCutlassMxfp8LinearKernel,
            "FlashInfer CUTLASS MXFP8 dense",
        ),
        (
            linear_kernels.MarlinMxfp8LinearKernel,
            "Marlin MXFP8 dense fallback",
        ),
        (
            linear_kernels.EmulationMxfp8LinearKernel,
            "MXFP8 emulation fallback",
        ),
    ],
)
def test_sm12x_mxfp8_dense_kernels_report_unsupported(
    kernel_cls,
    reason_fragment: str,
) -> None:
    supported, reason = kernel_cls.is_supported(compute_capability=121)

    assert not supported
    assert reason is not None
    assert "GB10/SM12x" in reason
    assert reason_fragment in reason


def test_sm12x_linear_backend_flashinfer_cutlass_mxfp8_dense_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(linear_kernels, "current_platform", _Sm12xCudaPlatform())
    monkeypatch.setattr(
        linear_kernels,
        "_get_linear_backend",
        lambda: "flashinfer_cutlass",
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        linear_kernels.init_mxfp8_linear_kernel()


@pytest.mark.parametrize(
    "kernel_cls, reason_fragment",
    [
        (
            linear_kernels.MarlinMxfp8LinearKernel,
            "Marlin MXFP8 dense fallback",
        ),
        (
            linear_kernels.EmulationMxfp8LinearKernel,
            "MXFP8 emulation fallback",
        ),
    ],
)
def test_sm12x_mxfp8_fallback_kernels_report_unsupported(
    kernel_cls,
    reason_fragment: str,
) -> None:
    supported, reason = kernel_cls.is_supported(compute_capability=121)

    assert not supported
    assert reason is not None
    assert "GB10/SM12x" in reason
    assert reason_fragment in reason


@pytest.mark.parametrize("backend", ["marlin", "emulation"])
def test_sm12x_linear_backend_mxfp8_dense_fallback_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    monkeypatch.setattr(linear_kernels, "current_platform", _Sm12xCudaPlatform())
    monkeypatch.setattr(linear_kernels, "_get_linear_backend", lambda: backend)
    _make_mxfp8_fallbacks_supported(monkeypatch)

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        linear_kernels.init_mxfp8_linear_kernel()


def test_sm12x_auto_rejects_mxfp8_dense_when_no_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(linear_kernels, "current_platform", _Sm12xCudaPlatform())
    monkeypatch.setattr(linear_kernels, "_get_linear_backend", lambda: "auto")
    monkeypatch.setattr(
        linear_kernels,
        "_POSSIBLE_MXFP8_KERNELS",
        {
            PlatformEnum.CUDA: [
                linear_kernels.MarlinMxfp8LinearKernel,
                linear_kernels.EmulationMxfp8LinearKernel,
            ]
        },
    )
    _make_mxfp8_fallbacks_supported(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        linear_kernels.init_mxfp8_linear_kernel()

    message = str(exc_info.value)
    assert "MarlinMxfp8LinearKernel" in message
    assert "EmulationMxfp8LinearKernel" in message
    assert "not supported on GB10/SM12x" in message
