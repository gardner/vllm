# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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

import vllm._custom_ops as custom_ops  # noqa: E402
import vllm.model_executor.kernels.linear.nvfp4.flashinfer as flashinfer_nvfp4  # noqa: E402
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (  # noqa: E402
    FlashInferB12xNvFp4LinearKernel,
)


def _layer(output_size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        output_size_per_partition=output_size,
        input_global_scale_inv=torch.tensor([1.0], dtype=torch.float32),
        weight=torch.empty(output_size, 8, dtype=torch.uint8),
        weight_scale=torch.empty(128, 4, dtype=torch.float8_e4m3fn),
        alpha=torch.tensor([1.0], dtype=torch.float32),
        weights_padding_cols=0,
    )


def _kernel() -> FlashInferB12xNvFp4LinearKernel:
    return FlashInferB12xNvFp4LinearKernel.__new__(FlashInferB12xNvFp4LinearKernel)


def test_scaled_fp4_quant_b12x_uses_flashinfer_128x4_quantizer(
    monkeypatch,
) -> None:
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    global_scale = torch.tensor([1.0], dtype=torch.float32)
    calls: dict[str, object] = {}

    def fake_quant(a: torch.Tensor, a_global_sf: torch.Tensor):
        calls["quant_shape"] = tuple(a.shape)
        calls["global_scale"] = a_global_sf
        return (
            torch.empty(a.shape[0], a.shape[1] // 2, dtype=torch.uint8),
            torch.empty(128, 4, dtype=torch.float8_e4m3fn),
        )

    monkeypatch.setattr(
        custom_ops,
        "flashinfer_quant_nvfp4_128x4_sf_layout",
        fake_quant,
    )

    out, out_scale = custom_ops.scaled_fp4_quant(x, global_scale, backend="b12x")

    assert out.shape == (6, 8)
    assert out_scale.dtype == torch.float8_e4m3fn
    assert calls["quant_shape"] == (6, 16)
    assert calls["global_scale"] is global_scale


def test_b12x_bf16_uses_flashinfer_128x4_activation_quantizer(
    monkeypatch,
) -> None:
    layer = _layer()
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    calls: dict[str, object] = {}

    def fail_scaled_fp4_quant(*args, **kwargs):
        raise AssertionError("SM12x BF16 b12x path must not call _C scaled_fp4_quant")

    def fake_quant(a: torch.Tensor, a_global_sf: torch.Tensor):
        calls["quant_shape"] = tuple(a.shape)
        calls["global_scale"] = a_global_sf
        return (
            torch.empty(a.shape[0], a.shape[1] // 2, dtype=torch.uint8),
            torch.empty(128, 4, dtype=torch.float8_e4m3fn),
        )

    def fake_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str,
        block_size: int = 16,
        use_nvfp4: bool = True,
    ):
        calls["mm_shape"] = tuple(a.shape)
        calls["backend"] = backend
        assert block_scale_a.dtype == torch.float8_e4m3fn
        return torch.zeros(a.shape[0], b.shape[0], dtype=out_dtype)

    monkeypatch.setattr(flashinfer_nvfp4, "scaled_fp4_quant", fail_scaled_fp4_quant)
    monkeypatch.setattr(
        flashinfer_nvfp4,
        "flashinfer_quant_nvfp4_128x4_sf_layout",
        fake_quant,
    )
    monkeypatch.setattr(flashinfer_nvfp4, "flashinfer_scaled_fp4_mm", fake_mm)

    out = _kernel().apply_weights(layer, x)

    assert out.shape == (2, 3, layer.output_size_per_partition)
    assert calls["quant_shape"] == (6, 16)
    assert calls["global_scale"] is layer.input_global_scale_inv
    assert calls["mm_shape"] == (6, 8)
    assert calls["backend"] == "b12x"


def test_b12x_fp16_keeps_scaled_quant_cutlass_fallback(monkeypatch) -> None:
    layer = _layer()
    x = torch.randn(2, 16, dtype=torch.float16)
    calls: dict[str, object] = {}

    def fail_flashinfer_quant(*args, **kwargs):
        raise AssertionError("FP16 fallback should use the existing quantization path")

    def fake_scaled_fp4_quant(
        input: torch.Tensor,
        input_global_scale: torch.Tensor,
        is_sf_swizzled_layout: bool,
        backend: str,
        padded_n: int | None = None,
    ):
        calls["quant_shape"] = tuple(input.shape)
        calls["quant_backend"] = backend
        assert is_sf_swizzled_layout
        assert padded_n is None
        return (
            torch.empty(input.shape[0], input.shape[1] // 2, dtype=torch.uint8),
            torch.empty(128, 4, dtype=torch.float8_e4m3fn),
        )

    def fake_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str,
        block_size: int = 16,
        use_nvfp4: bool = True,
    ):
        calls["mm_shape"] = tuple(a.shape)
        calls["mm_backend"] = backend
        return torch.zeros(a.shape[0], b.shape[0], dtype=out_dtype)

    monkeypatch.setattr(
        flashinfer_nvfp4,
        "flashinfer_quant_nvfp4_128x4_sf_layout",
        fail_flashinfer_quant,
    )
    monkeypatch.setattr(flashinfer_nvfp4, "scaled_fp4_quant", fake_scaled_fp4_quant)
    monkeypatch.setattr(flashinfer_nvfp4, "flashinfer_scaled_fp4_mm", fake_mm)

    out = _kernel().apply_weights(layer, x)

    assert out.shape == (2, layer.output_size_per_partition)
    assert calls["quant_shape"] == (2, 16)
    assert calls["quant_backend"] == "b12x"
    assert calls["mm_shape"] == (2, 8)
    assert calls["mm_backend"] == "cutlass"
