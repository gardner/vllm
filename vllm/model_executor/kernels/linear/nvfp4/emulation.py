# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    kE2M1ToFloat_handle,
    run_nvfp4_emulations,
)
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class EmulationNvFp4LinearKernel(NvFp4LinearKernel):
    """Software emulation fallback for NVFP4 (dequant → BF16 matmul)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            is_sm12x = current_platform.is_device_capability_family(120)
        else:
            is_sm12x = compute_capability // 10 == 12
        if is_sm12x:
            return (
                False,
                "NVFP4 emulation fallback is not supported on GB10/SM12x; "
                "use FlashInfer b12x, FlashInfer CUTLASS, or CUTLASS native "
                "NVFP4 dense backends instead.",
            )
        # Always available as a last-resort fallback off GB10.
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Move the E2M1 lookup table to the device now, because
        # `.to(device)` is not allowed during CUDA graph capture.
        kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(layer.weight.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = run_nvfp4_emulations(
            x=x,
            input_global_scale=layer.input_global_scale_inv,
            weight=layer.weight,
            weight_scale_swizzled=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            swizzle=False,
        )
        if bias is not None:
            out = out + bias
        return out
