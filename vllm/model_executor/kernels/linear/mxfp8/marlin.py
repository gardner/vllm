# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class MarlinMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A16 GEMM via Marlin (SM80+)."""

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
                "Marlin MXFP8 dense fallback is not supported on GB10/SM12x; "
                "use FlashInfer CUTLASS native MXFP8 dense support after "
                "correctness evidence is available, or keep the path "
                "unselected.",
            )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            is_fp8_marlin_supported,
        )

        if is_fp8_marlin_supported():
            return True, None
        return False, "Marlin FP8 not available"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_mxfp8_layer_for_marlin,
        )

        prepare_mxfp8_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_mxfp8_marlin_linear,
        )

        return apply_mxfp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
