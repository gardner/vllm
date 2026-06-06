# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


class MarlinMxFp4LinearKernel(MxFp4LinearKernel):
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
                "Marlin MXFP4 dense fallback is not supported on GB10/SM12x; "
                "use FlashInfer CUTLASS native MXFP4 dense support after "
                "correctness evidence is available, or keep the path "
                "unselected.",
            )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            is_fp4_marlin_supported,
        )

        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, c: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_fp4_layer_for_marlin,
        )

        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
        )

        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=None,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
