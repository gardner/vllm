# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsScheme,
    CompressedTensorsW4A4Fp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    TransformTuple,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    _is_sm12x_device,
)

__all__ = [
    "_gb10_qutlass_nvfp4_transform_unsupported_reason",
    "is_qutlass_fp4_scheme",
    "QutlassNvFP4LinearMethod",
]


def _gb10_qutlass_nvfp4_transform_unsupported_reason() -> str | None:
    if not _is_sm12x_device():
        return None
    return (
        "CompressedTensors Qutlass NVFP4 transform loading is "
        "not supported on GB10/SM12x. QutlassNvFP4LinearMethod.apply is "
        "not implemented today, "
        "so the transformed NVFP4 path can prove reachability but not native "
        "GB10 transformed NVFP4 correctness evidence."
    )


def is_qutlass_fp4_scheme(
    quant_scheme: CompressedTensorsScheme | None,
    input_tfms: dict[int, TransformTuple],
) -> bool:
    return (
        isinstance(quant_scheme, (CompressedTensorsW4A4Fp4,))
        and len(input_tfms) == 1
        and input_tfms[0].scheme.head_dim == quant_scheme.group_size
    )


class QutlassNvFP4LinearMethod(CompressedTensorsLinearTransformMethod):
    def __init__(
        self,
        quant_method,
        input_tfms,
        output_tfms,
    ):
        unsupported_reason = _gb10_qutlass_nvfp4_transform_unsupported_reason()
        if unsupported_reason is not None:
            raise ValueError(unsupported_reason)
        super().__init__(quant_method, input_tfms, output_tfms)

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        # initializes fp4 qparams
        assert isinstance(layer.scheme, (CompressedTensorsW4A4Fp4,))
        ret = super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        assert self.input_transform is not None
        assert len(self.input_transform.weight) == 1
        assert self.input_transform.weight[0].size(0) == layer.scheme.group_size

        return ret

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError()
