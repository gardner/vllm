# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GB10/SM12x native W4A16 NVFP4 MoE backend wiring.

``nvidia/Qwen3.6-35B-A3B-NVFP4`` declares its MoE experts as ``W4A16_NVFP4``
(FP4 weights x bf16 activations) in the checkpoint's ``quantized_layers`` map.
On GB10 this must select the native ``FlashInferB12xW4A16Experts`` backend
(FlashInfer b12x ``quant_mode="w4a16"``), NOT the Marlin weight-only fallback
(which is rejected on SM12x and whose ``gptq_marlin_repack`` op is absent from
the sm_121a build).

These are static wiring assertions (no GPU needed). The end-to-end runtime
selection ("Using 'FLASHINFER_B12X_W4A16' NvFp4 MoE backend") is exercised by
the GB10 model smoke.
"""

from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
    FlashInferB12xW4A16Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    FLASHINFER_NVFP4_MOE_BACKENDS,
    NvFp4MoeBackend,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)


def test_w4a16_backend_maps_to_w4a16_experts():
    assert backend_to_kernel_cls(NvFp4MoeBackend.FLASHINFER_B12X_W4A16) == [
        FlashInferB12xW4A16Experts
    ]
    assert NvFp4MoeBackend.FLASHINFER_B12X_W4A16 in FLASHINFER_NVFP4_MOE_BACKENDS


def test_w4a16_experts_accept_w4a16_scheme_only():
    # W4A16: NVFP4 weights, no activation quant (activation_key is None).
    assert FlashInferB12xW4A16Experts._supports_quant_scheme(kNvfp4Static, None)
    # The W4A16 class must NOT also claim the W4A4 (dynamic activation) scheme.
    assert not FlashInferB12xW4A16Experts._supports_quant_scheme(
        kNvfp4Static, kNvfp4Dynamic
    )


def test_w4a4_experts_reject_w4a16_scheme():
    # The base W4A4 class must NOT match the W4A16 (None activation) scheme, so
    # the selector cannot mis-route a W4A16 checkpoint to the W4A4 kernel
    # (which would quantize activations and corrupt outputs).
    assert not FlashInferB12xExperts._supports_quant_scheme(kNvfp4Static, None)
    assert FlashInferB12xExperts._supports_quant_scheme(kNvfp4Static, kNvfp4Dynamic)


def test_w4a16_experts_quant_mode_is_w4a16():
    # _quant_mode drives the flashinfer b12x ``quant_mode`` kwarg in apply().
    # It ignores self, so call it unbound with the class as the receiver.
    assert FlashInferB12xW4A16Experts._quant_mode(FlashInferB12xW4A16Experts) == "w4a16"
    assert FlashInferB12xExperts._quant_mode(FlashInferB12xExperts) == "nvfp4"
