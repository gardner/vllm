"""Shared GB10 release contract constants for release and evidence tools."""

from __future__ import annotations

import os
import re
from pathlib import Path

FLASHINFER_RUNTIME_DISTRIBUTIONS = (
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)

REQUIRED_FLASHINFER_COMPONENTS = (
    "flashinfer_python",
    "flashinfer_cubin",
    "flashinfer_jit_cache",
)

REQUIRED_SOURCE_DEPENDENCIES = (
    "deepgemm",
    "flashmla",
    "triton_kernels",
)

REQUIRED_GB10_SUPPORT_MATRIX = {
    "flashinfer_b12x_nvfp4_dense": "supported_native",
    "flashinfer_cutlass_nvfp4_dense": "supported_native",
    "flashinfer_nvfp4_quantization": "supported_native",
    "modelopt_fp4_quantization": "supported_native",
    "flashinfer_attention_fa2": "supported_native",
    "flashinfer_b12x_non_ep_moe": "supported_native",
    "flashinfer_cutlass_non_ep_moe": "supported_native",
    "flashmla_attention": "supported_native",
    "flashinfer_gdn_prefill": "supported_native",
    "gb10_attention_trtllm_gen_to_flashinfer_fa2": "supported_routed",
    "gb10_attention_public_flashattention_to_flashinfer_or_flashmla": (
        "supported_routed"
    ),
    "gb10_moe_trtllm_gen_to_flashinfer_non_ep": "supported_routed",
    "public_flashattention_runtime": "not_supported",
    "flashinfer_trtllm_nvfp4_dense": "not_supported",
    "flashinfer_trtllm_mxfp4_moe": "not_supported",
    "flashinfer_cutedsl_nvfp4_moe": "not_supported",
    "trtllm_gen_attention": "not_supported",
    "triton_attention_fallback": "not_supported",
    "flex_attention_fallback": "not_supported",
    "turboquant_attention": "not_supported",
    "gdn_prefill_triton_fallback": "not_supported",
    "gdn_prefill_cutedsl_backend": "not_supported",
    "trtllm_gen_moe": "not_supported",
    "public_fp8_quantization": "not_supported",
    "deepseek_v4_fp8_quantization": "not_supported",
    "torchao_fp8_activation_quantization": "not_supported",
    "torchao_weight_quantization": "not_supported",
    "bitsandbytes_quantization": "not_supported",
    "awq_quantization": "not_supported",
    "gptq_quantization": "not_supported",
    "inc_quantization": "not_supported",
    "gguf_quantization": "not_supported",
    "humming_quantization": "not_supported",
    "humming_mxfp4_moe_backend": "not_supported",
    "rocm_aiter_unquantized_moe": "not_supported",
    "unquantized_moe_triton_fallback": "not_supported",
    "rocm_aiter_fp8_moe": "not_supported",
    "deep_gemm_fp8_moe": "not_supported",
    "triton_fp8_moe": "not_supported",
    "vllm_cutlass_fp8_moe": "not_supported",
    "rocm_aiter_mxfp4_moe": "not_supported",
    "gpt_oss_triton_mxfp4_moe": "not_supported",
    "marlin_nvfp4_fallback": "not_supported",
    "fbgemm_nvfp4_dense": "not_supported",
    "modelopt_w4a16_nvfp4_checkpoint_loading": "not_supported",
    "modelopt_nvfp4_kv_cache_loading": "not_supported",
    "marlin_mxfp4_fallback": "not_supported",
    "mxfp4_moe_fallback": "not_supported",
    "public_mxfp4_quantization": "not_supported",
    "fp8_w8a16_marlin_fallback": "not_supported",
    "fp8_w8a16_moe_fallback": "not_supported",
    "int8_moe_triton_fallback": "not_supported",
    "wna16_moe_fallback": "not_supported",
    "compressed_tensors_wna16_dense_loading": "not_supported",
    "compressed_tensors_wna16_moe_fallback": "not_supported",
    "moe_wna16_legacy_fallback": "not_supported",
    "mxfp8_dense_fallback": "not_supported",
    "mxfp8_moe_fallback": "not_supported",
    "modelopt_fp8_quantization": "not_supported",
    "modelopt_mxfp8_quantization": "not_supported",
    "modelopt_mixed_quantization": "not_supported",
    "fbgemm_fp8_quantization": "not_supported",
    "experts_int8_quantization": "not_supported",
    "fp_quant_fp4_quantization": "not_supported",
    "online_fp8_quantization": "not_supported",
    "online_mxfp8_quantization": "not_supported",
    "online_mxfp4_quantization": "not_supported",
    "online_int8_moe_quantization": "not_supported",
    "compressed_tensors_w8a8_mxfp8_dense_loading": "not_supported",
    "compressed_tensors_w8a8_mxfp8_moe_loading": "not_supported",
    "quark_nvfp4_checkpoint_loading": "not_supported",
    "quark_ocp_mx_checkpoint_loading": "not_supported",
    "quark_w4a8_mxfp4_fp8_checkpoint_loading": "not_supported",
    "quark_w4a8_fp8_moe_loading": "not_supported",
    "quark_w8a8_fp8_checkpoint_loading": "not_supported",
    "quark_w8a8_int8_checkpoint_loading": "not_supported",
    "quark_w8a8_fp8_moe_loading": "not_supported",
    "quark_w8a8_int8_moe_loading": "not_supported",
    "compressed_tensors_w4a8_fp8_loading": "not_supported",
    "compressed_tensors_w4a8_int_dense_loading": "not_supported",
    "compressed_tensors_w4a8_int_moe_loading": "not_supported",
    "compressed_tensors_w8a16_fp8_loading": "not_supported",
    "compressed_tensors_w8a8_fp8_loading": "not_supported",
    "compressed_tensors_w8a8_fp8_moe_loading": "not_supported",
    "compressed_tensors_w8a8_int_dense_loading": "not_supported",
    "compressed_tensors_w8a8_int_moe_loading": "not_supported",
    "compressed_tensors_w4a4_nvfp4_dense_loading": "supported_native",
    "compressed_tensors_w4a4_nvfp4_moe_loading": "supported_native",
    "compressed_tensors_qutlass_nvfp4_transform_loading": "not_supported",
    "compressed_tensors_w4a4_mxfp4_dense_loading": "not_supported",
    "compressed_tensors_w4a4_mxfp4_moe_loading": "not_supported",
    "compressed_tensors_w4a16_nvfp4_loading": "not_supported",
    "compressed_tensors_w4a16_nvfp4_moe_loading": "not_supported",
    "deepseek_v4_deep_gemm_mega_moe": "deferred",
    "flashinfer_b12x_ep_all2all_eplb": "deferred",
    "flashinfer_cudnn_nvfp4_dense": "deferred",
    "multi_spark_ep_all2all_eplb": "deferred",
}

GB10_SUPPORTED_ROUTED_PATH_REASONS = {
    "gb10_attention_trtllm_gen_to_flashinfer_fa2": (
        "TRTLLM Gen attention is unavailable on SM121; route GB10 attention "
        "through validated FlashInfer FA2 until SM121 TRTLLM Gen artifacts "
        "exist."
    ),
    "gb10_attention_public_flashattention_to_flashinfer_or_flashmla": (
        "Public FlashAttention runtime is not supported for the first GB10 "
        "runtime path; route GB10 attention through validated FlashInfer or "
        "FlashMLA backends."
    ),
    "gb10_moe_trtllm_gen_to_flashinfer_non_ep": (
        "TRTLLM Gen MoE is unavailable on SM121; route first-path non-EP "
        "NVFP4 MoE through validated FlashInfer b12x or FlashInfer CUTLASS "
        "backends."
    ),
}

GB10_NOT_SUPPORTED_PATH_REASONS = {
    "public_flashattention_runtime": (
        "Public FlashAttention runtime is not validated for the GB10 first "
        "release path; route attention through FlashInfer or FlashMLA."
    ),
    "flashinfer_trtllm_nvfp4_dense": (
        "FlashInfer TRTLLM dense NVFP4 is not validated on GB10/SM12x; use "
        "FlashInfer b12x or FlashInfer CUTLASS dense NVFP4 evidence instead."
    ),
    "flashinfer_trtllm_mxfp4_moe": (
        "FlashInfer TRTLLM MXFP4 MoE is an SM100-family path today and is not "
        "validated on GB10/SM12x; keep it unselected until native SM121A MXFP4 "
        "MoE evidence exists."
    ),
    "flashinfer_cutedsl_nvfp4_moe": (
        "Generic FlashInfer CuteDSL and batched CuteDSL NVFP4 MoE variants are "
        "not validated native GB10 evidence; use FlashInfer b12x or FlashInfer "
        "CUTLASS NVFP4 MoE instead."
    ),
    "trtllm_gen_attention": (
        "TRTLLM Gen attention artifacts and metadata do not support SM121; "
        "route GB10 attention through FlashInfer or FlashMLA."
    ),
    "triton_attention_fallback": (
        "Generic Triton attention fallback can prove reachability, but it is "
        "not native GB10 attention correctness evidence; route GB10 attention "
        "through FlashInfer or FlashMLA until native SM12x Triton attention "
        "runtime evidence exists."
    ),
    "flex_attention_fallback": (
        "PyTorch FlexAttention fallback can prove reachability, but it is not "
        "native GB10 attention correctness evidence; route GB10 attention "
        "through FlashInfer or FlashMLA until native SM12x FlexAttention "
        "runtime evidence exists."
    ),
    "turboquant_attention": (
        "TurboQuant KV-cache compression can prove reachability, but it is not "
        "native GB10 attention or KV-cache correctness evidence; route GB10 "
        "attention through FlashInfer with validated FP8 KV cache until native "
        "SM12x TurboQuant runtime evidence exists."
    ),
    "gdn_prefill_triton_fallback": (
        "GDN prefill Triton/FLA fallback can prove reachability, but it is not "
        "native GB10 GDN prefill correctness evidence; use the native "
        "FlashInfer SM12x GDN prefill backend or keep GDN models unselected."
    ),
    "gdn_prefill_cutedsl_backend": (
        "GDN prefill CuteDSL is SM100-family evidence today and must not "
        "satisfy GB10 GDN prefill release evidence; use the native FlashInfer "
        "SM12x GDN prefill backend or keep GDN models unselected."
    ),
    "trtllm_gen_moe": (
        "TRTLLM Gen MoE rejects SM121 today and must not satisfy GB10 MoE "
        "release evidence unless native SM121A TRTLLM fused-MoE artifacts "
        "and runtime evidence exist."
    ),
    "rocm_aiter_fp8_moe": (
        "AITER FP8 MoE is a ROCm-specific backend and is not a native GB10 "
        "CUDA path."
    ),
    "deep_gemm_fp8_moe": (
        "DeepGEMM FP8 MoE is selectable through moe_backend='deep_gemm' and "
        "DeepGEMM env flags, but it is not native GB10 release evidence until "
        "GB10 DeepGEMM FP8 MoE artifacts, correctness, and runtime evidence "
        "exist."
    ),
    "triton_fp8_moe": (
        "Generic Triton FP8 MoE is selectable through moe_backend='triton' and "
        "auto-selection, but it is not native GB10 release evidence until GB10 "
        "Triton FP8 MoE correctness and runtime evidence exist."
    ),
    "vllm_cutlass_fp8_moe": (
        "vLLM CUTLASS FP8 MoE is selectable through moe_backend='cutlass' and "
        "allow_vllm_cutlass=True call sites, but it is not native GB10 release "
        "evidence until GB10 vLLM CUTLASS FP8 MoE artifacts, correctness, and "
        "runtime evidence exist."
    ),
    "rocm_aiter_unquantized_moe": (
        "AITER unquantized MoE is a ROCm-specific backend and is not a "
        "native GB10 CUDA path."
    ),
    "unquantized_moe_triton_fallback": (
        "Generic Triton unquantized MoE fallback can prove reachability, but "
        "it is not native GB10 unquantized MoE evidence."
    ),
    "marlin_nvfp4_fallback": (
        "Marlin can prove fallback serving reachability, but it is not native "
        "GB10 NVFP4 Tensor Core evidence."
    ),
    "fbgemm_nvfp4_dense": (
        "FBGEMM NVFP4 dense can prove backend reachability when fbgemm_gpu is "
        "installed, but it is not native GB10 NVFP4 evidence until SM121A "
        "FBGEMM artifacts and correctness evidence exist."
    ),
    "modelopt_w4a16_nvfp4_checkpoint_loading": (
        "ModelOpt W4A16 NVFP4 checkpoint loading is not validated on "
        "GB10/SM12x; reject it until native GB10 W4A16 NVFP4 dense and MoE "
        "correctness evidence exists."
    ),
    "modelopt_nvfp4_kv_cache_loading": (
        "ModelOpt NVFP4 KV-cache loading can auto-select vLLM "
        "kv_cache_dtype='nvfp4', but the current GB10 release evidence only "
        "validates FP8 E4M3 KV cache; reject it until native SM12x NVFP4 "
        "KV-cache correctness evidence exists."
    ),
    "marlin_mxfp4_fallback": (
        "Marlin can prove MXFP4 dense/MoE fallback reachability, but it is "
        "not native GB10 MXFP4 Tensor Core evidence."
    ),
    "mxfp4_moe_fallback": (
        "MXFP4 MoE Marlin, batched Marlin, emulation, and CPU fallbacks can "
        "prove reachability, but they are not native GB10 MXFP4 Tensor Core "
        "evidence."
    ),
    "rocm_aiter_mxfp4_moe": (
        "AITER MXFP4 MoE backends are ROCm-specific paths today; reject them "
        "on GB10/SM12x until native GB10 CUDA MXFP4 MoE correctness evidence "
        "exists."
    ),
    "gpt_oss_triton_mxfp4_moe": (
        "GPT-OSS Triton MXFP4 MoE can select triton_kernels OAI MXFP4/SwiGLU "
        "kernels today; reject it on GB10/SM12x until native GB10 GPT-OSS "
        "MXFP4 MoE correctness evidence exists."
    ),
    "public_mxfp4_quantization": (
        "Public MXFP4 quantization can reach unquantized linear/attention "
        "handling and MXFP4 MoE backend selection today; reject it until "
        "native GB10 public MXFP4 correctness evidence exists."
    ),
    "public_fp8_quantization": (
        "Public FP8 quantization can reach online FP8 quantization, FP8 "
        "scaled-mm dense kernel selection, and FP8 MoE backend selection today; "
        "reject it until native GB10 public FP8 correctness evidence exists."
    ),
    "deepseek_v4_fp8_quantization": (
        "DeepSeek V4 FP8 quantization can reach FP8 block-quantized "
        "linear/attention layers and FP8, MXFP4, or ModelOpt NVFP4 MoE "
        "dispatch today; reject it until native GB10 DeepSeek V4 correctness "
        "evidence exists."
    ),
    "torchao_fp8_activation_quantization": (
        "TorchAO FP8 activation quantization can call "
        "torchao.quantization.quantize_ and hardware-specific tensor packing "
        "today; reject it until native GB10 TorchAO FP8 activation correctness "
        "evidence exists."
    ),
    "torchao_weight_quantization": (
        "TorchAO weight quantization can call torchao.quantization.quantize_ "
        "and hardware-specific tensor packing today; reject it until native "
        "GB10 TorchAO weight correctness evidence exists."
    ),
    "bitsandbytes_quantization": (
        "BitsAndBytes quantization can select bitsandbytes 4-bit linear "
        "kernels, bitsandbytes 8-bit matmul kernels, and BitsAndBytesMoE "
        "handling today; reject it until native GB10 BitsAndBytes correctness "
        "evidence exists."
    ),
    "awq_quantization": (
        "AWQ quantization can select AWQ dense kernels, AWQ-Marlin dense "
        "kernels, AWQ-Marlin MoE, and Moe WNA16 fallback handling today; "
        "reject it until native GB10 AWQ correctness evidence exists."
    ),
    "gptq_quantization": (
        "GPTQ quantization can select GPTQ dense kernel routing, "
        "AutoGPTQ-Marlin MoE, and Moe WNA16 fallback handling today; reject "
        "it until native GB10 GPTQ correctness evidence exists."
    ),
    "inc_quantization": (
        "INC/AutoRound quantization can select AWQ or GPTQ Marlin dense "
        "kernels, AWQ/GPTQ MoE, and Moe WNA16 fallback handling today; reject "
        "it until native GB10 INC/AutoRound correctness evidence exists."
    ),
    "gguf_quantization": (
        "GGUF quantization can select GGUF dense, embedding, and MoE kernels, "
        "including GGML matmul and dequantization fallbacks today; reject it "
        "until native GB10 GGUF correctness evidence exists."
    ),
    "humming_quantization": (
        "Humming quantization can select Humming dense and MoE kernels today; "
        "reject it until native GB10 Humming correctness evidence exists."
    ),
    "humming_mxfp4_moe_backend": (
        "The Humming MXFP4 MoE backend can select Humming Mixed Precision "
        "kernels today; reject it until native GB10 Humming MXFP4 MoE "
        "correctness evidence exists."
    ),
    "fp8_w8a16_marlin_fallback": (
        "FP8 W8A16 Marlin fallback can prove reachability, but it is not "
        "native GB10 FP8 W8A16 dense evidence."
    ),
    "fp8_w8a16_moe_fallback": (
        "FP8 MoE Marlin and CPU W8A16 fallbacks can prove reachability, but "
        "they are not native GB10 FP8 MoE evidence."
    ),
    "int8_moe_triton_fallback": (
        "Int8 MoE Triton fallback can prove reachability, but it is not "
        "native GB10 Int8 MoE evidence."
    ),
    "wna16_moe_fallback": (
        "WNA16 MoE Marlin and batched Marlin fallbacks can prove "
        "reachability, but they are not native GB10 WNA16/MXINT MoE evidence."
    ),
    "compressed_tensors_wna16_dense_loading": (
        "CompressedTensors WNA16 dense loading can select generic "
        "mixed-precision WNA16/W4A16 kernels today; reject it until native "
        "GB10 WNA16/MXINT dense correctness evidence exists."
    ),
    "compressed_tensors_wna16_moe_fallback": (
        "CompressedTensors WNA16 MoE legacy fused-experts fallback can prove "
        "reachability, but it is not native GB10 WNA16/MXINT MoE evidence."
    ),
    "moe_wna16_legacy_fallback": (
        "MoeWNA16 legacy fused-experts fallback can prove reachability, but "
        "it is not native GB10 WNA16/MXINT MoE evidence."
    ),
    "mxfp8_dense_fallback": (
        "MXFP8 dense Marlin and emulation fallbacks can prove reachability, "
        "but they are not native GB10 MXFP8 dense evidence."
    ),
    "mxfp8_moe_fallback": (
        "MXFP8 MoE Marlin fallback can prove reachability, but it is not "
        "native GB10 MXFP8 MoE evidence."
    ),
    "modelopt_fp8_quantization": (
        "ModelOpt FP8 quantization can reach FP8 dense kernel selection and "
        "FP8 MoE backend selection today; reject it until native GB10 ModelOpt "
        "FP8 correctness evidence exists."
    ),
    "modelopt_mxfp8_quantization": (
        "ModelOpt MXFP8 quantization can reach MXFP8 dense kernel selection "
        "and MXFP8 MoE backend selection today; reject it until native GB10 "
        "ModelOpt MXFP8 correctness evidence exists."
    ),
    "modelopt_mixed_quantization": (
        "ModelOpt mixed precision quantization can reach FP8 dense or MoE "
        "selection, NVFP4 dense or MoE selection, and W4A16 NVFP4 fallback "
        "selection today; reject it until native GB10 ModelOpt mixed precision "
        "correctness evidence exists."
    ),
    "fbgemm_fp8_quantization": (
        "FBGEMM FP8 quantization is a deprecated public quantization method "
        "that can reach generic FP8 linear kernel selection today; reject it "
        "until native GB10 FBGEMM FP8 correctness evidence exists."
    ),
    "experts_int8_quantization": (
        "ExpertsInt8 quantization is a backward-compatible public "
        "quantization method that can reach online Int8 MoE backend selection "
        "today; reject it until native GB10 online Int8 MoE correctness "
        "evidence exists."
    ),
    "fp_quant_fp4_quantization": (
        "FPQuant FP4 quantization is a deprecated public quantization method "
        "that can reach MXFP4/NVFP4 FPQuant linear kernels today; reject it "
        "until native GB10 FPQuant FP4 correctness evidence exists."
    ),
    "online_fp8_quantization": (
        "Online FP8 quantization can reach FP8 scaled-mm dense kernels and "
        "generic FP8 MoE backend selection today; reject it until native GB10 "
        "online FP8 dense and MoE correctness evidence exists."
    ),
    "online_mxfp8_quantization": (
        "Online MXFP8 quantization can reach FlashInfer CUTLASS MXFP8 dense "
        "and generic MXFP8 MoE backend selection today; reject it until "
        "native GB10 online MXFP8 dense and MoE correctness evidence exists."
    ),
    "online_mxfp4_quantization": (
        "Online MXFP4 quantization can accept weight='mxfp4' for dense or "
        "MoE online quantization, but no online MXFP4 method is wired today; "
        "reject it until native GB10 online MXFP4 correctness evidence exists."
    ),
    "online_int8_moe_quantization": (
        "Online Int8 MoE quantization can reach online Int8 MoE backend "
        "selection through int8_per_channel_weight_only today; reject it until "
        "native GB10 online Int8 MoE correctness evidence exists."
    ),
    "compressed_tensors_w8a8_mxfp8_dense_loading": (
        "CompressedTensors W8A8 MXFP8 dense loading can reach MXFP8 dense "
        "kernel selection today; reject it until native GB10 W8A8 MXFP8 dense "
        "correctness evidence exists."
    ),
    "compressed_tensors_w8a8_mxfp8_moe_loading": (
        "CompressedTensors W8A8 MXFP8 MoE loading can reach generic MXFP8 "
        "MoE backend selection today; reject it until native GB10 W8A8 "
        "MXFP8 MoE correctness evidence exists."
    ),
    "quark_nvfp4_checkpoint_loading": (
        "Quark NVFP4 checkpoint loading is not validated on GB10/SM12x; reject "
        "it until dense and MoE correctness evidence exists for native GB10 "
        "backends."
    ),
    "quark_ocp_mx_checkpoint_loading": (
        "Quark OCP-MX/MXFP4 checkpoint loading is not validated on GB10/SM12x; "
        "reject it until dense and MoE correctness evidence exists for native "
        "GB10 MXFP4 backends."
    ),
    "quark_w4a8_mxfp4_fp8_checkpoint_loading": (
        "Quark W4A8 MXFP4+FP8 checkpoint loading is not validated on "
        "GB10/SM12x; reject it until dense correctness evidence exists for a "
        "native GB10 MXFP4 backend."
    ),
    "quark_w4a8_fp8_moe_loading": (
        "Quark W4A8 FP8 MoE checkpoint loading requires ROCm AITER fused MoE "
        "support today; reject it until native GB10 W4A8 FP8 MoE correctness "
        "evidence exists."
    ),
    "quark_w8a8_fp8_checkpoint_loading": (
        "Quark W8A8 FP8 checkpoint loading can select FP8 scaled-mm dense "
        "kernel selection today; reject it until native GB10 W8A8 FP8 dense "
        "correctness evidence exists."
    ),
    "quark_w8a8_int8_checkpoint_loading": (
        "Quark W8A8 Int8 checkpoint loading can select Int8 scaled-mm dense "
        "kernel selection today; reject it until native GB10 W8A8 Int8 dense "
        "correctness evidence exists."
    ),
    "quark_w8a8_fp8_moe_loading": (
        "Quark W8A8 FP8 MoE checkpoint loading can select generic FP8 W8A8 "
        "MoE backend selection today; reject it until native GB10 W8A8 FP8 "
        "MoE correctness evidence exists."
    ),
    "quark_w8a8_int8_moe_loading": (
        "Quark W8A8 Int8 MoE checkpoint loading can select generic Int8 W8A8 "
        "MoE backend selection today; reject it until native GB10 W8A8 Int8 "
        "MoE correctness evidence exists."
    ),
    "compressed_tensors_w4a8_fp8_loading": (
        "CompressedTensors W4A8 FP8 loading uses exact-SM90 CUTLASS W4A8 "
        "kernels today and is not native GB10 evidence."
    ),
    "compressed_tensors_w4a8_int_dense_loading": (
        "CompressedTensors W4A8 Int dense loading can select generic "
        "mixed-precision W4A8/W4A16 kernels today; reject it until native "
        "GB10 W4A8 Int dense correctness evidence exists."
    ),
    "compressed_tensors_w4a8_int_moe_loading": (
        "CompressedTensors W4A8 Int8 MoE loading can select CPU-only W4A8 "
        "Int8 MoE backend selection today; reject it until native GB10 W4A8 "
        "Int8 MoE correctness evidence exists."
    ),
    "compressed_tensors_w8a16_fp8_loading": (
        "CompressedTensors W8A16 FP8 loading selects the FP8 W8A16 Marlin "
        "fallback today; reject it until native GB10 FP8 W8A16 dense "
        "correctness evidence exists."
    ),
    "compressed_tensors_w8a8_fp8_loading": (
        "CompressedTensors W8A8 FP8 loading can select scaled-mm W8A8 FP8 "
        "kernels today; reject it until native GB10 W8A8 FP8 dense "
        "correctness evidence exists."
    ),
    "compressed_tensors_w8a8_fp8_moe_loading": (
        "CompressedTensors W8A8 FP8 MoE loading can select generic FP8 W8A8 "
        "MoE backends today; reject it until native GB10 W8A8 FP8 MoE "
        "correctness evidence exists."
    ),
    "compressed_tensors_w8a8_int_dense_loading": (
        "CompressedTensors W8A8 Int dense loading can select Cutlass/Triton "
        "W8A8 Int8 scaled-mm kernels today; reject it until native GB10 W8A8 "
        "Int8 dense correctness evidence exists."
    ),
    "compressed_tensors_w8a8_int_moe_loading": (
        "CompressedTensors W8A8 Int8 MoE loading can select generic Int8 W8A8 "
        "MoE backends today; reject it until native GB10 W8A8 Int8 MoE "
        "correctness evidence exists."
    ),
    "compressed_tensors_w4a4_mxfp4_dense_loading": (
        "CompressedTensors W4A4 MXFP4 dense loading is not validated on "
        "GB10/SM12x; reject it until native GB10 MXFP4 dense correctness "
        "evidence exists."
    ),
    "compressed_tensors_w4a4_mxfp4_moe_loading": (
        "CompressedTensors W4A4 MXFP4 MoE loading can select CUTLASS on "
        "supported devices but falls back to Marlin when CUTLASS does not "
        "advertise support. Reject it on GB10/SM12x until native SM12x "
        "MXFP4 MoE checkpoint-loading correctness evidence exists."
    ),
    "compressed_tensors_qutlass_nvfp4_transform_loading": (
        "CompressedTensors Qutlass NVFP4 transform loading can select "
        "QutlassNvFP4LinearMethod, whose apply path is not implemented today; "
        "reject it until native GB10 transformed NVFP4 correctness evidence "
        "exists."
    ),
    "compressed_tensors_w4a16_nvfp4_loading": (
        "CompressedTensors W4A16 NVFP4 loading selects FP4 Marlin today; reject "
        "it on GB10/SM12x until a native dense backend or routed support path "
        "is validated."
    ),
    "compressed_tensors_w4a16_nvfp4_moe_loading": (
        "CompressedTensors W4A16 NVFP4 MoE loading can reach weight-only NVFP4 "
        "MoE handling today; reject it until native GB10 W4A16 NVFP4 MoE "
        "correctness evidence exists."
    ),
}

GB10_DEFERRED_PATH_REASONS = {
    "deepseek_v4_deep_gemm_mega_moe": (
        "DeepSeek V4 DeepGEMM MegaMoE is an explicit expert-parallel backend "
        "with SM120-family runtime allowance, but it is deferred until GB10 "
        "artifact, correctness, and runtime evidence exist."
    ),
    "flashinfer_b12x_ep_all2all_eplb": (
        "FlashInfer b12x expert-parallel all-to-all/EPLB NVFP4 MoE is "
        "deferred until multi-Spark communication contracts are validated on "
        "GB10 hardware."
    ),
    "flashinfer_cudnn_nvfp4_dense": (
        "FlashInfer cuDNN dense NVFP4 is deferred on GB10/SM12x until "
        "correctness, artifact, and runtime evidence exist."
    ),
    "multi_spark_ep_all2all_eplb": (
        "Multi-Spark expert-parallel all-to-all/EPLB serving is deferred until "
        "there is hardware to validate NCCL/Ray/vLLM communication and load "
        "balancing behavior."
    ),
}

GB10_SUPPORT_STATUSES = frozenset(
    {
        "supported_native",
        "supported_routed",
        "not_supported",
        "deferred",
    }
)

RELEASE_NVFP4_SMOKE_REPORT_FILE = "gb10-nvfp4-smoke.json"
RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE = "gb10-openai-server-smoke-image.json"
RELEASE_EVIDENCE_SUMMARY_REPORT_FILE = "gb10-release-evidence-image.json"
RELEASE_SMOKED_IMAGE_DIGEST_FILE = "gb10-smoked-image-digest.txt"

EXPECTED_RELEASE_REPORTS = (
    RELEASE_NVFP4_SMOKE_REPORT_FILE,
    RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE,
    RELEASE_EVIDENCE_SUMMARY_REPORT_FILE,
)

EXPECTED_RELEASE_EVIDENCE_FILES = (
    *EXPECTED_RELEASE_REPORTS,
    RELEASE_SMOKED_IMAGE_DIGEST_FILE,
)

PROVENANCE_FILES = {
    "release_manifest": "gb10-release-manifest.json",
    "runtime_image_metadata": "buildx-runtime-image-metadata.json",
}

VLLM_RELEASE_ASSET_FILES = {
    "runtime_image_ref": "gb10-runtime-image-ref.txt",
    "runtime_image_digest": "gb10-runtime-image-digest.txt",
    "checksums": "gb10-vllm-release-SHA256SUMS",
}

DEFAULT_RELEASE_MANIFEST_DIR = Path("gb10-release-manifest")
DEFAULT_VLLM_RELEASE_DIST_DIR = Path("dist")
DEFAULT_RELEASE_EVIDENCE_REPORT_DIR = Path("gb10-smoke-reports")
DEFAULT_RELEASE_EVIDENCE_OUTPUT_DIR = Path("dist/gb10-release-evidence")
DEFAULT_RELEASE_PROVENANCE_DIR = Path("gb10-release-provenance")
GITHUB_RELEASE_MANIFEST_ARTIFACT_NAME = "gb10-release-manifest"
GITHUB_RELEASE_INPUTS_ARTIFACT_NAME = "gb10-release-inputs"
GITHUB_RELEASE_EVIDENCE_ARTIFACT_NAME = "gb10-release-evidence"
RELEASE_EVIDENCE_BUNDLE_NAME = "gb10-release-evidence"
RELEASE_EVIDENCE_METADATA_FILE = "release-evidence-metadata.json"
RELEASE_EVIDENCE_CHECKSUM_FILE = "SHA256SUMS"
VLLM_RELEASE_WHEEL_GLOB = "vllm-*.whl"

PROVENANCE_RELATIVE_PATHS = {
    "release_manifest": "provenance/gb10-release-manifest.json",
    "runtime_image_metadata": "provenance/buildx-runtime-image-metadata.json",
}

REQUIRED_RELEASE_EVIDENCE_PROVENANCE = (
    "release_manifest",
    "runtime_image_metadata",
)

RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS = (
    "release_manifest",
    "runtime_image_metadata",
    "runtime_image_ref",
    "runtime_image_digest",
)

SHA256_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


def default_release_manifest_dir() -> Path:
    return Path(
        os.environ.get("GB10_RELEASE_MANIFEST_DIR", DEFAULT_RELEASE_MANIFEST_DIR)
    )


def default_release_manifest_json() -> Path:
    explicit = os.environ.get("GB10_RELEASE_MANIFEST_JSON")
    if explicit:
        return Path(explicit)
    return default_release_manifest_dir() / PROVENANCE_FILES["release_manifest"]


def default_runtime_image_metadata_json() -> Path:
    explicit = os.environ.get("GB10_RUNTIME_IMAGE_METADATA_JSON")
    if explicit:
        return Path(explicit)
    return default_release_manifest_dir() / PROVENANCE_FILES["runtime_image_metadata"]


def default_release_evidence_manifest_json() -> Path | None:
    explicit = os.environ.get("GB10_RELEASE_EVIDENCE_MANIFEST_JSON") or os.environ.get(
        "GB10_RELEASE_MANIFEST_JSON"
    )
    if explicit:
        return Path(explicit)
    manifest_dir = os.environ.get("GB10_RELEASE_MANIFEST_DIR")
    if manifest_dir:
        return Path(manifest_dir) / PROVENANCE_FILES["release_manifest"]
    return None


def default_release_evidence_runtime_image_metadata_json() -> Path | None:
    explicit = os.environ.get(
        "GB10_RELEASE_EVIDENCE_RUNTIME_IMAGE_METADATA_JSON"
    ) or os.environ.get("GB10_RUNTIME_IMAGE_METADATA_JSON")
    if explicit:
        return Path(explicit)
    return None


def default_release_evidence_report_dir() -> Path:
    return Path(
        os.environ.get(
            "GB10_RELEASE_EVIDENCE_REPORT_DIR",
            DEFAULT_RELEASE_EVIDENCE_REPORT_DIR,
        )
    )


def default_release_evidence_output_dir() -> Path:
    return Path(
        os.environ.get(
            "GB10_RELEASE_EVIDENCE_OUTPUT_DIR",
            DEFAULT_RELEASE_EVIDENCE_OUTPUT_DIR,
        )
    )


def default_release_provenance_dir() -> Path:
    return Path(
        os.environ.get(
            "GB10_RELEASE_PROVENANCE_DIR",
            DEFAULT_RELEASE_PROVENANCE_DIR,
        )
    )


def default_release_evidence_bundle_name() -> str:
    return os.environ.get(
        "GB10_RELEASE_EVIDENCE_BUNDLE_NAME",
        RELEASE_EVIDENCE_BUNDLE_NAME,
    )


def release_evidence_bundle_archive_name(
    bundle_name: str = RELEASE_EVIDENCE_BUNDLE_NAME,
) -> str:
    return f"{bundle_name}.tar.gz"


def release_evidence_bundle_archive_checksum_name(
    bundle_name: str = RELEASE_EVIDENCE_BUNDLE_NAME,
) -> str:
    return f"{release_evidence_bundle_archive_name(bundle_name)}.sha256"


def release_evidence_asset_paths(
    output_dir: Path,
    bundle_name: str = RELEASE_EVIDENCE_BUNDLE_NAME,
) -> list[Path]:
    return [
        output_dir / release_evidence_bundle_archive_name(bundle_name),
        output_dir / release_evidence_bundle_archive_checksum_name(bundle_name),
        output_dir / RELEASE_EVIDENCE_METADATA_FILE,
        output_dir / RELEASE_EVIDENCE_CHECKSUM_FILE,
    ]


def release_evidence_file_paths(report_dir: Path) -> list[Path]:
    return [
        report_dir / filename
        for filename in EXPECTED_RELEASE_EVIDENCE_FILES
    ]


def release_provenance_artifact_paths(provenance_dir: Path) -> dict[str, Path]:
    paths = {
        "release_manifest": provenance_dir / PROVENANCE_FILES["release_manifest"],
        "runtime_image_metadata": provenance_dir
        / PROVENANCE_FILES["runtime_image_metadata"],
        "runtime_image_ref": provenance_dir
        / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
        "runtime_image_digest": provenance_dir
        / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"],
    }
    return {
        kind: paths[kind]
        for kind in RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS
    }


def find_vllm_wheel_assets(dist_dir: Path) -> list[Path]:
    return sorted(dist_dir.glob(VLLM_RELEASE_WHEEL_GLOB))


def vllm_release_asset_paths(
    *,
    dist_dir: Path,
    release_manifest_dir: Path,
    runtime_image_metadata_json: Path,
) -> list[Path]:
    return [
        *find_vllm_wheel_assets(dist_dir),
        release_manifest_dir / PROVENANCE_FILES["release_manifest"],
        runtime_image_metadata_json,
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"],
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["checksums"],
    ]


def vllm_release_checksum_asset_paths(
    *,
    dist_dir: Path,
    release_manifest_dir: Path,
    runtime_image_metadata_json: Path,
) -> list[Path]:
    checksum_assets = [
        *find_vllm_wheel_assets(dist_dir),
        release_manifest_dir / PROVENANCE_FILES["release_manifest"],
        runtime_image_metadata_json,
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
    ]

    runtime_image_digest = (
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"]
    )
    if runtime_image_digest.is_file():
        checksum_assets.append(runtime_image_digest)
    return checksum_assets
