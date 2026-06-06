#!/usr/bin/env python3
"""Write a GB10 release manifest from resolved GitHub Actions settings."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

from gb10_release_contract import (
    GB10_SUPPORT_STATUSES,
    REQUIRED_FLASHINFER_COMPONENTS,
    REQUIRED_GB10_SUPPORT_MATRIX,
    REQUIRED_SOURCE_DEPENDENCIES,
    default_release_manifest_json,
)

DOCKER_REPOSITORY_COMPONENT_RE = re.compile(r"[a-z0-9]+(?:[._-]+[a-z0-9]+)*")
DOCKER_TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


def _env(env: Mapping[str, str], name: str, default: str = "") -> str:
    return env.get(name, default)


def _env_bool(env: Mapping[str, str], name: str) -> bool:
    value = _env(env, name)
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Unsupported GB10 boolean setting {name}={value}. "
        "Use 1, 0, true, false, yes, no, on, or off."
    )


def _env_json_string_list(env: Mapping[str, str], name: str) -> list[str]:
    try:
        value = json.loads(_env(env, name, "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _is_full_git_sha(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _is_gb10_vllm_version(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return (
        re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)*"
            r"(?:(?:a|b|rc)[0-9]+)?"
            r"(?:\.post[0-9]+)?"
            r"(?:\.dev[0-9]+)?"
            r"\+gb10\.[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*",
            value,
        )
        is not None
    )


def _run_url(env: Mapping[str, str]) -> str:
    server = _env(env, "GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    repository = _env(env, "GITHUB_REPOSITORY")
    run_id = _env(env, "GITHUB_RUN_ID")
    if not repository or not run_id:
        return ""

    url = f"{server}/{repository}/actions/runs/{run_id}"
    attempt = _env(env, "GITHUB_RUN_ATTEMPT")
    if attempt and attempt != "1":
        url = f"{url}/attempts/{attempt}"
    return url


def _github_release_tag(url: str) -> str | None:
    path = unquote(urlparse(url).path)
    marker = "/releases/download/"
    if marker not in path:
        return None
    tail = path.split(marker, 1)[1]
    return tail.split("/", 1)[0] or None


def _github_release_identity(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        return None

    path = unquote(parsed.path).lstrip("/")
    marker = "/releases/download/"
    if marker not in path:
        return None

    repository, tail = path.split(marker, 1)
    tag = tail.split("/", 1)[0]
    if not repository or not tag:
        return None
    return repository, tag


def _url_filename(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def _wheel_component(url: str) -> str:
    filename = _url_filename(url)
    for component in REQUIRED_FLASHINFER_COMPONENTS:
        if filename.startswith(f"{component}-"):
            return component
    return filename.split("-", 1)[0]


def _gb10_cuda_wheel_filename(filename: str) -> bool:
    parts = filename.split("-", 2)
    if len(parts) < 2:
        return False
    version = parts[1]
    return re.search(r"\+cu[0-9]+gb10", version) is not None


def _flashinfer_wheels(env: Mapping[str, str]) -> list[dict[str, str | None]]:
    urls = _env(env, "GB10_PREBUILT_WHEEL_URLS").split()
    wheels = []
    for url in urls:
        release_identity = _github_release_identity(url)
        wheels.append(
            {
                "component": _wheel_component(url),
                "filename": Path(unquote(urlparse(url).path)).name,
                "release_repository": release_identity[0]
                if release_identity is not None
                else None,
                "release_tag": _github_release_tag(url),
                "url": url,
            }
        )
    return wheels


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cmake_cache_default(path: Path, variable: str) -> str:
    content = path.read_text()
    pattern = rf"set\(\s*{re.escape(variable)}\s+\"([^\"]+)\""
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"{path}: could not find default for {variable}")
    return match.group(1)


def _pinned_source_dependency(
    *,
    name: str,
    cmake_path: Path,
    repository_variable: str,
    ref_variable: str,
    env: Mapping[str, str],
) -> dict[str, object]:
    repository = _env(
        env,
        repository_variable,
        _cmake_cache_default(cmake_path, repository_variable),
    )
    ref = _env(env, ref_variable, _cmake_cache_default(cmake_path, ref_variable))
    return {
        "name": name,
        "cmake_file": str(cmake_path.relative_to(_repo_root())),
        "repository": repository,
        "ref": ref,
        "ref_is_full_git_sha": _is_full_git_sha(ref),
    }


def _source_dependencies(env: Mapping[str, str]) -> dict[str, object]:
    root = _repo_root()
    return {
        "deepgemm": _pinned_source_dependency(
            name="DeepGEMM",
            cmake_path=root / "cmake" / "external_projects" / "deepgemm.cmake",
            repository_variable="DEEPGEMM_GIT_REPOSITORY",
            ref_variable="DEEPGEMM_GIT_TAG",
            env=env,
        ),
        "flashmla": _pinned_source_dependency(
            name="FlashMLA",
            cmake_path=root / "cmake" / "external_projects" / "flashmla.cmake",
            repository_variable="FLASH_MLA_GIT_REPOSITORY",
            ref_variable="FLASH_MLA_GIT_TAG",
            env=env,
        ),
        "triton_kernels": _pinned_source_dependency(
            name="triton_kernels",
            cmake_path=root
            / "cmake"
            / "external_projects"
            / "triton_kernels.cmake",
            repository_variable="TRITON_KERNELS_GIT_REPOSITORY",
            ref_variable="TRITON_KERNELS_GIT_TAG",
            env=env,
        ),
    }


def _gb10_support_matrix() -> dict[str, object]:
    return {
        "architecture": "sm_121a",
        "hardware": "NVIDIA DGX Spark GB10",
        "first_release_scope": "single_spark_first_path",
        "status_definitions": {
            "supported_native": (
                "Runs native SM121A code and must have package/runtime "
                "evidence before satisfying release gates."
            ),
            "supported_routed": (
                "vLLM intentionally routes to another validated GB10-safe "
                "backend and reports that route."
            ),
            "not_supported": (
                "Must be rejected or reported as non-native before it can "
                "masquerade as GB10 support."
            ),
            "deferred": (
                "Not required for the first release and must not be selected "
                "accidentally."
            ),
        },
        "entries": {
            "flashinfer_b12x_nvfp4_dense": {
                "status": "supported_native",
                "release_contract": (
                    "First-path smoke may observe native FlashInfer b12x "
                    "dense NVFP4 backend selection on SM12x."
                ),
            },
            "flashinfer_cutlass_nvfp4_dense": {
                "status": "supported_native",
                "release_contract": (
                    "First-path smoke may observe native FlashInfer CUTLASS "
                    "dense NVFP4 backend selection on SM12x."
                ),
            },
            "flashinfer_nvfp4_quantization": {
                "status": "supported_native",
                "release_contract": (
                    "First-path smoke must observe GB10 FlashInfer runtime "
                    "packages used by native NVFP4 quantization paths."
                ),
            },
            "modelopt_fp4_quantization": {
                "status": "supported_native",
                "release_contract": (
                    "First-path smoke must observe ModelOpt FP4 model-load "
                    "quantization for NVFP4 checkpoints."
                ),
            },
            "compressed_tensors_w4a4_nvfp4_dense_loading": {
                "status": "supported_native",
                "release_contract": (
                    "CompressedTensors W4A4 NVFP4 dense checkpoints are "
                    "allowed on GB10 when runtime backend selection records "
                    "native FlashInfer b12x or FlashInfer CUTLASS dense "
                    "NVFP4 evidence."
                ),
            },
            "compressed_tensors_w4a4_nvfp4_moe_loading": {
                "status": "supported_native",
                "release_contract": (
                    "CompressedTensors W4A4 NVFP4 MoE checkpoints are "
                    "allowed on GB10 when runtime backend selection records "
                    "native FlashInfer b12x or FlashInfer CUTLASS non-EP "
                    "NVFP4 MoE evidence."
                ),
            },
            "compressed_tensors_qutlass_nvfp4_transform_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors Qutlass NVFP4 transform loading can "
                    "select QutlassNvFP4LinearMethod, whose apply path is not "
                    "implemented today, and must reject on GB10/SM12x until "
                    "native GB10 transformed NVFP4 correctness evidence "
                    "exists."
                ),
            },
            "flashinfer_attention_fa2": {
                "status": "supported_native",
                "release_contract": (
                    "First-path smoke must observe the requested FlashInfer "
                    "attention backend with FP8 KV cache."
                ),
            },
            "flashinfer_b12x_non_ep_moe": {
                "status": "supported_native",
                "release_contract": (
                    "Required when MoE release evidence is requested; EP and "
                    "all-to-all variants remain separate entries."
                ),
            },
            "flashinfer_cutlass_non_ep_moe": {
                "status": "supported_native",
                "release_contract": (
                    "Allowed native non-EP NVFP4 MoE path for GB10, including "
                    "SwiGLU-clamp models where b12x is not applicable."
                ),
            },
            "flashmla_attention": {
                "status": "supported_native",
                "release_contract": (
                    "FlashMLA source provenance is required and vLLM may route "
                    "to native SM121A FlashMLA dense kernels where selected."
                ),
            },
            "flashmla_sparse_attention": {
                "status": "supported_native",
                "release_contract": (
                    "FlashMLA source provenance is required and vLLM may route "
                    "to native SM121A FlashMLA sparse kernels where selected."
                ),
            },
            "flashinfer_mamba_ssu": {
                "status": "supported_native",
                "release_contract": (
                    "GB10 Mamba selective-state-update must use native "
                    "FlashInfer SM12x kernels when Mamba1 or Mamba2 SSU "
                    "models are selected; Triton fallback is a separate Not "
                    "Supported entry."
                ),
            },
            "flashinfer_gdn_prefill": {
                "status": "supported_native",
                "release_contract": (
                    "GB10 GDN prefill must use native FlashInfer SM12x "
                    "kernels when GDN models are selected; Triton/FLA and "
                    "CuteDSL fallbacks are separate Not Supported entries."
                ),
            },
            "gb10_attention_trtllm_gen_to_flashinfer_fa2": {
                "status": "supported_routed",
                "release_contract": (
                    "TRTLLM Gen attention is unavailable on SM121; GB10 "
                    "attention must route through validated FlashInfer FA2 "
                    "evidence until SM121 TRTLLM Gen artifacts exist."
                ),
            },
            "gb10_attention_public_flashattention_to_flashinfer_or_flashmla": {
                "status": "supported_routed",
                "release_contract": (
                    "Public FlashAttention runtime is Not Supported for the "
                    "first GB10 runtime path; GB10 attention must route "
                    "through validated FlashInfer or FlashMLA evidence."
                ),
            },
            "gb10_moe_trtllm_gen_to_flashinfer_non_ep": {
                "status": "supported_routed",
                "release_contract": (
                    "TRTLLM Gen MoE is unavailable on SM121; first-path "
                    "non-EP NVFP4 MoE must route through validated FlashInfer "
                    "b12x or FlashInfer CUTLASS evidence."
                ),
            },
            "public_flashattention_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "GB10 vLLM runtime selection must not use public "
                    "FlashAttention on SM12x unless a future validated entry "
                    "changes this status."
                ),
            },
            "flashinfer_trtllm_nvfp4_dense": {
                "status": "not_supported",
                "release_contract": (
                    "FlashInfer TRTLLM dense NVFP4 is not validated on "
                    "GB10/SM12x and must not satisfy native dense release "
                    "evidence."
                ),
            },
            "flashinfer_trtllm_mxfp4_moe": {
                "status": "not_supported",
                "release_contract": (
                    "FlashInfer TRTLLM MXFP4 MoE is an SM100-family path "
                    "today and is not validated on GB10/SM12x; keep it "
                    "unselected until native SM121A MXFP4 MoE evidence exists."
                ),
            },
            "flashinfer_cutedsl_nvfp4_moe": {
                "status": "not_supported",
                "release_contract": (
                    "Generic FlashInfer CuteDSL and batched CuteDSL NVFP4 MoE "
                    "variants are not validated native GB10 evidence; use "
                    "FlashInfer b12x or FlashInfer CUTLASS NVFP4 MoE instead."
                ),
            },
            "trtllm_gen_attention": {
                "status": "not_supported",
                "release_contract": (
                    "TRTLLM Gen attention rejects SM121 today; vLLM should "
                    "route GB10 attention through FlashInfer or FlashMLA."
                ),
            },
            "triton_attention_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Generic Triton attention fallback can prove reachability, "
                    "but it cannot satisfy native GB10 attention release "
                    "evidence until native SM12x Triton attention correctness "
                    "and runtime evidence exist."
                ),
            },
            "flex_attention_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "PyTorch FlexAttention fallback can prove reachability, "
                    "but it cannot satisfy native GB10 attention release "
                    "evidence until native SM12x FlexAttention correctness "
                    "and runtime evidence exist."
                ),
            },
            "turboquant_attention": {
                "status": "not_supported",
                "release_contract": (
                    "TurboQuant KV-cache compression can prove reachability, "
                    "but it cannot satisfy native GB10 attention or KV-cache "
                    "release evidence until native SM12x TurboQuant "
                    "correctness and runtime evidence exist."
                ),
            },
            "triton_mla_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Triton MLA can prove fallback reachability, but it "
                    "cannot satisfy native GB10 MLA release evidence. GB10 "
                    "MLA must use native FlashMLA dense or sparse kernels."
                ),
            },
            "flashinfer_trtllm_mla_attention": {
                "status": "not_supported",
                "release_contract": (
                    "FlashInfer TRT-LLM MLA is an SM100-family MLA path today "
                    "and must reject before dispatch on GB10/SM12x until "
                    "native SM121A correctness, artifact, and runtime "
                    "evidence exists."
                ),
            },
            "flashinfer_trtllm_sparse_mla_attention": {
                "status": "not_supported",
                "release_contract": (
                    "FlashInfer TRT-LLM Sparse MLA is an SM100-family sparse "
                    "MLA path today and must reject before dispatch on "
                    "GB10/SM12x until native SM121A correctness, artifact, "
                    "and runtime evidence exists."
                ),
            },
            "public_flashattention_mla_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Public FlashAttention MLA is not validated on GB10/SM12x "
                    "and must reject before dispatch until native SM121A MLA "
                    "correctness, artifact, and runtime evidence exists."
                ),
            },
            "cutlass_mla_sm100_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "SM100 CUTLASS MLA can prove Blackwell-family "
                    "reachability, but it cannot satisfy native GB10 MLA "
                    "release evidence until native SM121A correctness, "
                    "artifact, and runtime evidence exists."
                ),
            },
            "tokenspeed_mla_cutedsl_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "TokenSpeed CuTe DSL MLA can prove SM100-family "
                    "reachability, but it cannot satisfy native GB10 MLA "
                    "release evidence until native SM121A correctness, "
                    "artifact, and runtime evidence exists."
                ),
            },
            "triton_mamba_ssu_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Triton Mamba selective-state-update can prove fallback "
                    "reachability, but it cannot satisfy native GB10 Mamba "
                    "SSU release evidence. GB10 Mamba1 and Mamba2 SSU must "
                    "use native FlashInfer SM12x kernels."
                ),
            },
            "mamba1_triton_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Mamba1 runtime uses generic Triton causal-conv and "
                    "prefill/scan kernels beyond native FlashInfer SSU, so "
                    "it must reject on GB10/SM12x until full native Mamba1 "
                    "correctness, artifact, and runtime evidence exists."
                ),
            },
            "mamba2_triton_ssd_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Mamba2 runtime uses generic Triton causal-conv and SSD "
                    "prefill kernels beyond native FlashInfer SSU, so it must "
                    "reject on GB10/SM12x until full native Mamba2 "
                    "correctness, artifact, and runtime evidence exists."
                ),
            },
            "short_conv_triton_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "ShortConv runtime uses generic Triton causal-conv "
                    "kernels and must reject on GB10/SM12x until native "
                    "ShortConv correctness, artifact, and runtime evidence "
                    "exists."
                ),
            },
            "linear_attention_triton_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Linear attention runtime uses generic Triton "
                    "lightning/decode kernels and must reject on GB10/SM12x "
                    "until native linear-attention correctness, artifact, "
                    "and runtime evidence exists."
                ),
            },
            "cascade_attention_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Cascade attention opt-in changes attention execution "
                    "heuristics and uses split attention paths outside the "
                    "validated GB10 first release serving path. "
                    "model_config.disable_cascade_attn=False must reject "
                    "until native SM12x cascade attention correctness and "
                    "runtime evidence exists."
                ),
            },
            "speculative_decoding_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Speculative decoding changes scheduler, attention "
                    "metadata, sampler/rejection, and drafter runtime paths "
                    "across direct draft runner selection, MTP, EAGLE, "
                    "draft-model, and ngram methods; reject it on GB10/SM12x "
                    "until native SM12x correctness and runtime evidence "
                    "exists."
                ),
            },
            "pooling_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Pooling runtime changes model runner outputs, pooling "
                    "heads, embedding, classification, reward, and scoring "
                    "APIs outside the validated GB10 first release serving "
                    "path. It must reject on GB10/SM12x until native SM12x "
                    "pooling correctness and runtime evidence exists."
                ),
            },
            "reasoning_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Reasoning runtime changes token parsing and output "
                    "extraction outside the validated GB10 first release "
                    "serving path. It must reject on GB10/SM12x until native "
                    "SM12x reasoning correctness and runtime evidence exists."
                ),
            },
            "structured_outputs_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Structured outputs runtime compiles request-level "
                    "grammars and applies Triton grammar bitmasks to logits "
                    "outside the validated GB10 first release serving path. "
                    "It must reject on GB10/SM12x until native SM12x "
                    "structured-output correctness and runtime evidence "
                    "exists."
                ),
            },
            "openai_tool_calling_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "OpenAI tool-calling runtime uses frontend tool parsers, "
                    "request-level tools/tool_choice handling, tool schema "
                    "structured-output injection, built-in/MCP tool sessions, "
                    "and output parsing outside the validated GB10 first "
                    "release serving path. It must reject on GB10/SM12x "
                    "until native SM12x tool-calling correctness and runtime "
                    "evidence exists."
                ),
            },
            "lora_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "LoRA runtime uses CUDA Punica and Triton adapter "
                    "kernels across dense, embedding, logits, and MoE "
                    "adapter paths; reject it on GB10/SM12x until native "
                    "SM12x LoRA correctness and runtime evidence exists."
                ),
            },
            "gdn_prefill_triton_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "GDN prefill Triton/FLA fallback can prove reachability, "
                    "but it cannot satisfy native GB10 GDN prefill release "
                    "evidence; use native FlashInfer SM12x GDN prefill."
                ),
            },
            "gdn_prefill_cutedsl_backend": {
                "status": "not_supported",
                "release_contract": (
                    "GDN prefill CuteDSL is SM100-family evidence today and "
                    "must not satisfy GB10 GDN prefill release evidence; use "
                    "native FlashInfer SM12x GDN prefill."
                ),
            },
            "mm_encoder_fp8_attention": {
                "status": "not_supported",
                "release_contract": (
                    "MM encoder FP8 attention uses FlashInfer cuDNN FP8 ViT "
                    "attention today, but it cannot satisfy native GB10 MM "
                    "encoder attention release evidence until native SM12x "
                    "correctness, artifact, and runtime evidence exist."
                ),
            },
            "mm_encoder_public_flashattention_backend": {
                "status": "not_supported",
                "release_contract": (
                    "Public FlashAttention MM encoder attention is not "
                    "validated on GB10/SM12x and must reject before dispatch "
                    "until native SM12x correctness, artifact, and runtime "
                    "evidence exists."
                ),
            },
            "mm_encoder_triton_attention_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Triton MM encoder attention fallback can prove "
                    "reachability, but it cannot satisfy native GB10 MM "
                    "encoder attention release evidence until native SM12x "
                    "correctness, artifact, and runtime evidence exist."
                ),
            },
            "mm_encoder_torch_sdpa_attention_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Torch SDPA MM encoder attention fallback can prove "
                    "reachability, but it cannot satisfy native GB10 MM "
                    "encoder attention release evidence until native SM12x "
                    "correctness, artifact, and runtime evidence exist."
                ),
            },
            "trtllm_gen_moe": {
                "status": "not_supported",
                "release_contract": (
                    "TRTLLM Gen MoE rejects SM121 today and must not satisfy "
                    "GB10 MoE release evidence unless native SM121A TRTLLM "
                    "fused-MoE artifacts and runtime evidence exist."
                ),
            },
            "rocm_aiter_fp8_moe": {
                "status": "not_supported",
                "release_contract": (
                    "AITER FP8 MoE is a ROCm-specific backend and cannot "
                    "satisfy native GB10 CUDA MoE release evidence."
                ),
            },
            "deep_gemm_fp8_moe": {
                "status": "not_supported",
                "release_contract": (
                    "DeepGEMM FP8 MoE is selectable through "
                    "moe_backend='deep_gemm' and DeepGEMM env flags, but it "
                    "cannot satisfy native GB10 release evidence until GB10 "
                    "DeepGEMM FP8 MoE artifacts, correctness, and runtime "
                    "evidence exist."
                ),
            },
            "triton_fp8_moe": {
                "status": "not_supported",
                "release_contract": (
                    "Generic Triton FP8 MoE is selectable through "
                    "moe_backend='triton' and auto-selection, but it cannot "
                    "satisfy native GB10 release evidence until GB10 Triton "
                    "FP8 MoE correctness and runtime evidence exist."
                ),
            },
            "vllm_cutlass_fp8_moe": {
                "status": "not_supported",
                "release_contract": (
                    "vLLM CUTLASS FP8 MoE is selectable through "
                    "moe_backend='cutlass' and allow_vllm_cutlass=True call "
                    "sites, but it cannot satisfy native GB10 release "
                    "evidence until GB10 vLLM CUTLASS FP8 MoE artifacts, "
                    "correctness, and runtime evidence exist."
                ),
            },
            "rocm_aiter_mxfp4_moe": {
                "status": "not_supported",
                "release_contract": (
                    "AITER MXFP4 MoE backends are ROCm-specific paths today "
                    "and must reject on GB10/SM12x until native GB10 CUDA "
                    "MXFP4 MoE correctness evidence exists."
                ),
            },
            "gpt_oss_triton_mxfp4_moe": {
                "status": "not_supported",
                "release_contract": (
                    "GPT-OSS Triton MXFP4 MoE can select triton_kernels OAI "
                    "MXFP4/SwiGLU kernels today and must reject on "
                    "GB10/SM12x until native GB10 GPT-OSS MXFP4 MoE "
                    "correctness evidence exists."
                ),
            },
            "rocm_aiter_unquantized_moe": {
                "status": "not_supported",
                "release_contract": (
                    "AITER unquantized MoE is a ROCm-specific backend and "
                    "cannot satisfy native GB10 CUDA MoE release evidence."
                ),
            },
            "unquantized_moe_triton_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Generic Triton unquantized MoE fallback can prove "
                    "reachability, but it cannot satisfy native GB10 "
                    "unquantized MoE release evidence."
                ),
            },
            "marlin_nvfp4_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Marlin-backed API smoke can prove serving reachability, "
                    "but it cannot satisfy native NVFP4 release evidence."
                ),
            },
            "fbgemm_nvfp4_dense": {
                "status": "not_supported",
                "release_contract": (
                    "FBGEMM NVFP4 dense can prove backend reachability when "
                    "fbgemm_gpu is installed, but it cannot satisfy native "
                    "GB10 NVFP4 release evidence until SM121A FBGEMM artifacts "
                    "and correctness evidence exist."
                ),
            },
            "modelopt_w4a16_nvfp4_checkpoint_loading": {
                "status": "not_supported",
                "release_contract": (
                    "ModelOpt W4A16 NVFP4 checkpoint loading is not "
                    "validated on GB10/SM12x and must reject before Marlin "
                    "dense fallback or MoE backend selection until native "
                    "GB10 W4A16 NVFP4 correctness evidence exists."
                ),
            },
            "modelopt_nvfp4_kv_cache_loading": {
                "status": "not_supported",
                "release_contract": (
                    "ModelOpt NVFP4 KV-cache loading can auto-select vLLM "
                    "kv_cache_dtype='nvfp4', but the current GB10 release "
                    "evidence only validates FP8 E4M3 KV cache. It must "
                    "reject on GB10/SM12x until native SM12x NVFP4 KV-cache "
                    "correctness evidence exists."
                ),
            },
            "nvfp4_kv_cache_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "User-selected vLLM kv_cache_dtype='nvfp4' is not "
                    "validated on GB10/SM12x. It must reject until native "
                    "SM12x NVFP4 KV-cache allocation, scale handling, "
                    "attention dispatch, and correctness evidence exists."
                ),
            },
            "unvalidated_kv_cache_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "User-selected GB10 KV-cache runtime dtypes outside FP8 "
                    "E4M3 and FlashMLA sparse fp8_ds_mla are not validated. "
                    "E5M2, Gaudi FP8, and per-token-head KV-cache formats "
                    "must reject until native SM12x correctness evidence "
                    "exists."
                ),
            },
            "kv_events_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "KV cache event publishing and replay expose KV block "
                    "lifecycle through external event publishers outside the "
                    "validated GB10 first release serving path. It must reject "
                    "until native SM12x KV event correctness and runtime "
                    "evidence exists."
                ),
            },
            "kv_offload_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Single-instance KV offload changes KV allocation, "
                    "slot-mapping, and transfer behavior outside the "
                    "validated GB10 first release path. It must reject until "
                    "native SM12x KV offload correctness evidence exists."
                ),
            },
            "kv_transfer_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Distributed KV transfer, disaggregated prefill/decode, "
                    "and external KV connector request paths are not "
                    "validated on GB10/SM12x. They must reject until native "
                    "SM12x KV transfer correctness evidence exists."
                ),
            },
            "ubatching_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Dual batch overlap and manual ubatching change scheduler "
                    "microbatching, cascade-attention handling, and DeepEP "
                    "all-to-all assumptions outside the validated GB10 first "
                    "release path. They must reject until native SM12x "
                    "ubatching correctness evidence exists."
                ),
            },
            "distributed_parallel_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Data parallel, tensor parallel, pipeline parallel, "
                    "context parallel, multi-node multiprocessing, and "
                    "external launcher process topologies are outside the "
                    "validated GB10 first release path. They must reject "
                    "until native SM12x distributed correctness evidence "
                    "exists."
                ),
            },
            "kv_sharing_fast_prefill_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "KV sharing fast prefill overrides attention metadata "
                    "and logits indexing for KV-sharing models and is outside "
                    "the validated GB10 first release path. It must reject "
                    "until native SM12x correctness evidence exists."
                ),
            },
            "ec_transfer_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Distributed EC cache transfer connectors are outside "
                    "the validated GB10 first release path and V2 "
                    "model-runner support. They must reject until native "
                    "SM12x EC transfer correctness evidence exists."
                ),
            },
            "weight_transfer_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "RL training weight transfer uses NCCL or IPC weight "
                    "update engines outside the validated GB10 first release "
                    "serving path. It must reject until native SM12x "
                    "weight-transfer correctness evidence exists."
                ),
            },
            "return_routed_experts_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Routed experts capture changes MoE scheduler and "
                    "model-runner bookkeeping outside the validated GB10 "
                    "first release serving path. It must reject until native "
                    "SM12x routed-expert capture correctness evidence exists."
                ),
            },
            "logprobs_logits_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "logprobs_mode raw_logits and processed_logits return "
                    "full logits through sampler and model-runner output "
                    "paths outside the validated GB10 first release serving "
                    "path. They must reject until native SM12x logits-return "
                    "correctness evidence exists."
                ),
            },
            "custom_logits_processors_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Custom logits processor hooks mutate sampler logits "
                    "outside the validated GB10 first release serving path. "
                    "They must reject until native SM12x custom logits "
                    "processor correctness evidence exists."
                ),
            },
            "io_processor_plugin_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "IO processor plugins load custom input/output processor "
                    "code at model startup outside the validated GB10 first "
                    "release serving path. They must reject until native SM12x "
                    "IO processor plugin correctness and runtime evidence "
                    "exists."
                ),
            },
            "hf_config_path_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Alternate HF config paths decouple the Hugging Face "
                    "configuration source from the model path before native "
                    "vLLM model/backend selection outside the validated GB10 "
                    "first release serving path. They must reject until native "
                    "SM12x HF config path correctness and runtime evidence "
                    "exists."
                ),
            },
            "hf_overrides_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "HF config overrides mutate Hugging Face model "
                    "configuration before native vLLM model/backend selection "
                    "outside the validated GB10 first release serving path. "
                    "They must reject until native SM12x HF config override "
                    "correctness and runtime evidence exists."
                ),
            },
            "specialized_tokenizer_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Specialized tokenizer modes can load Mistral, DeepSeek, "
                    "Grok, Kimi, Qwen-VL, TerraTorch, or custom tokenizer code "
                    "outside the validated GB10 first release serving path. "
                    "They must reject until native SM12x specialized tokenizer "
                    "correctness and runtime evidence exists."
                ),
            },
            "skip_tokenizer_init_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Skipping tokenizer initialization disables tokenizer and "
                    "detokenizer setup and switches serving to token-id-only "
                    "request/response semantics outside the validated GB10 "
                    "first release OpenAI-compatible serving path. "
                    "--skip-tokenizer-init must reject until native SM12x "
                    "tokenizerless serving correctness and runtime evidence "
                    "exists."
                ),
            },
            "transformers_model_impl_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Transformers model implementation runtime bypasses "
                    "native vLLM model implementations and can run generic "
                    "Hugging Face module code outside the validated GB10 "
                    "first release serving path. Explicit and auto-resolved "
                    "Transformers backend execution must reject until native "
                    "SM12x Transformers backend correctness evidence exists."
                ),
            },
            "trust_remote_code_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Trusted remote model code can replace or extend native "
                    "vLLM model, tokenizer, and configuration behavior "
                    "outside the validated GB10 first release serving path. "
                    "It must reject until native SM12x remote-code model "
                    "correctness and runtime evidence exists."
                ),
            },
            "custom_scheduler_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Custom scheduler classes replace the default vLLM "
                    "scheduler with user-provided scheduling code outside the "
                    "validated GB10 first release serving path. They must "
                    "reject until native SM12x custom scheduler correctness "
                    "and runtime evidence exists."
                ),
            },
            "custom_worker_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Custom worker classes and extensions replace or extend "
                    "the vLLM worker implementation with user-provided code "
                    "outside the validated GB10 first release serving path. "
                    "They must reject until native SM12x custom worker "
                    "correctness and runtime evidence exists."
                ),
            },
            "prompt_embeds_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Prompt embeds input handling changes request input "
                    "batching and embedding handling outside the validated "
                    "GB10 first release serving path. It must reject until "
                    "native SM12x prompt-embeds correctness evidence exists."
                ),
            },
            "stock_torch_compile_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Stock torch.compile selects the generic PyTorch compile "
                    "pipeline outside the validated GB10 first release "
                    "serving path. It must reject until native SM12x stock "
                    "torch.compile correctness evidence exists."
                ),
            },
            "mamba_align_cache_runtime": {
                "status": "not_supported",
                "release_contract": (
                    "Mamba align-cache mode changes Mamba state copy, "
                    "preprocessing, and scheduler-step cache alignment "
                    "outside the validated GB10 first release serving path. "
                    "It must reject until native SM12x Mamba align-cache "
                    "correctness evidence exists."
                ),
            },
            "marlin_mxfp4_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Marlin-backed MXFP4 dense/MoE can prove fallback "
                    "reachability, but it cannot satisfy native GB10 MXFP4 "
                    "release evidence."
                ),
            },
            "mxfp4_moe_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "MXFP4 MoE Marlin, batched Marlin, emulation, and CPU "
                    "fallback backends can prove reachability, but they "
                    "cannot satisfy native GB10 MXFP4 release evidence."
                ),
            },
            "public_mxfp4_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "Public MXFP4 quantization can reach unquantized linear/"
                    "attention handling and MXFP4 MoE backend selection today "
                    "and must reject on GB10/SM12x until native GB10 public "
                    "MXFP4 correctness evidence exists."
                ),
            },
            "public_fp8_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "Public FP8 quantization can reach online FP8 quantization, "
                    "FP8 scaled-mm dense kernel selection, and FP8 MoE backend "
                    "selection today and must reject on GB10/SM12x until native "
                    "GB10 public FP8 correctness evidence exists."
                ),
            },
            "deepseek_v4_fp8_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "DeepSeek V4 FP8 quantization can reach FP8 block-"
                    "quantized linear/attention layers and FP8, MXFP4, or "
                    "ModelOpt NVFP4 MoE dispatch today and must reject on "
                    "GB10/SM12x until native GB10 DeepSeek V4 correctness "
                    "evidence exists."
                ),
            },
            "torchao_fp8_activation_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "TorchAO FP8 activation quantization can call "
                    "torchao.quantization.quantize_ and hardware-specific "
                    "tensor packing today and must reject on GB10/SM12x until "
                    "native GB10 TorchAO FP8 activation correctness evidence "
                    "exists."
                ),
            },
            "torchao_weight_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "TorchAO weight quantization can call "
                    "torchao.quantization.quantize_ and hardware-specific "
                    "tensor packing today and must reject on GB10/SM12x until "
                    "native GB10 TorchAO weight correctness evidence exists."
                ),
            },
            "bitsandbytes_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "BitsAndBytes quantization can select bitsandbytes 4-bit "
                    "linear kernels, bitsandbytes 8-bit matmul kernels, and "
                    "BitsAndBytesMoE handling today and must reject on "
                    "GB10/SM12x until native GB10 BitsAndBytes correctness "
                    "evidence exists."
                ),
            },
            "awq_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "AWQ quantization can select AWQ dense kernels, "
                    "AWQ-Marlin dense kernels, AWQ-Marlin MoE, and Moe WNA16 "
                    "fallback handling today and must reject on GB10/SM12x "
                    "until native GB10 AWQ correctness evidence exists."
                ),
            },
            "gptq_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "GPTQ quantization can select GPTQ dense kernel routing, "
                    "AutoGPTQ-Marlin MoE, and Moe WNA16 fallback handling "
                    "today and must reject on GB10/SM12x until native GB10 "
                    "GPTQ correctness evidence exists."
                ),
            },
            "inc_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "INC/AutoRound quantization can select AWQ or GPTQ Marlin "
                    "dense kernels, AWQ/GPTQ MoE, and Moe WNA16 fallback "
                    "handling today and must reject on GB10/SM12x until "
                    "native GB10 INC/AutoRound correctness evidence exists."
                ),
            },
            "gguf_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "GGUF quantization can select GGUF dense, embedding, and "
                    "MoE kernels, including GGML matmul and dequantization "
                    "fallbacks today and must reject on GB10/SM12x until "
                    "native GB10 GGUF correctness evidence exists."
                ),
            },
            "humming_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "Humming quantization can select Humming dense and MoE "
                    "kernels today and must reject on GB10/SM12x until native "
                    "GB10 Humming correctness evidence exists."
                ),
            },
            "humming_mxfp4_moe_backend": {
                "status": "not_supported",
                "release_contract": (
                    "The Humming MXFP4 MoE backend can select Humming Mixed "
                    "Precision kernels today and must reject on GB10/SM12x "
                    "until native GB10 Humming MXFP4 MoE correctness evidence "
                    "exists."
                ),
            },
            "fp8_w8a16_marlin_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "FP8 W8A16 Marlin fallback can prove reachability, but it "
                    "cannot satisfy native GB10 FP8 W8A16 release evidence."
                ),
            },
            "fp8_w8a16_moe_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "FP8 MoE Marlin and CPU W8A16 fallbacks can prove "
                    "reachability, but they cannot satisfy native GB10 FP8 "
                    "MoE release evidence."
                ),
            },
            "int8_moe_triton_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Int8 MoE Triton fallback can prove reachability, but it "
                    "cannot satisfy native GB10 Int8 MoE release evidence."
                ),
            },
            "wna16_moe_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "WNA16 MoE Marlin and batched Marlin fallbacks can prove "
                    "reachability, but they cannot satisfy native GB10 "
                    "WNA16/MXINT MoE release evidence."
                ),
            },
            "compressed_tensors_wna16_dense_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors WNA16 dense loading can select generic "
                    "mixed-precision WNA16/W4A16 kernels today and must "
                    "reject on GB10/SM12x until native GB10 WNA16/MXINT "
                    "dense correctness evidence exists."
                ),
            },
            "compressed_tensors_wna16_moe_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors WNA16 MoE legacy fused-experts fallback "
                    "can prove reachability, but it cannot satisfy native "
                    "GB10 WNA16/MXINT MoE release evidence."
                ),
            },
            "moe_wna16_legacy_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "MoeWNA16 legacy fused-experts fallback can prove "
                    "reachability, but it cannot satisfy native GB10 "
                    "WNA16/MXINT MoE release evidence."
                ),
            },
            "mxfp8_dense_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "MXFP8 dense Marlin and emulation fallbacks can prove "
                    "reachability, but they cannot satisfy native GB10 "
                    "MXFP8 dense release evidence."
                ),
            },
            "mxfp8_moe_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "MXFP8 MoE Marlin fallback can prove reachability, but it "
                    "cannot satisfy native GB10 MXFP8 MoE release evidence."
                ),
            },
            "modelopt_fp8_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "ModelOpt FP8 quantization can reach FP8 dense kernel "
                    "selection and FP8 MoE backend selection today and must "
                    "reject on GB10/SM12x until native GB10 ModelOpt FP8 "
                    "correctness evidence exists."
                ),
            },
            "modelopt_mxfp8_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "ModelOpt MXFP8 quantization can reach MXFP8 dense kernel "
                    "selection and MXFP8 MoE backend selection today and must "
                    "reject on GB10/SM12x until native GB10 ModelOpt MXFP8 "
                    "correctness evidence exists."
                ),
            },
            "modelopt_mixed_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "ModelOpt mixed precision quantization can reach FP8 dense "
                    "or MoE selection, NVFP4 dense or MoE selection, and W4A16 "
                    "NVFP4 fallback selection today and must reject on "
                    "GB10/SM12x until native GB10 ModelOpt mixed precision "
                    "correctness evidence exists."
                ),
            },
            "fbgemm_fp8_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "FBGEMM FP8 quantization is a deprecated public "
                    "quantization method that can reach generic FP8 linear "
                    "kernel selection today and must reject on GB10/SM12x "
                    "until native GB10 FBGEMM FP8 correctness evidence exists."
                ),
            },
            "experts_int8_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "ExpertsInt8 quantization is a backward-compatible public "
                    "quantization method that can reach online Int8 MoE "
                    "backend selection today and must reject on GB10/SM12x "
                    "until native GB10 online Int8 MoE correctness evidence "
                    "exists."
                ),
            },
            "fp_quant_fp4_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "FPQuant FP4 quantization is a deprecated public "
                    "quantization method that can reach MXFP4/NVFP4 FPQuant "
                    "linear kernels today and must reject on GB10/SM12x until "
                    "native GB10 FPQuant FP4 correctness evidence exists."
                ),
            },
            "online_fp8_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "Online FP8 quantization can reach FP8 scaled-mm dense "
                    "kernels and generic FP8 MoE backend selection today and "
                    "must reject on GB10/SM12x until native GB10 online FP8 "
                    "dense and MoE correctness evidence exists."
                ),
            },
            "online_mxfp8_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "Online MXFP8 quantization can reach FlashInfer CUTLASS "
                    "MXFP8 dense and generic MXFP8 MoE backend selection today "
                    "and must reject on GB10/SM12x until native GB10 online "
                    "MXFP8 dense and MoE correctness evidence exists."
                ),
            },
            "online_mxfp4_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "Online MXFP4 quantization can accept weight='mxfp4' for "
                    "dense or MoE online quantization, but no online MXFP4 "
                    "method is wired today. It must reject on GB10/SM12x until "
                    "native GB10 online MXFP4 correctness evidence exists."
                ),
            },
            "online_int8_moe_quantization": {
                "status": "not_supported",
                "release_contract": (
                    "Online Int8 MoE quantization can reach online Int8 MoE "
                    "backend selection through int8_per_channel_weight_only "
                    "today and must reject on GB10/SM12x until native GB10 "
                    "online Int8 MoE correctness evidence exists."
                ),
            },
            "compressed_tensors_w8a8_mxfp8_dense_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W8A8 MXFP8 dense loading can reach "
                    "MXFP8 dense kernel selection today and must reject on "
                    "GB10/SM12x until native GB10 W8A8 MXFP8 dense correctness "
                    "evidence exists."
                ),
            },
            "compressed_tensors_w8a8_mxfp8_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W8A8 MXFP8 MoE loading can reach "
                    "generic MXFP8 MoE backend selection today and must reject "
                    "on GB10/SM12x until native GB10 W8A8 MXFP8 MoE "
                    "correctness evidence exists."
                ),
            },
            "quark_nvfp4_checkpoint_loading": {
                "status": "not_supported",
                "release_contract": (
                    "Quark NVFP4 checkpoint loading is not validated on "
                    "GB10/SM12x and must reject before backend selection "
                    "until dense and MoE correctness evidence exists."
                ),
            },
            "quark_ocp_mx_checkpoint_loading": {
                "status": "not_supported",
                "release_contract": (
                    "Quark OCP-MX/MXFP4 checkpoint loading is not validated "
                    "on GB10/SM12x and must reject before dense or MoE "
                    "backend selection until native GB10 MXFP4 correctness "
                    "evidence exists."
                ),
            },
            "quark_w4a8_mxfp4_fp8_checkpoint_loading": {
                "status": "not_supported",
                "release_contract": (
                    "Quark W4A8 MXFP4+FP8 checkpoint loading is not "
                    "validated on GB10/SM12x and must reject before dense "
                    "emulation or AITER-specific backend use until native "
                    "GB10 MXFP4 correctness evidence exists."
                ),
            },
            "quark_w4a8_fp8_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "Quark W4A8 FP8 MoE checkpoint loading requires ROCm "
                    "AITER fused MoE support today; reject it until native "
                    "GB10 W4A8 FP8 MoE correctness evidence exists."
                ),
            },
            "quark_w8a8_fp8_checkpoint_loading": {
                "status": "not_supported",
                "release_contract": (
                    "Quark W8A8 FP8 checkpoint loading can reach FP8 "
                    "scaled-mm dense kernel selection today and must reject "
                    "on GB10/SM12x until native GB10 W8A8 FP8 dense "
                    "correctness evidence exists."
                ),
            },
            "quark_w8a8_int8_checkpoint_loading": {
                "status": "not_supported",
                "release_contract": (
                    "Quark W8A8 Int8 checkpoint loading can reach Int8 "
                    "scaled-mm dense kernel selection today and must reject "
                    "on GB10/SM12x until native GB10 W8A8 Int8 dense "
                    "correctness evidence exists."
                ),
            },
            "quark_w8a8_fp8_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "Quark W8A8 FP8 MoE checkpoint loading can select generic "
                    "FP8 W8A8 MoE backend selection today; reject it until "
                    "native GB10 W8A8 FP8 MoE correctness evidence exists."
                ),
            },
            "quark_w8a8_int8_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "Quark W8A8 Int8 MoE checkpoint loading can select "
                    "generic Int8 W8A8 MoE backend selection today; reject it "
                    "until native GB10 W8A8 Int8 MoE correctness evidence "
                    "exists."
                ),
            },
            "compressed_tensors_w4a8_fp8_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W4A8 FP8 loading uses exact-SM90 "
                    "CUTLASS W4A8 kernels today and cannot satisfy native "
                    "GB10 dense or MoE release evidence."
                ),
            },
            "compressed_tensors_w4a8_int_dense_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W4A8 Int dense loading can select "
                    "generic mixed-precision W4A8/W4A16 kernels today and "
                    "must reject on GB10/SM12x until native GB10 W4A8 Int "
                    "dense correctness evidence exists."
                ),
            },
            "compressed_tensors_w4a8_int_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W4A8 Int8 MoE loading can select "
                    "CPU-only W4A8 Int8 MoE backend selection today and must "
                    "reject on GB10/SM12x until native GB10 W4A8 Int8 MoE "
                    "correctness evidence exists."
                ),
            },
            "compressed_tensors_w8a16_fp8_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W8A16 FP8 loading selects the FP8 "
                    "W8A16 Marlin fallback today and must reject on "
                    "GB10/SM12x until native GB10 FP8 W8A16 dense "
                    "correctness evidence exists."
                ),
            },
            "compressed_tensors_w8a8_fp8_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W8A8 FP8 loading can select "
                    "scaled-mm W8A8 FP8 kernels today and must reject on "
                    "GB10/SM12x until native GB10 W8A8 FP8 dense "
                    "correctness evidence exists."
                ),
            },
            "compressed_tensors_w8a8_fp8_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W8A8 FP8 MoE loading can select "
                    "generic FP8 W8A8 MoE backends today and must reject on "
                    "GB10/SM12x until native GB10 W8A8 FP8 MoE correctness "
                    "evidence exists."
                ),
            },
            "compressed_tensors_w8a8_int_dense_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W8A8 Int dense loading can select "
                    "Cutlass/Triton W8A8 Int8 scaled-mm kernels today and "
                    "must reject on GB10/SM12x until native GB10 W8A8 Int8 "
                    "dense correctness evidence exists."
                ),
            },
            "compressed_tensors_w8a8_int_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W8A8 Int8 MoE loading can select "
                    "generic Int8 W8A8 MoE backends today and must reject on "
                    "GB10/SM12x until native GB10 W8A8 Int8 MoE correctness "
                    "evidence exists."
                ),
            },
            "compressed_tensors_fp4_kv_cache_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors FP4 KV-cache loading is not supported "
                    "in vLLM and must reject before it can masquerade as the "
                    "validated GB10 FP8 KV-cache path."
                ),
            },
            "compressed_tensors_w4a4_mxfp4_dense_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W4A4 MXFP4 dense loading is not "
                    "validated on GB10/SM12x and must reject before dense "
                    "backend selection until native GB10 MXFP4 correctness "
                    "evidence exists."
                ),
            },
            "compressed_tensors_w4a4_mxfp4_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W4A4 MXFP4 MoE loading can select "
                    "CUTLASS on supported devices but falls back to Marlin "
                    "when CUTLASS does not advertise support, so it must "
                    "reject on GB10/SM12x until native SM12x MXFP4 MoE "
                    "checkpoint-loading correctness evidence exists."
                ),
            },
            "compressed_tensors_w4a16_nvfp4_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W4A16 NVFP4 loading selects FP4 "
                    "Marlin today and must reject on GB10/SM12x until native "
                    "dense or routed support is validated."
                ),
            },
            "compressed_tensors_w4a16_nvfp4_moe_loading": {
                "status": "not_supported",
                "release_contract": (
                    "CompressedTensors W4A16 NVFP4 MoE loading can reach "
                    "weight-only NVFP4 MoE handling today and must reject on "
                    "GB10/SM12x until native GB10 W4A16 NVFP4 MoE correctness "
                    "evidence exists."
                ),
            },
            "deepseek_v4_deep_gemm_mega_moe": {
                "status": "deferred",
                "release_contract": (
                    "DeepSeek V4 DeepGEMM MegaMoE is an explicit "
                    "expert-parallel backend with SM120-family runtime "
                    "allowance, but it is deferred until GB10 artifact, "
                    "correctness, and runtime evidence exist."
                ),
            },
            "flashinfer_b12x_ep_all2all_eplb": {
                "status": "deferred",
                "release_contract": (
                    "Blocked until multi-Spark EP/all-to-all/EPLB contracts "
                    "are validated on hardware."
                ),
            },
            "flashinfer_cudnn_nvfp4_dense": {
                "status": "deferred",
                "release_contract": (
                    "FlashInfer cuDNN dense NVFP4 is deferred on GB10/SM12x "
                    "until correctness, artifact, and runtime evidence exist."
                ),
            },
            "multi_spark_ep_all2all_eplb": {
                "status": "deferred",
                "release_contract": (
                    "Deferred until there is hardware to test multi-Spark "
                    "communication and load-balancing behavior."
                ),
            },
        },
    }


def build_manifest(env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Build the manifest payload from environment variables."""

    env = os.environ if env is None else env
    flashinfer_wheels = _flashinfer_wheels(env)
    flashinfer_components = {str(wheel["component"]) for wheel in flashinfer_wheels}
    missing_flashinfer = sorted(
        set(REQUIRED_FLASHINFER_COMPONENTS) - flashinfer_components
    )
    flash_attn_ref = _env(env, "GB10_FLASH_ATTN_REF")

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "github": {
            "workflow": _env(env, "GITHUB_WORKFLOW"),
            "repository": _env(env, "GITHUB_REPOSITORY"),
            "event_name": _env(env, "GITHUB_EVENT_NAME"),
            "ref": _env(env, "GITHUB_REF"),
            "run_id": _env(env, "GITHUB_RUN_ID"),
            "run_attempt": _env(env, "GITHUB_RUN_ATTEMPT"),
            "run_url": _run_url(env),
        },
        "git": {
            "commit": _env(env, "GITHUB_SHA"),
        },
        "release": {
            "tag": _env(env, "GB10_RELEASE_TAG"),
            "preflight_only": _env_bool(env, "GB10_PREFLIGHT_ONLY"),
        },
        "image": {
            "name": _env(env, "GB10_IMAGE_NAME"),
            "tag": _env(env, "GB10_IMAGE_TAG"),
            "push": _env_bool(env, "GB10_PUSH_IMAGE"),
        },
        "vllm": {
            "version": _env(env, "GB10_VLLM_VERSION"),
        },
        "dependencies": {
            "flashinfer": {
                "required_components": list(REQUIRED_FLASHINFER_COMPONENTS),
                "missing_components": missing_flashinfer,
                "all_required_components_present": not missing_flashinfer,
                "wheels": flashinfer_wheels,
            },
            "vllm_flash_attn": {
                "repository": _env(env, "GB10_FLASH_ATTN_REPO"),
                "ref": flash_attn_ref,
                "ref_is_full_git_sha": _is_full_git_sha(flash_attn_ref),
            },
            "source_dependencies": _source_dependencies(env),
            "required_source_dependencies": list(REQUIRED_SOURCE_DEPENDENCIES),
        },
        "gb10_support_matrix": _gb10_support_matrix(),
        "build": {
            "dockerfile": "docker/Dockerfile",
            "preflight_target": "gb10-flashinfer-preflight",
            "wheel_target": "build",
            "runtime_target": "vllm-openai",
            "local_gb10_dependency_checkouts": _env_bool(
                env,
                "VLLM_USE_LOCAL_GB10_DEPS",
            ),
            "parallelism": {
                "max_jobs": _env(env, "GB10_MAX_JOBS"),
                "nvcc_threads": _env(env, "GB10_NVCC_THREADS"),
            },
            "native_cuda_archs_only": _env_bool(
                env,
                "GB10_NATIVE_CUDA_ARCHS_ONLY",
            ),
            "runner_labels": _env_json_string_list(env, "GB10_RUNNER_LABELS"),
            "cache_refs": {
                "preflight": _env(env, "GB10_PREFLIGHT_CACHE_REF"),
                "wheel": _env(env, "GB10_WHEEL_CACHE_REF"),
                "runtime": _env(env, "GB10_RUNTIME_CACHE_REF"),
            },
        },
    }


def _mapping_value(value: object, key: str) -> object:
    if not isinstance(value, Mapping):
        return None
    return value.get(key)


def _github_release_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and _github_release_tag(value) is not None
    )


def _github_repository_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and bool(parsed.path.strip("/"))
    )


def _ghcr_image_repository_name(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("ghcr.io/"):
        return False
    repository = value.removeprefix("ghcr.io/")
    repository_parts = repository.split("/")
    return (
        len(repository_parts) >= 2
        and all(repository_parts)
        and ":" not in repository
        and "@" not in repository
        and not any(character.isspace() for character in value)
        and all(
            DOCKER_REPOSITORY_COMPONENT_RE.fullmatch(part) is not None
            for part in repository_parts
        )
    )


def _docker_image_tag(value: object) -> bool:
    return isinstance(value, str) and DOCKER_TAG_RE.fullmatch(value) is not None


def _ghcr_image_ref(value: object) -> bool:
    if not isinstance(value, str) or "@" in value:
        return False
    if ":" not in value:
        return False
    image_name, image_tag = value.rsplit(":", 1)
    return _ghcr_image_repository_name(image_name) and _docker_image_tag(image_tag)


def validate_manifest(manifest: Mapping[str, object]) -> list[str]:
    """Return release-input validation errors for a GB10 manifest."""

    errors: list[str] = []
    git_commit = _mapping_value(_mapping_value(manifest, "git"), "commit")
    if not isinstance(git_commit, str) or not _is_full_git_sha(git_commit):
        errors.append("vLLM release manifest git.commit must be a full Git SHA.")

    vllm_version = _mapping_value(_mapping_value(manifest, "vllm"), "version")
    if not _is_gb10_vllm_version(vllm_version):
        errors.append(
            "vLLM release manifest vllm.version must be a PEP 440 GB10 local "
            f"version, got {vllm_version!r}."
        )

    dependencies = _mapping_value(manifest, "dependencies")
    flashinfer = _mapping_value(dependencies, "flashinfer")
    wheels = _mapping_value(flashinfer, "wheels")
    if isinstance(wheels, list):
        component_counts: Counter[str] = Counter()
        unexpected_components: list[str] = []
        release_identities: set[tuple[str, str]] = set()

        for wheel in wheels:
            component = _mapping_value(wheel, "component")
            url = _mapping_value(wheel, "url")
            url_filename = _url_filename(url) if isinstance(url, str) else None
            url_component = _wheel_component(url) if isinstance(url, str) else None
            release_identity = (
                _github_release_identity(url) if isinstance(url, str) else None
            )
            if not _github_release_url(url):
                errors.append(
                    f"FlashInfer wheel must come from a GitHub Release: "
                    f"{component or url}."
                )

            if (
                isinstance(component, str)
                and url_component is not None
                and component != url_component
            ):
                errors.append(
                    "FlashInfer wheel component metadata does not match URL: "
                    f"component={component!r}, url_component={url_component!r}, "
                    f"url={url!r}."
                )

            if url_component in REQUIRED_FLASHINFER_COMPONENTS:
                component_counts[str(url_component)] += 1
                if release_identity is not None:
                    release_identities.add(release_identity)
                if url_filename is None or not _gb10_cuda_wheel_filename(
                    url_filename
                ):
                    errors.append(
                        "FlashInfer wheel filename must include a GB10 CUDA "
                        "local version: "
                        f"component={url_component!r}, filename={url_filename!r}."
                    )
            else:
                unexpected_components.append(str(url_component or component or url))

        missing_components = [
            component
            for component in REQUIRED_FLASHINFER_COMPONENTS
            if component_counts[component] == 0
        ]
        duplicate_components = {
            component: count
            for component, count in component_counts.items()
            if count > 1
        }
        if missing_components or duplicate_components:
            errors.append(
                "FlashInfer GB10 wheel URLs must include exactly one of each "
                f"required component; missing={missing_components!r}, "
                f"duplicates={duplicate_components!r}."
            )

        if unexpected_components:
            errors.append(
                "FlashInfer release manifest has unexpected wheel components: "
                f"{sorted(unexpected_components)!r}."
            )

        if (
            _mapping_value(flashinfer, "all_required_components_present")
            is not True
            and not missing_components
        ):
            errors.append(
                "FlashInfer release manifest all_required_components_present "
                "does not match the wheel URL set."
            )

        if (
            not missing_components
            and not duplicate_components
            and len(release_identities) != 1
        ):
            release_identity_details = sorted(
                repr(identity) for identity in release_identities
            )
            errors.append(
                "FlashInfer GB10 wheels must come from one GitHub Release "
                f"repo/tag; got {release_identity_details!r}."
            )
    else:
        errors.append("FlashInfer release manifest must list GB10 wheel URLs.")

    flash_attn = _mapping_value(dependencies, "vllm_flash_attn")
    if not _github_repository_url(_mapping_value(flash_attn, "repository")):
        errors.append("vLLM flash-attn repository must be a GitHub HTTPS URL.")
    if _mapping_value(flash_attn, "ref_is_full_git_sha") is not True:
        errors.append("vLLM flash-attn ref must be a full Git SHA.")

    source_dependencies = _mapping_value(dependencies, "source_dependencies")
    if isinstance(source_dependencies, Mapping):
        missing_source_dependencies = sorted(
            set(REQUIRED_SOURCE_DEPENDENCIES) - set(source_dependencies)
        )
        if missing_source_dependencies:
            errors.append(
                "GB10 release manifest source_dependencies must include "
                "DeepGEMM, FlashMLA, and triton_kernels; missing="
                f"{missing_source_dependencies!r}."
            )
        for dependency in source_dependencies.values():
            name = _mapping_value(dependency, "name") or "source dependency"
            if not _github_repository_url(_mapping_value(dependency, "repository")):
                errors.append(f"{name} repository must be a GitHub HTTPS URL.")
            if _mapping_value(dependency, "ref_is_full_git_sha") is not True:
                errors.append(f"{name} ref must be a full Git SHA.")
    else:
        errors.append("GB10 release manifest must list pinned source dependencies.")

    support_matrix = _mapping_value(manifest, "gb10_support_matrix")
    if isinstance(support_matrix, Mapping):
        architecture = _mapping_value(support_matrix, "architecture")
        if architecture != "sm_121a":
            errors.append(
                "GB10 release manifest support matrix architecture must be "
                f"'sm_121a', got {architecture!r}."
            )
        entries = _mapping_value(support_matrix, "entries")
        if isinstance(entries, Mapping):
            missing_entries = sorted(
                set(REQUIRED_GB10_SUPPORT_MATRIX) - set(entries)
            )
            if missing_entries:
                errors.append(
                    "GB10 release manifest support matrix must include all "
                    f"required entries; missing={missing_entries!r}."
                )

            for entry_name, expected_status in REQUIRED_GB10_SUPPORT_MATRIX.items():
                entry = _mapping_value(entries, entry_name)
                status = _mapping_value(entry, "status")
                if status is not None and status not in GB10_SUPPORT_STATUSES:
                    errors.append(
                        "GB10 release manifest support matrix status is not "
                        f"recognized: entry={entry_name!r}, status={status!r}."
                    )
                if status != expected_status:
                    errors.append(
                        "GB10 release manifest support matrix status mismatch: "
                        f"entry={entry_name!r}, expected={expected_status!r}, "
                        f"got {status!r}."
                    )
        else:
            errors.append("GB10 release manifest support matrix must list entries.")
    else:
        errors.append("GB10 release manifest must include a support matrix.")

    build = _mapping_value(manifest, "build")
    if _mapping_value(build, "local_gb10_dependency_checkouts") is True:
        errors.append(
            "GB10 local dependency checkouts are not allowed for release builds."
        )
    if _mapping_value(build, "native_cuda_archs_only") is not True:
        errors.append(
            "GB10 release builds must set native_cuda_archs_only=true so "
            "cross-major PTX fallback images cannot enter SM121A artifacts."
        )
    cache_refs = _mapping_value(build, "cache_refs")
    if isinstance(cache_refs, Mapping):
        for cache_name in ("preflight", "wheel", "runtime"):
            cache_ref = _mapping_value(cache_refs, cache_name)
            if not _ghcr_image_ref(cache_ref):
                errors.append(
                    f"GB10 release manifest build.cache_refs.{cache_name} "
                    "must be a GHCR image ref with a Docker-compatible tag, "
                    f"got {cache_ref!r}."
                )
    else:
        errors.append("GB10 release manifest must list BuildKit cache refs.")

    release = _mapping_value(manifest, "release")
    image = _mapping_value(manifest, "image")
    release_tag = _mapping_value(release, "tag")
    tagged_full_release = (
        isinstance(release_tag, str)
        and bool(release_tag)
        and _mapping_value(release, "preflight_only") is not True
    )
    if tagged_full_release:
        image_name = _mapping_value(image, "name")
        image_tag = _mapping_value(image, "tag")
        if _mapping_value(image, "push") is not True:
            errors.append("GB10 tagged full release requires image.push=true.")
        if not isinstance(image_name, str) or not image_name.startswith("ghcr.io/"):
            errors.append(
                "GB10 tagged full release image.name must be a GHCR image, "
                f"got {image_name!r}."
            )
        elif not _ghcr_image_repository_name(image_name):
            errors.append(
                "GB10 tagged full release image.name must be a GHCR repository "
                f"name without tag or digest, got {image_name!r}."
            )
        if not _docker_image_tag(image_tag):
            errors.append(
                "GB10 tagged full release image.tag must be a Docker-compatible "
                f"tag, got {image_tag!r}."
            )
        if image_tag != release_tag:
            errors.append(
                "GB10 tagged full release image.tag must match release.tag, "
                f"got image.tag={image_tag!r}, release.tag={release_tag!r}."
            )

    return errors


def write_manifest(
    output_path: str | Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    remove_stale_manifest_file(output_path)
    manifest = build_manifest(env)
    write_manifest_file(output_path, manifest)
    return manifest


def remove_stale_manifest_file(output_path: str | Path) -> None:
    path = Path(output_path)
    if path.is_file() or path.is_symlink():
        path.unlink()


def write_manifest_file(
    output_path: str | Path,
    manifest: Mapping[str, object],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write resolved GB10 release settings as a JSON manifest."
    )
    parser.add_argument(
        "--gb10-output-json",
        default=default_release_manifest_json(),
        help="Output path for the GB10 release manifest JSON.",
    )
    parser.add_argument(
        "--gb10-validate-release-inputs",
        action="store_true",
        help=(
            "Validate durable GB10 release inputs before writing the manifest. "
            "Fails on local dependency checkouts, non-SHA refs, missing "
            "FlashInfer release wheels, or tagged full releases without an "
            "image push."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remove_stale_manifest_file(args.gb10_output_json)
    try:
        manifest = build_manifest()
    except ValueError as exc:
        print(f"GB10 release manifest input error: {exc}")
        raise SystemExit(1) from exc
    if args.gb10_validate_release_inputs:
        errors = validate_manifest(manifest)
        if errors:
            print("GB10 release manifest validation failed:")
            for error in errors:
                print(f"- {error}")
            raise SystemExit(1)
        print("GB10 release manifest validation passed.")
    write_manifest_file(args.gb10_output_json, manifest)
    print(f"GB10 release manifest written to {args.gb10_output_json}")


if __name__ == "__main__":
    main()
