import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DEEPGEMM_GIT_TAG = "fb9c137443998c535daaa39aace6685a98352514"
FLASHMLA_GIT_TAG = "cb378f8bf6f76f4998a24b42f3d638f98fc94125"
TRITON_KERNELS_GIT_TAG = "28c73277042f3140a7c8c448913416d24fb57e61"
VLLM_FLASH_ATTN_GIT_TAG = "de3849e75d07edd1c00aec02c92ec852ba757adc"
FLASHINFER_RELEASE_TAG = "gb10-flashinfer-v0.6.12-1c80efb3"
FLASHINFER_RELEASE_WHEELS = (
    "flashinfer_python-0.6.12+cu130gb10-py3-none-any.whl",
    "flashinfer_cubin-0.6.12+cu130gb10-py3-none-any.whl",
    "flashinfer_jit_cache-0.6.12+cu130gb10-cp39-abi3-manylinux_2_28_aarch64.whl",
)
GB10_RELEASE_CACHE_REF_ENV = {
    "GB10_PREFLIGHT_CACHE_REF": "ghcr.io/gardner/vllm-gb10-buildcache:preflight",
    "GB10_WHEEL_CACHE_REF": "ghcr.io/gardner/vllm-gb10-buildcache:wheel",
    "GB10_RUNTIME_CACHE_REF": "ghcr.io/gardner/vllm-gb10-buildcache:runtime",
}
GB10_REQUIRED_SOURCE_DEPENDENCIES = (
    "deepgemm",
    "flashmla",
    "triton_kernels",
)
GB10_FLASHINFER_RUNTIME_DISTRIBUTIONS = (
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)
GB10_EXPECTED_RELEASE_REPORTS = (
    "gb10-nvfp4-smoke.json",
    "gb10-openai-server-smoke-image.json",
    "gb10-release-evidence-image.json",
)
GB10_EXPECTED_RELEASE_EVIDENCE_FILES = (
    *GB10_EXPECTED_RELEASE_REPORTS,
    "gb10-smoked-image-digest.txt",
)
GB10_RELEASE_SMOKED_IMAGE_DIGEST_FILE = "gb10-smoked-image-digest.txt"
GB10_REQUIRED_SUPPORT_MATRIX = {
    "flashinfer_b12x_nvfp4_dense": "supported_native",
    "flashinfer_cutlass_nvfp4_dense": "supported_native",
    "flashinfer_nvfp4_quantization": "supported_native",
    "modelopt_fp4_quantization": "supported_native",
    "flashinfer_attention_fa2": "supported_native",
    "flashinfer_b12x_non_ep_moe": "supported_native",
    "flashinfer_cutlass_non_ep_moe": "supported_native",
    "flashmla_attention": "supported_native",
    "flashmla_sparse_attention": "supported_native",
    "flashinfer_mamba_ssu": "supported_native",
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
    "triton_mla_fallback": "not_supported",
    "flashinfer_trtllm_mla_attention": "not_supported",
    "flashinfer_trtllm_sparse_mla_attention": "not_supported",
    "public_flashattention_mla_runtime": "not_supported",
    "cutlass_mla_sm100_fallback": "not_supported",
    "tokenspeed_mla_cutedsl_fallback": "not_supported",
    "triton_mamba_ssu_fallback": "not_supported",
    "mamba1_triton_runtime": "not_supported",
    "mamba2_triton_ssd_runtime": "not_supported",
    "short_conv_triton_runtime": "not_supported",
    "linear_attention_triton_runtime": "not_supported",
    "speculative_decoding_runtime": "not_supported",
    "pooling_runtime": "not_supported",
    "reasoning_runtime": "not_supported",
    "structured_outputs_runtime": "not_supported",
    "openai_tool_calling_runtime": "not_supported",
    "lora_runtime": "not_supported",
    "gdn_prefill_triton_fallback": "not_supported",
    "gdn_prefill_cutedsl_backend": "not_supported",
    "mm_encoder_fp8_attention": "not_supported",
    "mm_encoder_public_flashattention_backend": "not_supported",
    "mm_encoder_triton_attention_fallback": "not_supported",
    "mm_encoder_torch_sdpa_attention_fallback": "not_supported",
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
    "nvfp4_kv_cache_runtime": "not_supported",
    "unvalidated_kv_cache_runtime": "not_supported",
    "kv_events_runtime": "not_supported",
    "kv_offload_runtime": "not_supported",
    "kv_transfer_runtime": "not_supported",
    "ubatching_runtime": "not_supported",
    "distributed_parallel_runtime": "not_supported",
    "kv_sharing_fast_prefill_runtime": "not_supported",
    "ec_transfer_runtime": "not_supported",
    "weight_transfer_runtime": "not_supported",
    "return_routed_experts_runtime": "not_supported",
    "logprobs_logits_runtime": "not_supported",
    "custom_logits_processors_runtime": "not_supported",
    "io_processor_plugin_runtime": "not_supported",
    "hf_overrides_runtime": "not_supported",
    "transformers_model_impl_runtime": "not_supported",
    "trust_remote_code_runtime": "not_supported",
    "custom_scheduler_runtime": "not_supported",
    "custom_worker_runtime": "not_supported",
    "prompt_embeds_runtime": "not_supported",
    "stock_torch_compile_runtime": "not_supported",
    "mamba_align_cache_runtime": "not_supported",
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
    "compressed_tensors_fp4_kv_cache_loading": "not_supported",
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


def _load_script_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    script_dir = str(script_path.parent)
    added_script_dir = script_dir not in sys.path
    if added_script_dir:
        sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if added_script_dir:
            sys.path.remove(script_dir)
    return module


def _load_gb10_smoke_module():
    return _load_script_module(
        "gb10_smoke_nvfp4",
        REPO_ROOT / "scripts" / "gb10-smoke-nvfp4.py",
    )


def _load_gb10_openai_smoke_module():
    return _load_script_module(
        "gb10_smoke_openai_server",
        REPO_ROOT / "scripts" / "gb10-smoke-openai-server.py",
    )


def _load_gb10_release_evidence_module():
    return _load_script_module(
        "gb10_verify_release_evidence",
        REPO_ROOT / "scripts" / "gb10-verify-release-evidence.py",
    )


def _load_gb10_release_bundle_module():
    return _load_script_module(
        "gb10_bundle_release_evidence",
        REPO_ROOT / "scripts" / "gb10-bundle-release-evidence.py",
    )


def _load_gb10_release_asset_validator_module():
    return _load_script_module(
        "gb10_validate_evidence_release_assets",
        REPO_ROOT / "scripts" / "gb10-validate-evidence-release-assets.py",
    )


def _load_gb10_vllm_release_asset_validator_module():
    return _load_script_module(
        "gb10_validate_vllm_release_assets",
        REPO_ROOT / "scripts" / "gb10-validate-vllm-release-assets.py",
    )


def _load_gb10_runtime_image_provenance_module():
    return _load_script_module(
        "gb10_write_runtime_image_provenance",
        REPO_ROOT / "scripts" / "gb10-write-runtime-image-provenance.py",
    )


def _load_gb10_vllm_release_checksum_writer_module():
    return _load_script_module(
        "gb10_write_vllm_release_checksums",
        REPO_ROOT / "scripts" / "gb10-write-vllm-release-checksums.py",
    )


def _load_gb10_vllm_release_asset_lister_module():
    return _load_script_module(
        "gb10_list_vllm_release_assets",
        REPO_ROOT / "scripts" / "gb10-list-vllm-release-assets.py",
    )


def _load_gb10_evidence_release_asset_lister_module():
    return _load_script_module(
        "gb10_list_evidence_release_assets",
        REPO_ROOT / "scripts" / "gb10-list-evidence-release-assets.py",
    )


def _load_gb10_release_evidence_report_file_lister_module():
    return _load_script_module(
        "gb10_list_release_evidence_report_files",
        REPO_ROOT / "scripts" / "gb10-list-release-evidence-report-files.py",
    )


def _load_gb10_release_provenance_artifact_file_lister_module():
    return _load_script_module(
        "gb10_list_release_provenance_artifact_files",
        REPO_ROOT / "scripts" / "gb10-list-release-provenance-artifact-files.py",
    )


def _load_gb10_image_package_verifier_module():
    return _load_script_module(
        "gb10_verify_image_packages",
        REPO_ROOT / "scripts" / "gb10-verify-image-packages.py",
    )


def _load_gb10_release_contract_module():
    return _load_script_module(
        "gb10_release_contract_for_tests",
        REPO_ROOT / "scripts" / "gb10_release_contract.py",
    )


def _load_gb10_release_manifest_module():
    return _load_script_module(
        "gb10_write_release_manifest",
        REPO_ROOT / "scripts" / "gb10-write-release-manifest.py",
    )


def _load_gb10_release_settings_resolver_module():
    return _load_script_module(
        "gb10_resolve_release_settings",
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py",
    )


def _load_gb10_flashinfer_jit_cache_validator_module():
    return _load_script_module(
        "gb10_verify_flashinfer_jit_cache",
        REPO_ROOT / "docker" / "verify_gb10_flashinfer_jit_cache.py",
    )


def test_gb10_release_scripts_avoid_python311_only_datetime_utc():
    scripts = sorted((REPO_ROOT / "scripts").glob("gb10*.py"))
    assert scripts

    forbidden = ("from datetime import UTC", "datetime.UTC", "datetime.now(UTC)")
    offenders = {}
    for script in scripts:
        text = script.read_text()
        matches = [token for token in forbidden if token in text]
        if matches:
            offenders[script.name] = matches

    assert offenders == {}


def _cuda13_supported_archs() -> list[str]:
    cmake_lists = (REPO_ROOT / "CMakeLists.txt").read_text()
    match = re.search(
        r"CMAKE_CUDA_COMPILER_VERSION\s+VERSION_GREATER_EQUAL\s+13\.0\)\s+"
        r"(?:#[^\n]*\n\s*)*"
        r'set\(CUDA_SUPPORTED_ARCHS "([^"]+)"\)',
        cmake_lists,
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1).split(";")


def _run_cmake_script(script: str, tmp_path: Path) -> str:
    script_path = tmp_path / "check_sm121_arch.cmake"
    output_path = tmp_path / "result.txt"
    script_path.write_text(script)
    subprocess.run(
        ["cmake", "-P", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path.read_text().strip()


def test_cuda13_supported_archs_preserve_native_sm121_target(tmp_path):
    supported_archs = _cuda13_supported_archs()
    assert "12.1" in supported_archs

    script = f"""
cmake_minimum_required(VERSION 3.26)
include("{REPO_ROOT / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(
  CUDA_ARCHS
  "{";".join(supported_archs)}"
  "12.1a")
file(WRITE "{tmp_path / "result.txt"}" "${{CUDA_ARCHS}}")
"""
    assert _run_cmake_script(script, tmp_path) == "12.1a"


def test_cuda13_sm12x_family_fp4_kernels_build_native_sm121a(tmp_path):
    script = f"""
cmake_minimum_required(VERSION 3.26)
include("{REPO_ROOT / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(FP4_ARCHS "12.0f" "12.1a")
file(WRITE "{tmp_path / "result.txt"}" "${{FP4_ARCHS}}")
"""
    assert _run_cmake_script(script, tmp_path) == "12.1a"


def test_native_cuda_archs_only_rejects_cross_major_ptx_fallback(tmp_path):
    script = f"""
cmake_minimum_required(VERSION 3.26)
include("{REPO_ROOT / "cmake" / "utils.cmake"}")
set(VLLM_NATIVE_CUDA_ARCHS_ONLY ON)
cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX;12.0f" "12.1a")
cuda_archs_loose_intersection(MARLIN_FALLBACK_ARCHS "8.0+PTX;9.0+PTX" "12.1a")
file(WRITE "{tmp_path / "result.txt"}"
  "${{MARLIN_ARCHS}}\\n${{MARLIN_FALLBACK_ARCHS}}")
"""
    assert _run_cmake_script(script, tmp_path) == "12.1a"


def test_gb10_build_can_enable_native_cuda_archs_only():
    cmake_lists = (REPO_ROOT / "CMakeLists.txt").read_text()
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()

    assert "VLLM_NATIVE_CUDA_ARCHS_ONLY" in cmake_lists
    assert "Skipping cross-major PTX fallback CUDA archs" in cmake_lists
    assert "ARG vllm_native_cuda_archs_only=false" in dockerfile
    assert (
        "ENV VLLM_NATIVE_CUDA_ARCHS_ONLY=${vllm_native_cuda_archs_only}"
        in dockerfile
    )


def test_gb10_build_uses_pinned_dependency_forks_without_siblings():
    cmake_lists = (REPO_ROOT / "CMakeLists.txt").read_text()
    deepgemm_cmake = (
        REPO_ROOT / "cmake" / "external_projects" / "deepgemm.cmake"
    ).read_text()
    flashmla_cmake = (
        REPO_ROOT / "cmake" / "external_projects" / "flashmla.cmake"
    ).read_text()
    triton_kernels_cmake = (
        REPO_ROOT / "cmake" / "external_projects" / "triton_kernels.cmake"
    ).read_text()

    assert "VLLM_USE_LOCAL_GB10_DEPS" in cmake_lists
    assert "VLLM_USE_LOCAL_GB10_DEPS OFF" in cmake_lists

    assert "DEEPGEMM_GIT_REPOSITORY" in deepgemm_cmake
    assert "https://github.com/gardner/DeepGEMM.git" in deepgemm_cmake
    assert DEEPGEMM_GIT_TAG in deepgemm_cmake
    assert "GIT_REPOSITORY ${DEEPGEMM_GIT_REPOSITORY}" in deepgemm_cmake
    assert "GIT_TAG ${DEEPGEMM_GIT_TAG}" in deepgemm_cmake
    assert "12.0f" in deepgemm_cmake

    assert "FLASH_MLA_GIT_REPOSITORY" in flashmla_cmake
    assert "https://github.com/gardner/FlashMLA.git" in flashmla_cmake
    assert FLASHMLA_GIT_TAG in flashmla_cmake
    assert "GIT_REPOSITORY ${FLASH_MLA_GIT_REPOSITORY}" in flashmla_cmake
    assert "GIT_TAG ${FLASH_MLA_GIT_TAG}" in flashmla_cmake
    assert "csrc/api/api.cpp" in flashmla_cmake
    assert "csrc/sm121/prefill/dense/fmha_dense_prefill_sm121.cu" in flashmla_cmake
    assert "csrc/sm121/decode/dense/dense_decode_sm121.cu" in flashmla_cmake
    assert "12.0f" in flashmla_cmake

    assert "TRITON_KERNELS_GIT_REPOSITORY" in triton_kernels_cmake
    assert "https://github.com/gardner/triton.git" in triton_kernels_cmake
    assert TRITON_KERNELS_GIT_TAG in triton_kernels_cmake
    assert "GIT_REPOSITORY ${TRITON_KERNELS_GIT_REPOSITORY}" in triton_kernels_cmake
    assert "GIT_TAG ${TRITON_KERNELS_GIT_TAG}" in triton_kernels_cmake

    # Local dependency checkouts remain available for development, but are no
    # longer required for a fresh clone to build the GB10 fork.
    assert "../DeepGEMM" in deepgemm_cmake
    assert "DEEPGEMM_SRC_DIR" in deepgemm_cmake
    assert "../FlashMLA" in flashmla_cmake
    assert "FLASH_MLA_SRC_DIR" in flashmla_cmake
    assert "../triton/python/triton_kernels/triton_kernels" in triton_kernels_cmake
    assert "TRITON_KERNELS_SRC_DIR" in triton_kernels_cmake


def test_cuda13_build_uses_gb10_bundled_vllm_flash_attn_source():
    cmake_lists = (REPO_ROOT / "CMakeLists.txt").read_text()
    vllm_flash_attn_cmake = (
        REPO_ROOT / "cmake" / "external_projects" / "vllm_flash_attn.cmake"
    ).read_text()
    setup_py = (REPO_ROOT / "setup.py").read_text()
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert "VLLM_BUILD_FLASH_ATTN" in cmake_lists
    assert "include(cmake/external_projects/vllm_flash_attn.cmake)" in cmake_lists
    assert (
        "VERSION_GREATER_EQUAL 13.0)\n        set(VLLM_BUILD_FLASH_ATTN OFF)"
        not in cmake_lists
    )
    assert "VLLM_FLASH_ATTN_GIT_REPOSITORY" in vllm_flash_attn_cmake
    assert (
        "https://github.com/gardner/vllm-flash-attention.git"
        in vllm_flash_attn_cmake
    )
    assert "VLLM_FLASH_ATTN_GIT_TAG" in vllm_flash_attn_cmake
    assert VLLM_FLASH_ATTN_GIT_TAG in vllm_flash_attn_cmake
    assert "def _build_vllm_flash_attn" in setup_py
    assert 'torch.version.cuda.split(".")[0] in ("12", "13")' in setup_py
    assert "if _build_vllm_flash_attn():" in setup_py
    assert "VLLM_BUILD_FLASH_ATTN=1" in dockerfile
    assert "VLLM_BUILD_FLASH_ATTN_FA3=0" in dockerfile
    assert "gb10_prebuilt_wheel_urls" in dockerfile
    assert "gb10_prebuilt_wheel_urls" in gb10_workflow
    assert "gh release upload" in gb10_workflow
    assert "ghcr.io/gardner/vllm-gb10" in gb10_workflow


def test_gb10_flashinfer_wheels_fail_fast():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    docker_bake = (REPO_ROOT / "docker" / "docker-bake.hcl").read_text()
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()
    versions_json = (REPO_ROOT / "docker" / "versions.json").read_text()

    for text in (dockerfile, docker_bake, gb10_workflow, versions_json):
        assert "gb10_require_flashinfer_wheels" in text

    for text in (dockerfile, gb10_workflow):
        assert "flashinfer_python" in text
        assert "flashinfer_cubin" in text
        assert "flashinfer_jit_cache" in text

    assert "GB10 prebuilt FlashInfer wheel URLs are required" in (
        release_settings_resolver
    )
    assert "GB10 FlashInfer wheels are required" in dockerfile


def test_gb10_release_workflow_defaults_to_published_flashinfer_wheels():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()

    assert "GB10_DEFAULT_PREBUILT_WHEEL_URLS" in gb10_workflow
    assert FLASHINFER_RELEASE_TAG in gb10_workflow
    for wheel in FLASHINFER_RELEASE_WHEELS:
        assert wheel in gb10_workflow
        assert (
            f"https://github.com/gardner/flashinfer/releases/download/"
            f"{FLASHINFER_RELEASE_TAG}/{wheel}"
        ) in gb10_workflow
    assert '"GB10_DEFAULT_PREBUILT_WHEEL_URLS"' in release_settings_resolver


def test_gb10_release_workflow_uses_vllm_dockerfile():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert gb10_workflow.count("--file docker/Dockerfile") == 3


def test_gb10_flashinfer_preflight_target_is_cheap():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    preflight_stage = dockerfile.split(
        "FROM ${BUILD_BASE_IMAGE} AS gb10-flashinfer-preflight",
        1,
    )[1].split("#################### BASE BUILD IMAGE", 1)[0]

    assert "verify_gb10_flashinfer_jit_cache.py" in preflight_stage
    assert "command -v cuobjdump" in preflight_stage
    assert "python3 -m venv /tmp/flashinfer-preflight-venv" in preflight_stage
    assert "GB10 FlashInfer preflight requires gb10_prebuilt_wheel_urls" in (
        preflight_stage
    )
    assert "setup.py bdist_wheel" not in preflight_stage
    assert "requirements/cuda.txt" not in preflight_stage
    assert "uv pip install" not in preflight_stage


def test_gb10_flashinfer_jit_cache_validator_requires_gb10_cuda13_distributions(
    monkeypatch,
):
    validator = _load_gb10_flashinfer_jit_cache_validator_module()
    versions = {
        "flashinfer-python": "0.6.12+cu130gb10",
        "flashinfer-cubin": "0.6.12+cu130gb10",
        "flashinfer-jit-cache": "0.6.12+cu130gb10",
    }

    monkeypatch.setattr(
        validator.importlib_metadata,
        "version",
        lambda distribution: versions[distribution],
    )

    assert validator.validate_distribution_versions() == []

    versions["flashinfer-python"] = "0.6.12+cu130"
    versions["flashinfer-jit-cache"] = "0.6.12"

    errors = validator.validate_distribution_versions()

    assert any("flashinfer-python" in error and "+cu13" in error for error in errors)
    assert any("flashinfer-jit-cache" in error and "gb10" in error for error in errors)


def test_gb10_flashinfer_jit_cache_validator_distribution_contract_is_shared():
    contract = _load_gb10_release_contract_module()
    validator = _load_gb10_flashinfer_jit_cache_validator_module()

    assert validator.REQUIRED_FLASHINFER_DISTRIBUTIONS == (
        contract.FLASHINFER_RUNTIME_DISTRIBUTIONS
    )


def test_gb10_release_workflow_preflights_before_expensive_build():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert gb10_workflow.index("Preflight GB10 FlashInfer wheels") < (
        gb10_workflow.index("Build wheel stage")
    )

    preflight_step = gb10_workflow.split(
        "- name: Preflight GB10 FlashInfer wheels",
        1,
    )[1].split("- name: Build wheel stage", 1)[0]

    assert "--target gb10-flashinfer-preflight" in preflight_step
    assert "--load" not in preflight_step
    assert "--tag vllm-gb10-flashinfer-preflight" not in preflight_step
    assert "--cache-from type=gha,scope=gb10-vllm-preflight" in preflight_step
    assert '--cache-from type=registry,ref="$GB10_PREFLIGHT_CACHE_REF"' in (
        preflight_step
    )
    assert "--cache-to type=gha,scope=gb10-vllm-preflight,mode=max" in (
        preflight_step
    )
    assert '--cache-to type=registry,ref="$GB10_PREFLIGHT_CACHE_REF",mode=max' in (
        preflight_step
    )


def test_gb10_release_workflow_checks_native_arch_contract_before_docker_build():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    script = REPO_ROOT / "scripts" / "gb10-check-native-cuda-arch-contract.py"
    assert script.exists()

    assert "Verify GB10 native CUDA arch contract" in gb10_workflow
    assert (
        "python3 scripts/gb10-check-native-cuda-arch-contract.py "
        "--gb10-require-env"
    ) in gb10_workflow
    assert gb10_workflow.index("Verify GB10 native CUDA arch contract") < (
        gb10_workflow.index("Preflight GB10 FlashInfer wheels")
    )
    assert gb10_workflow.index("Verify GB10 native CUDA arch contract") < (
        gb10_workflow.index("Build wheel stage")
    )


def test_gb10_native_cuda_arch_contract_script_passes_current_release_path():
    script = REPO_ROOT / "scripts" / "gb10-check-native-cuda-arch-contract.py"
    proc = subprocess.run(
        [
            "python3",
            str(script),
            "--gb10-require-env",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "GB10_NATIVE_CUDA_ARCHS_ONLY": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "GB10 native CUDA arch contract OK" in proc.stdout


def test_gb10_release_workflow_supports_manual_preflight_only():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()

    assert "preflight-only:" in gb10_workflow
    assert "GB10_INPUT_PREFLIGHT_ONLY" in gb10_workflow
    assert "GB10_PREFLIGHT_ONLY" in release_settings_resolver

    for step_name in (
        "Build wheel stage",
        "Extract vLLM wheel",
        "Verify wheel contains only SM121A CUDA images",
        "Upload wheel artifact",
        "Build runtime image",
        "Write GB10 runtime image refs",
        "Write GB10 release checksums",
        "Validate GB10 release assets",
        "Publish GB10 release assets",
    ):
        step_block = gb10_workflow.split(f"- name: {step_name}", 1)[1].split(
            "\n      - name:",
            1,
        )[0]
        assert "env.GB10_PREFLIGHT_ONLY != 'true'" in step_block


def test_gb10_release_workflow_routes_full_builds_to_self_hosted_gb10():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()

    assert "runner-labels:" in gb10_workflow
    assert 'default: \'["ubuntu-22.04-arm"]\'' in gb10_workflow
    assert "GB10_SELF_HOSTED_RUNNER_LABELS" in gb10_workflow
    assert "runs-on: ${{ fromJSON(" in gb10_workflow
    assert "inputs['runner-labels']" in gb10_workflow

    resolve_step = gb10_workflow.split(
        "- name: Resolve release settings",
        1,
    )[1].split("- name: Write GB10 release manifest", 1)[0]
    assert "GB10_INPUT_RUNNER_LABELS" in resolve_step
    assert "scripts/gb10-resolve-release-settings.py" in resolve_step
    assert "GB10_RUNNER_LABELS" in release_settings_resolver
    assert "DEFAULT_SELF_HOSTED_RUNNER_LABELS" in release_settings_resolver
    assert "preflight_only != \"true\"" in release_settings_resolver
    assert '"self-hosted" not in _runner_labels(labels_json)' in (
        release_settings_resolver
    )
    assert "GB10 full release builds require self-hosted runner labels" in (
        release_settings_resolver
    )


def test_gb10_release_workflow_requires_pushed_image_for_tagged_release():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()

    resolve_step = gb10_workflow.split(
        "- name: Resolve release settings",
        1,
    )[1].split("- name: Write GB10 release manifest", 1)[0]

    assert "GB10_INPUT_PUSH_IMAGE" in resolve_step
    assert "preflight_only != \"true\"" in release_settings_resolver
    assert "release_tag and push_image != \"true\"" in release_settings_resolver
    assert (
        "GB10 full release publication requires push-image=true"
        in release_settings_resolver
    )


def test_gb10_release_workflow_requires_durable_image_ref_for_tagged_release():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()

    resolve_step = gb10_workflow.split(
        "- name: Resolve release settings",
        1,
    )[1].split("- name: Write GB10 release manifest", 1)[0]

    assert "GB10_INPUT_IMAGE_NAME" in resolve_step
    assert 'image_name.startswith("ghcr.io/")' in release_settings_resolver
    assert "requires a GHCR image-name" in release_settings_resolver
    assert '":" in image_name or "@" in image_name' in release_settings_resolver
    assert "must not include a tag or digest" in release_settings_resolver
    assert 'image_name.removeprefix("ghcr.io/")' in release_settings_resolver
    assert "DOCKER_REPOSITORY_COMPONENT_RE" in release_settings_resolver
    assert "must be a lowercase Docker " in release_settings_resolver
    assert "repository name" in release_settings_resolver
    assert "DOCKER_TAG_RE" in release_settings_resolver
    assert "runtime image tag must be a Docker-compatible " in (
        release_settings_resolver
    )
    assert "tag, got" in release_settings_resolver
    assert "image_tag != release_tag" in release_settings_resolver
    assert "runtime image tag must match the release tag" in release_settings_resolver


def test_gb10_release_workflow_rejects_multiline_env_values():
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()

    assert "def _reject_multiline" in release_settings_resolver
    assert '"\\n" in value' in release_settings_resolver
    assert '"\\r" in value' in release_settings_resolver
    assert "GB10 release setting must be single-line" in release_settings_resolver
    for variable in (
        "GB10_RELEASE_TAG",
        "GB10_IMAGE_NAME",
        "GB10_IMAGE_TAG",
        "GB10_VLLM_VERSION",
        "GB10_PREBUILT_WHEEL_URLS",
        "GB10_FLASH_ATTN_REPO",
        "GB10_FLASH_ATTN_REF",
        "GB10_PUSH_IMAGE",
        "GB10_PREFLIGHT_ONLY",
    ):
        assert f'"{variable}"' in release_settings_resolver


def test_gb10_release_workflow_uses_durable_split_build_caches():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert (
        "GB10_PREFLIGHT_CACHE_REF: "
        "ghcr.io/gardner/vllm-gb10-buildcache:preflight"
    ) in gb10_workflow
    assert "GB10_WHEEL_CACHE_REF: ghcr.io/gardner/vllm-gb10-buildcache:wheel" in (
        gb10_workflow
    )
    assert "GB10_RUNTIME_CACHE_REF: ghcr.io/gardner/vllm-gb10-buildcache:runtime" in (
        gb10_workflow
    )
    assert "--cache-from type=gha,scope=gb10-vllm" in gb10_workflow
    assert "--cache-to type=gha,scope=gb10-vllm,mode=max" not in gb10_workflow
    assert "--cache-from type=gha,scope=gb10-vllm-wheel" in gb10_workflow
    assert "--cache-to type=gha,scope=gb10-vllm-wheel,mode=max" in gb10_workflow
    assert "--cache-from type=gha,scope=gb10-vllm-runtime" in gb10_workflow
    assert "--cache-to type=gha,scope=gb10-vllm-runtime,mode=max" in gb10_workflow
    assert '--cache-from type=registry,ref="$GB10_WHEEL_CACHE_REF"' in gb10_workflow
    assert '--cache-to type=registry,ref="$GB10_WHEEL_CACHE_REF",mode=max' in (
        gb10_workflow
    )
    assert '--cache-from type=registry,ref="$GB10_RUNTIME_CACHE_REF"' in (
        gb10_workflow
    )
    assert '--cache-to type=registry,ref="$GB10_RUNTIME_CACHE_REF",mode=max' in (
        gb10_workflow
    )


def test_gb10_release_workflow_heartbeats_long_buildx_steps():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert 'GB10_BUILD_HEARTBEAT_SECONDS: "300"' in gb10_workflow
    assert "scripts/gb10-run-with-heartbeat.sh" in gb10_workflow
    assert gb10_workflow.count("--progress=plain") == 3

    for step_name in ("Build wheel stage", "Build runtime image"):
        step_block = gb10_workflow.split(f"- name: {step_name}", 1)[1].split(
            "\n      - name:",
            1,
        )[0]
        assert "bash scripts/gb10-run-with-heartbeat.sh" in step_block
        assert "docker buildx build" in step_block
        assert "--progress=plain" in step_block


def test_gb10_build_heartbeat_wrapper_logs_and_preserves_exit_status():
    script = REPO_ROOT / "scripts" / "gb10-run-with-heartbeat.sh"

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "test build",
            "bash",
            "-c",
            "sleep 2.2; exit 7",
        ],
        env={
            **os.environ,
            "GB10_BUILD_HEARTBEAT_SECONDS": "1",
            "GB10_BUILD_HEARTBEAT_DOCKER_DF": "0",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=10,
    )

    assert proc.returncode == 7
    assert "GB10 test build starting" in proc.stdout
    assert "GB10 test build still running" in proc.stdout
    assert proc.stdout.count("GB10 test build resource snapshot") >= 2
    assert "GB10 test build failed with exit code 7" in proc.stdout


def test_gb10_release_workflows_cancel_superseded_runs():
    release_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    smoke_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-smoke-release-image.yml"
    ).read_text()

    assert "concurrency:" in release_workflow
    assert "group: ${{ github.workflow }}-${{ github.ref }}" in release_workflow
    assert "cancel-in-progress: true" in release_workflow

    assert "concurrency:" in smoke_workflow
    assert "group: ${{ github.workflow }}-${{ github.ref }}" in smoke_workflow
    assert "cancel-in-progress: true" in smoke_workflow


def test_gb10_actionlint_config_allows_self_hosted_labels():
    config_path = REPO_ROOT / ".github" / "actionlint.yaml"
    config = yaml.safe_load(config_path.read_text())
    labels = set(config["self-hosted-runner"]["labels"])

    assert {"aarch64", "cuda13", "dgx-spark", "sm121"} <= labels
    assert "self-hosted" not in labels
    assert "linux" not in labels


def test_gb10_release_workflow_uses_conservative_self_hosted_parallelism():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()

    assert "max-jobs:" in gb10_workflow
    assert "nvcc-threads:" in gb10_workflow
    assert 'GB10_MAX_JOBS: "1"' in gb10_workflow
    assert 'GB10_NVCC_THREADS: "1"' in gb10_workflow
    assert 'GB10_NATIVE_CUDA_ARCHS_ONLY: "1"' in gb10_workflow
    assert "1 / 1 = 1 job" in gb10_workflow
    assert "GB10_INPUT_MAX_JOBS" in gb10_workflow
    assert "GB10_INPUT_NVCC_THREADS" in gb10_workflow
    assert '"GB10_MAX_JOBS"' in release_settings_resolver
    assert '"GB10_NVCC_THREADS"' in release_settings_resolver
    assert "GB10 max-jobs must be a positive integer" in release_settings_resolver
    assert "GB10 nvcc-threads must be a positive integer" in release_settings_resolver
    assert gb10_workflow.count('--build-arg max_jobs="$GB10_MAX_JOBS"') == 2
    assert gb10_workflow.count('--build-arg nvcc_threads="$GB10_NVCC_THREADS"') == 2
    assert gb10_workflow.count(
        '--build-arg vllm_native_cuda_archs_only="$GB10_NATIVE_CUDA_ARCHS_ONLY"'
    ) == 2


def _gb10_release_resolver_env(**overrides: str) -> dict[str, str]:
    env = {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/gb10-native-nvfp4",
        "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        "GB10_DEFAULT_PREBUILT_WHEEL_URLS": " ".join(
            f"https://github.com/gardner/flashinfer/releases/download/"
            f"{FLASHINFER_RELEASE_TAG}/{wheel}"
            for wheel in FLASHINFER_RELEASE_WHEELS
        ),
        "GB10_SELF_HOSTED_RUNNER_LABELS": json.dumps(
            ["self-hosted", "linux", "aarch64", "cuda13", "dgx-spark", "sm121"]
        ),
        "GB10_MAX_JOBS": "1",
        "GB10_NVCC_THREADS": "1",
        "GB10_INPUT_RELEASE_TAG": "",
        "GB10_INPUT_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
        "GB10_INPUT_PREBUILT_WHEEL_URLS": "",
        "GB10_INPUT_FLASH_ATTN_REPO": (
            "https://github.com/gardner/vllm-flash-attention.git"
        ),
        "GB10_INPUT_FLASH_ATTN_REF": VLLM_FLASH_ATTN_GIT_TAG,
        "GB10_INPUT_PUSH_IMAGE": "true",
        "GB10_INPUT_PREFLIGHT_ONLY": "true",
        "GB10_INPUT_RUNNER_LABELS": json.dumps(["ubuntu-22.04-arm"]),
        "GB10_INPUT_MAX_JOBS": "1",
        "GB10_INPUT_NVCC_THREADS": "1",
    }
    env.update(overrides)
    return env


def test_gb10_release_settings_resolver_supports_hosted_preflight():
    resolver = _load_gb10_release_settings_resolver_module()

    settings = resolver.resolve_release_settings(_gb10_release_resolver_env())

    assert settings["GB10_RELEASE_TAG"] == ""
    assert settings["GB10_IMAGE_TAG"] == "gb10-abcdef123456"
    assert settings["GB10_VLLM_VERSION"] == "0.22.1rc0+gb10.abcdef123456"
    assert settings["GB10_PREFLIGHT_ONLY"] == "true"
    assert settings["GB10_PUSH_IMAGE"] == "true"
    assert settings["GB10_RUNNER_LABELS"] == json.dumps(["ubuntu-22.04-arm"])
    assert settings["GB10_MAX_JOBS"] == "1"
    assert settings["GB10_NVCC_THREADS"] == "1"
    assert settings["GB10_PREBUILT_WHEEL_URLS"].split() == [
        f"https://github.com/gardner/flashinfer/releases/download/"
        f"{FLASHINFER_RELEASE_TAG}/{wheel}"
        for wheel in FLASHINFER_RELEASE_WHEELS
    ]


def test_gb10_release_settings_resolver_owns_default_flashinfer_wheel_urls():
    resolver = _load_gb10_release_settings_resolver_module()

    settings = resolver.resolve_release_settings(
        _gb10_release_resolver_env(
            GB10_DEFAULT_PREBUILT_WHEEL_URLS="",
            GB10_INPUT_PREBUILT_WHEEL_URLS="",
        )
    )

    assert settings["GB10_PREBUILT_WHEEL_URLS"].split() == [
        f"https://github.com/gardner/flashinfer/releases/download/"
        f"{FLASHINFER_RELEASE_TAG}/{wheel}"
        for wheel in FLASHINFER_RELEASE_WHEELS
    ]


def test_gb10_release_settings_resolver_push_tag_uses_release_defaults():
    resolver = _load_gb10_release_settings_resolver_module()
    release_tag = "gb10-vllm-v0.22.1rc0-abcdef123"

    settings = resolver.resolve_release_settings(
        _gb10_release_resolver_env(
            GITHUB_EVENT_NAME="push",
            GITHUB_REF=f"refs/tags/{release_tag}",
        )
    )

    assert settings["GB10_RELEASE_TAG"] == release_tag
    assert settings["GB10_IMAGE_NAME"] == "ghcr.io/gardner/vllm-gb10"
    assert settings["GB10_IMAGE_TAG"] == release_tag
    assert settings["GB10_VLLM_VERSION"] == "0.22.1rc0+gb10.abcdef123456"
    assert settings["GB10_PUSH_IMAGE"] == "true"
    assert settings["GB10_PREFLIGHT_ONLY"] == "false"
    assert settings["GB10_RUNNER_LABELS"] == json.dumps(
        ["self-hosted", "linux", "aarch64", "cuda13", "dgx-spark", "sm121"]
    )


def test_gb10_release_settings_resolver_rejects_hosted_full_build():
    resolver = _load_gb10_release_settings_resolver_module()

    with pytest.raises(ValueError, match="full release builds require self-hosted"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(GB10_INPUT_PREFLIGHT_ONLY="false")
        )


def test_gb10_release_settings_resolver_rejects_invalid_boolean_inputs():
    resolver = _load_gb10_release_settings_resolver_module()

    with pytest.raises(ValueError, match="GB10_INPUT_PUSH_IMAGE=ture"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(GB10_INPUT_PUSH_IMAGE="ture")
        )

    with pytest.raises(ValueError, match="GB10_INPUT_PREFLIGHT_ONLY=maybe"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(GB10_INPUT_PREFLIGHT_ONLY="maybe")
        )


def test_gb10_release_settings_resolver_rejects_invalid_parallelism():
    resolver = _load_gb10_release_settings_resolver_module()

    with pytest.raises(ValueError, match="max-jobs must be a positive integer"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(GB10_INPUT_MAX_JOBS="0")
        )

    with pytest.raises(ValueError, match="nvcc-threads must be a positive integer"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(GB10_INPUT_NVCC_THREADS="many")
        )


def test_gb10_release_settings_resolver_rejects_bad_tagged_release_image():
    resolver = _load_gb10_release_settings_resolver_module()
    release_tag = "gb10-vllm-v0.22.1rc0-abcdef123"

    with pytest.raises(ValueError, match="requires push-image=true"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(
                GB10_INPUT_RELEASE_TAG=release_tag,
                GB10_INPUT_PREFLIGHT_ONLY="false",
                GB10_INPUT_PUSH_IMAGE="false",
                GB10_INPUT_RUNNER_LABELS=json.dumps(
                    [
                        "self-hosted",
                        "linux",
                        "aarch64",
                        "cuda13",
                        "dgx-spark",
                        "sm121",
                    ]
                ),
            )
        )

    with pytest.raises(ValueError, match="requires a GHCR image-name"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(
                GB10_INPUT_RELEASE_TAG=release_tag,
                GB10_INPUT_PREFLIGHT_ONLY="false",
                GB10_INPUT_IMAGE_NAME="docker.io/gardner/vllm-gb10",
                GB10_INPUT_RUNNER_LABELS=json.dumps(
                    [
                        "self-hosted",
                        "linux",
                        "aarch64",
                        "cuda13",
                        "dgx-spark",
                        "sm121",
                    ]
                ),
            )
        )


def test_gb10_release_settings_resolver_rejects_invalid_release_tag():
    resolver = _load_gb10_release_settings_resolver_module()

    with pytest.raises(ValueError, match="release-tag must match"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(
                GB10_INPUT_RELEASE_TAG="bad-tag",
                GB10_INPUT_PREFLIGHT_ONLY="false",
                GB10_INPUT_PUSH_IMAGE="true",
                GB10_INPUT_IMAGE_NAME="ghcr.io/gardner/vllm-gb10",
                GB10_INPUT_RUNNER_LABELS=json.dumps(
                    [
                        "self-hosted",
                        "linux",
                        "aarch64",
                        "cuda13",
                        "dgx-spark",
                        "sm121",
                    ]
                ),
            )
        )

    with pytest.raises(ValueError, match="release-tag commit suffix"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(
                GB10_INPUT_RELEASE_TAG="gb10-vllm-v0.22.1rc0-deadbee",
                GB10_INPUT_PREFLIGHT_ONLY="false",
                GB10_INPUT_PUSH_IMAGE="true",
                GB10_INPUT_IMAGE_NAME="ghcr.io/gardner/vllm-gb10",
                GB10_INPUT_RUNNER_LABELS=json.dumps(
                    [
                        "self-hosted",
                        "linux",
                        "aarch64",
                        "cuda13",
                        "dgx-spark",
                        "sm121",
                    ]
                ),
            )
        )


def test_gb10_release_settings_resolver_rejects_bad_pushed_image_destination():
    resolver = _load_gb10_release_settings_resolver_module()

    with pytest.raises(ValueError, match="requires a GHCR image-name"):
        resolver.resolve_release_settings(
            _gb10_release_resolver_env(
                GB10_INPUT_RELEASE_TAG="",
                GB10_INPUT_PREFLIGHT_ONLY="false",
                GB10_INPUT_PUSH_IMAGE="true",
                GB10_INPUT_IMAGE_NAME="vllm-gb10",
                GB10_INPUT_RUNNER_LABELS=json.dumps(
                    [
                        "self-hosted",
                        "linux",
                        "aarch64",
                        "cuda13",
                        "dgx-spark",
                        "sm121",
                    ]
                ),
            )
        )


def test_gb10_release_settings_resolver_writes_github_env_file(tmp_path):
    resolver = _load_gb10_release_settings_resolver_module()
    settings = resolver.resolve_release_settings(_gb10_release_resolver_env())
    github_env = tmp_path / "github-env"

    resolver.write_github_env(github_env, settings)

    lines = github_env.read_text().splitlines()
    assert lines[0] == "GB10_RELEASE_TAG="
    assert f"GB10_RUNNER_LABELS={json.dumps(['ubuntu-22.04-arm'])}" in lines
    assert any(line.startswith("GB10_PREBUILT_WHEEL_URLS=") for line in lines)
    assert not any("<<" in line for line in lines)


def test_gb10_release_settings_resolver_writes_shell_safe_exports():
    resolver = _load_gb10_release_settings_resolver_module()
    settings = resolver.resolve_release_settings(_gb10_release_resolver_env())

    lines = resolver.shell_env_lines(settings)

    assert lines[0] == "export GB10_RELEASE_TAG=''"
    assert any(line.startswith("export GB10_PREBUILT_WHEEL_URLS=") for line in lines)
    assert any(
        line == "export GB10_RUNNER_LABELS='[\"ubuntu-22.04-arm\"]'"
        for line in lines
    )
    assert not any("\n" in line for line in lines)


def test_gb10_release_workflow_resolves_settings_with_tested_script():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    resolver_script = REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"

    assert resolver_script.exists()
    assert "scripts/gb10-resolve-release-settings.py" in gb10_workflow
    assert "--gb10-output-env \"$GITHUB_ENV\"" in gb10_workflow
    assert "GB10_INPUT_RELEASE_TAG" in gb10_workflow
    assert "GB10_INPUT_RUNNER_LABELS" in gb10_workflow
    assert "reject_multiline_env_value()" not in gb10_workflow


def test_gb10_release_workflow_sets_runtime_image_build_metadata():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    for step_name in ("Build wheel stage", "Build runtime image"):
        step_block = gb10_workflow.split(f"- name: {step_name}", 1)[1].split(
            "\n      - name:",
            1,
        )[0]
        assert '--build-arg VLLM_BUILD_COMMIT="$GITHUB_SHA"' in step_block
        assert '--build-arg VLLM_BUILD_PIPELINE="$GITHUB_WORKFLOW"' in step_block
        assert (
            "--build-arg "
            'VLLM_BUILD_URL="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/'
            'actions/runs/${GITHUB_RUN_ID}"'
        ) in step_block
        assert '--build-arg VLLM_IMAGE_TAG="$GB10_IMAGE_TAG"' in step_block


def test_gb10_image_package_verifier_requires_gb10_runtime_distributions():
    verifier = _load_gb10_image_package_verifier_module()

    good_versions = {
        "vllm": "0.22.1rc0+gb10.abc123",
        "flashinfer-python": "0.6.12+cu130gb10",
        "flashinfer-cubin": "0.6.12+cu130gb10",
        "flashinfer-jit-cache": "0.6.12+cu130gb10",
    }

    def good_version_getter(distribution_name: str) -> str:
        return good_versions[distribution_name]

    good_report = verifier.collect_image_package_report(
        image_ref="ghcr.io/gardner/vllm-gb10:tag",
        image_digest="ghcr.io/gardner/vllm-gb10@sha256:" + "a" * 64,
        version_getter=good_version_getter,
    )

    assert good_report["status"] == "passed"
    assert good_report["vllm_version_is_gb10"] is True
    assert all(good_report["flashinfer_versions_are_gb10_cuda13"].values())

    bad_versions = dict(good_versions)
    bad_versions["flashinfer-cubin"] = "0.6.12+cu130"

    def bad_version_getter(distribution_name: str) -> str:
        return bad_versions[distribution_name]

    bad_report = verifier.collect_image_package_report(
        version_getter=bad_version_getter,
    )

    assert bad_report["status"] == "failed"
    assert bad_report["non_gb10_flashinfer_distributions"] == {
        "flashinfer-cubin": "0.6.12+cu130",
    }


def test_gb10_release_workflow_publishes_release_manifest():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    contract = _load_gb10_release_contract_module()

    assert "GB10_RELEASE_MANIFEST_DIR" in gb10_workflow
    assert "GB10_RUNTIME_IMAGE_METADATA_JSON" in gb10_workflow
    assert (
        f"GB10_RELEASE_MANIFEST_DIR: "
        f"{contract.DEFAULT_RELEASE_MANIFEST_DIR.as_posix()}"
        in gb10_workflow
    )
    assert (
        "GB10_RUNTIME_IMAGE_METADATA_JSON: "
        f"{contract.DEFAULT_RELEASE_MANIFEST_DIR.as_posix()}/"
        f"{contract.PROVENANCE_FILES['runtime_image_metadata']}"
    ) in gb10_workflow
    assert "${{ github.workspace }}/gb10-release-manifest" not in gb10_workflow
    assert "Write GB10 release manifest" in gb10_workflow
    assert "scripts/gb10-write-release-manifest.py" in gb10_workflow
    assert "--gb10-validate-release-inputs" in gb10_workflow
    assert "gb10-release-manifest.json" in gb10_workflow
    assert "Upload GB10 release inputs" in gb10_workflow
    assert (
        "GB10_RELEASE_INPUTS_ARTIFACT_NAME: "
        f"{contract.GITHUB_RELEASE_INPUTS_ARTIFACT_NAME}"
        in gb10_workflow
    )
    assert "name: ${{ env.GB10_RELEASE_INPUTS_ARTIFACT_NAME }}" in gb10_workflow
    assert (
        "path: ${{ env.GB10_RELEASE_MANIFEST_DIR }}/gb10-release-manifest.json"
        in gb10_workflow
    )
    assert "buildx-runtime-image-metadata.json" in gb10_workflow
    assert "gb10-vllm-release-SHA256SUMS" in gb10_workflow
    release_asset_contract = "\n".join(
        (
            REPO_ROOT / "scripts" / name
        ).read_text()
        for name in (
            "gb10-write-runtime-image-provenance.py",
            "gb10-write-vllm-release-checksums.py",
            "gb10-validate-vllm-release-assets.py",
            "gb10-list-vllm-release-assets.py",
        )
    )
    assert "gb10-runtime-image-ref.txt" in release_asset_contract
    assert "gb10-runtime-image-digest.txt" in release_asset_contract
    assert "gb10-vllm-release-SHA256SUMS" in release_asset_contract
    assert "Write GB10 runtime image refs" in gb10_workflow
    assert "Write GB10 release checksums" in gb10_workflow
    assert "Validate GB10 release assets" in gb10_workflow
    assert "Upload GB10 release manifest" in gb10_workflow
    assert (
        "GB10_RELEASE_MANIFEST_ARTIFACT_NAME: "
        f"{contract.GITHUB_RELEASE_MANIFEST_ARTIFACT_NAME}"
        in gb10_workflow
    )
    assert "name: ${{ env.GB10_RELEASE_MANIFEST_ARTIFACT_NAME }}" in gb10_workflow
    assert "path: ${{ env.GB10_RELEASE_MANIFEST_DIR }}/**" in gb10_workflow
    assert "path: gb10-release-manifest/**" not in gb10_workflow
    assert "if: always()" in gb10_workflow
    assert "Publish GB10 release assets" in gb10_workflow
    assert "Publish wheel to GitHub Release" not in gb10_workflow
    assert '"$GB10_RELEASE_MANIFEST_DIR/gb10-release-manifest.json"' in (
        gb10_workflow
    )
    assert '--metadata-file "$GB10_RUNTIME_IMAGE_METADATA_JSON"' in gb10_workflow

    assert gb10_workflow.index("Write GB10 release manifest") < (
        gb10_workflow.index("Upload GB10 release inputs")
    )
    assert gb10_workflow.index("Upload GB10 release inputs") < (
        gb10_workflow.index("Preflight GB10 FlashInfer wheels")
    )
    assert gb10_workflow.index("Build runtime image") < (
        gb10_workflow.index("Write GB10 runtime image refs")
    )
    assert gb10_workflow.index("Write GB10 runtime image refs") < (
        gb10_workflow.index("Write GB10 release checksums")
    )
    assert gb10_workflow.index("Write GB10 release checksums") < (
        gb10_workflow.index("Validate GB10 release assets")
    )
    assert gb10_workflow.index("Validate GB10 release assets") < (
        gb10_workflow.index("Upload GB10 release manifest")
    )
    assert gb10_workflow.index("Validate GB10 release assets") < (
        gb10_workflow.index("Publish GB10 release assets")
    )
    assert gb10_workflow.index("Build runtime image") < (
        gb10_workflow.index("Publish GB10 release assets")
    )

    checksum_step = gb10_workflow.split(
        "- name: Write GB10 release checksums",
        1,
    )[1].split("- name: Validate GB10 release assets", 1)[0]
    assert "scripts/gb10-write-vllm-release-checksums.py" in checksum_step
    assert "--gb10-dist-dir dist" in checksum_step
    assert '--gb10-release-manifest-dir "$GB10_RELEASE_MANIFEST_DIR"' in (
        checksum_step
    )
    assert '--gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON"' in (
        checksum_step
    )
    assert 'cat "$GB10_RELEASE_MANIFEST_DIR/gb10-vllm-release-SHA256SUMS"' in (
        checksum_step
    )
    assert "checksum_inputs=(" not in checksum_step
    assert "sha256sum" not in checksum_step

    checksum_writer = (
        REPO_ROOT / "scripts" / "gb10-write-vllm-release-checksums.py"
    ).read_text()
    assert "hashlib.sha256" in checksum_writer
    assert "default_release_manifest_dir" in checksum_writer
    assert "default_runtime_image_metadata_json" in checksum_writer
    assert "VLLM_RELEASE_ASSET_FILES" in checksum_writer
    assert "find_vllm_wheel_assets" in checksum_writer
    assert "vllm_release_checksum_asset_paths" in checksum_writer
    assert "GB10 release checksum generation expects exactly one vLLM wheel" in (
        checksum_writer
    )

    validation_step = gb10_workflow.split(
        "- name: Validate GB10 release assets",
        1,
    )[1].split("- name: Upload GB10 release manifest", 1)[0]
    assert "env.GB10_PREFLIGHT_ONLY != 'true'" in validation_step
    assert "env.GB10_RELEASE_TAG != ''" in validation_step
    assert "scripts/gb10-validate-vllm-release-assets.py" in validation_step
    assert "--gb10-dist-dir dist" in validation_step
    assert '--gb10-release-manifest-dir "$GB10_RELEASE_MANIFEST_DIR"' in (
        validation_step
    )
    assert "$GB10_RUNTIME_IMAGE_METADATA_JSON" in validation_step
    assert "required_release_assets=(" not in validation_step
    assert "sha256sum --check" not in validation_step

    release_step = gb10_workflow.split(
        "- name: Publish GB10 release assets",
        1,
    )[1]
    assert "env.GB10_PREFLIGHT_ONLY != 'true'" in release_step
    assert "env.GB10_RELEASE_TAG != ''" in release_step
    assert "scripts/gb10-list-vllm-release-assets.py" in release_step
    assert "--gb10-dist-dir dist" in release_step
    assert '--gb10-release-manifest-dir "$GB10_RELEASE_MANIFEST_DIR"' in release_step
    assert '--gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON"' in (
        release_step
    )
    assert 'release_assets_file="$(mktemp)"' in release_step
    assert 'release_ref_file="$(mktemp)"' in release_step
    assert (
        'trap \'rm -f "$release_assets_file" "$release_ref_file"\' EXIT'
        in release_step
    )
    assert '> "$release_assets_file"' in release_step
    assert "mapfile -t release_assets" in release_step
    assert '< "$release_assets_file"' in release_step
    assert "< <(" not in release_step
    assert "dist/vllm-*.whl" not in release_step
    assert "GB10 release publication expects exactly one vLLM wheel" not in (
        release_step
    )
    assert "release_assets=(" not in release_step
    assert "$GB10_RUNTIME_IMAGE_METADATA_JSON" in release_step
    assert 'if [ -f "$GB10_RUNTIME_IMAGE_METADATA_JSON" ]' not in release_step
    assert (
        'if [ -f "$GB10_RELEASE_MANIFEST_DIR/gb10-runtime-image-digest.txt" ]'
        not in release_step
    )
    assert 'repos/${GITHUB_REPOSITORY}/git/ref/tags/${GB10_RELEASE_TAG}' in (
        release_step
    )
    assert "if gh api " in release_step
    assert '> "$release_ref_file" 2>/dev/null; then' in release_step
    assert "|| true" not in release_step
    assert 'tag_sha="$(jq -r \'.object.sha // ""\' "$release_ref_file")"' in (
        release_step
    )
    assert (
        'tag_type="$(jq -r \'.object.type // ""\' "$release_ref_file")"'
        in release_step
    )
    tag_guard = (
        'if [ "$tag_type" != "commit" ] || '
        '[ "$tag_sha" != "$GITHUB_SHA" ]; then'
    )
    assert tag_guard in release_step
    assert "GB10 release tag must point at the workflow commit" in release_step
    assert 'gh release create "$GB10_RELEASE_TAG" --target "$GITHUB_SHA"' in (
        release_step
    )

    asset_lister = (
        REPO_ROOT / "scripts" / "gb10-list-vllm-release-assets.py"
    ).read_text()
    assert "GB10 release publication expects exactly one vLLM wheel" in asset_lister
    assert "default_release_manifest_dir" in asset_lister
    assert "default_runtime_image_metadata_json" in asset_lister
    assert "find_vllm_wheel_assets" in asset_lister
    assert "vllm_release_asset_paths" in asset_lister

    refs_step = gb10_workflow.split(
        "- name: Write GB10 runtime image refs",
        1,
    )[1].split("- name: Write GB10 release checksums", 1)[0]
    assert "scripts/gb10-write-runtime-image-provenance.py" in refs_step
    assert '--gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON"' in (
        refs_step
    )
    assert '--gb10-release-manifest-dir "$GB10_RELEASE_MANIFEST_DIR"' in refs_step
    assert '--gb10-image-name "$GB10_IMAGE_NAME"' in refs_step
    assert '--gb10-image-tag "$GB10_IMAGE_TAG"' in refs_step
    assert '--gb10-push-image "$GB10_PUSH_IMAGE"' in refs_step
    assert "python3 -" not in refs_step
    assert '[[ ! "$image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]' not in refs_step

    provenance_writer = (
        REPO_ROOT / "scripts" / "gb10-write-runtime-image-provenance.py"
    ).read_text()
    assert "containerimage.digest" in provenance_writer
    assert "GB10 pushed runtime image metadata did not include a digest" in (
        provenance_writer
    )
    assert "GB10 runtime image metadata digest is not a valid sha256 digest" in (
        provenance_writer
    )
    assert "SHA256_DIGEST_RE.fullmatch" in provenance_writer
    assert "VLLM_RELEASE_ASSET_FILES" in provenance_writer


def test_gb10_runtime_image_provenance_writer_accepts_pushed_digest(tmp_path):
    writer = _load_gb10_runtime_image_provenance_module()
    metadata_json = tmp_path / "buildx-runtime-image-metadata.json"
    manifest_dir = tmp_path / "gb10-release-manifest"
    digest = "sha256:" + "a" * 64
    metadata_json.write_text(
        json.dumps({"containerimage.digest": digest}) + "\n",
        encoding="utf-8",
    )

    errors = writer.write_runtime_image_provenance(
        runtime_image_metadata_json=metadata_json,
        release_manifest_dir=manifest_dir,
        image_name="ghcr.io/gardner/vllm-gb10",
        image_tag="gb10-test",
        push_image=True,
    )

    assert errors == []
    assert (
        manifest_dir / "gb10-runtime-image-ref.txt"
    ).read_text() == "ghcr.io/gardner/vllm-gb10:gb10-test\n"
    assert (manifest_dir / "gb10-runtime-image-digest.txt").read_text() == (
        digest + "\n"
    )


def test_gb10_runtime_image_provenance_writer_rejects_missing_pushed_digest(
    tmp_path,
):
    writer = _load_gb10_runtime_image_provenance_module()
    metadata_json = tmp_path / "buildx-runtime-image-metadata.json"
    manifest_dir = tmp_path / "gb10-release-manifest"
    metadata_json.write_text(json.dumps({"image": {}}) + "\n", encoding="utf-8")
    stale_ref = manifest_dir / "gb10-runtime-image-ref.txt"
    stale_digest = manifest_dir / "gb10-runtime-image-digest.txt"
    manifest_dir.mkdir()
    stale_ref.write_text("ghcr.io/gardner/vllm-gb10:stale\n", encoding="utf-8")
    stale_digest.write_text("sha256:" + "b" * 64 + "\n", encoding="utf-8")

    errors = writer.write_runtime_image_provenance(
        runtime_image_metadata_json=metadata_json,
        release_manifest_dir=manifest_dir,
        image_name="ghcr.io/gardner/vllm-gb10",
        image_tag="gb10-test",
        push_image=True,
    )

    assert errors == ["GB10 pushed runtime image metadata did not include a digest."]
    assert not stale_ref.exists()
    assert not stale_digest.exists()


def test_gb10_runtime_image_provenance_writer_allows_loaded_image_without_digest(
    tmp_path,
):
    writer = _load_gb10_runtime_image_provenance_module()
    metadata_json = tmp_path / "buildx-runtime-image-metadata.json"
    manifest_dir = tmp_path / "gb10-release-manifest"
    metadata_json.write_text(json.dumps({"image": {}}) + "\n", encoding="utf-8")

    errors = writer.write_runtime_image_provenance(
        runtime_image_metadata_json=metadata_json,
        release_manifest_dir=manifest_dir,
        image_name="vllm-gb10",
        image_tag="gb10-test",
        push_image=False,
    )

    assert errors == []
    assert (
        manifest_dir / "gb10-runtime-image-ref.txt"
    ).read_text() == "vllm-gb10:gb10-test\n"
    assert not (manifest_dir / "gb10-runtime-image-digest.txt").exists()


def test_gb10_runtime_image_provenance_cli_rejects_invalid_push_flag(
    tmp_path,
):
    provenance_script = REPO_ROOT / "scripts" / "gb10-write-runtime-image-provenance.py"
    metadata_json = tmp_path / "buildx-runtime-image-metadata.json"
    manifest_dir = tmp_path / "gb10-release-manifest"
    metadata_json.write_text(json.dumps({"image": {}}) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(provenance_script),
            "--gb10-runtime-image-metadata-json",
            str(metadata_json),
            "--gb10-release-manifest-dir",
            str(manifest_dir),
            "--gb10-image-name",
            "ghcr.io/gardner/vllm-gb10",
            "--gb10-image-tag",
            "gb10-test",
            "--gb10-push-image",
            "maybe",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "Unsupported GB10 boolean setting --gb10-push-image=maybe" in proc.stdout
    assert "Traceback" not in proc.stdout
    assert not (manifest_dir / "gb10-runtime-image-ref.txt").exists()
    assert not (manifest_dir / "gb10-runtime-image-digest.txt").exists()


def test_gb10_runtime_image_provenance_writer_rejects_malformed_digest(
    tmp_path,
):
    writer = _load_gb10_runtime_image_provenance_module()
    metadata_json = tmp_path / "buildx-runtime-image-metadata.json"
    manifest_dir = tmp_path / "gb10-release-manifest"
    metadata_json.write_text(
        json.dumps({"containerimage.descriptor": {"digest": "sha256:not-a-digest"}})
        + "\n",
        encoding="utf-8",
    )

    errors = writer.write_runtime_image_provenance(
        runtime_image_metadata_json=metadata_json,
        release_manifest_dir=manifest_dir,
        image_name="ghcr.io/gardner/vllm-gb10",
        image_tag="gb10-test",
        push_image=True,
    )

    assert errors == [
        "GB10 runtime image metadata digest is not a valid sha256 digest: "
        "sha256:not-a-digest"
    ]
    assert not (manifest_dir / "gb10-runtime-image-ref.txt").exists()
    assert not (manifest_dir / "gb10-runtime-image-digest.txt").exists()


def test_gb10_vllm_release_asset_helpers_share_contract(tmp_path):
    contract = _load_gb10_release_contract_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)
    wheel = dist_dir / "vllm-0.22.1rc0+gb10.test-cp313-cp313-linux_aarch64.whl"
    manifest = manifest_dir / "gb10-release-manifest.json"
    image_ref = manifest_dir / "gb10-runtime-image-ref.txt"
    image_digest = manifest_dir / "gb10-runtime-image-digest.txt"
    checksums = manifest_dir / "gb10-vllm-release-SHA256SUMS"

    assert contract.find_vllm_wheel_assets(dist_dir) == [wheel]
    assert contract.vllm_release_asset_paths(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
    ) == [wheel, manifest, metadata, image_ref, image_digest, checksums]
    assert contract.vllm_release_checksum_asset_paths(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
    ) == [wheel, manifest, metadata, image_ref, image_digest]

    image_digest.unlink()
    assert contract.vllm_release_checksum_asset_paths(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
    ) == [wheel, manifest, metadata, image_ref]

    script_texts = {
        name: (REPO_ROOT / "scripts" / name).read_text()
        for name in (
            "gb10-list-vllm-release-assets.py",
            "gb10-write-vllm-release-checksums.py",
            "gb10-validate-vllm-release-assets.py",
        )
    }
    assert "vllm_release_asset_paths" in script_texts[
        "gb10-list-vllm-release-assets.py"
    ]
    assert "vllm_release_asset_paths" in script_texts[
        "gb10-validate-vllm-release-assets.py"
    ]
    assert "vllm_release_checksum_asset_paths" in script_texts[
        "gb10-write-vllm-release-checksums.py"
    ]


def test_gb10_vllm_release_default_paths_share_contract(monkeypatch):
    contract = _load_gb10_release_contract_module()
    monkeypatch.delenv("GB10_RELEASE_MANIFEST_DIR", raising=False)
    monkeypatch.delenv("GB10_RUNTIME_IMAGE_METADATA_JSON", raising=False)

    assert Path("dist") == contract.DEFAULT_VLLM_RELEASE_DIST_DIR
    assert contract.default_release_manifest_dir() == Path("gb10-release-manifest")
    assert contract.default_release_manifest_json() == (
        Path("gb10-release-manifest") / "gb10-release-manifest.json"
    )
    assert contract.default_runtime_image_metadata_json() == (
        Path("gb10-release-manifest") / "buildx-runtime-image-metadata.json"
    )

    monkeypatch.setenv("GB10_RELEASE_MANIFEST_DIR", "custom-manifest")
    assert contract.default_release_manifest_dir() == Path("custom-manifest")
    assert contract.default_release_manifest_json() == (
        Path("custom-manifest") / "gb10-release-manifest.json"
    )
    assert contract.default_runtime_image_metadata_json() == (
        Path("custom-manifest") / "buildx-runtime-image-metadata.json"
    )

    monkeypatch.setenv("GB10_RELEASE_MANIFEST_JSON", "custom/manifest.json")
    assert contract.default_release_manifest_json() == Path("custom/manifest.json")

    monkeypatch.setenv("GB10_RUNTIME_IMAGE_METADATA_JSON", "custom/metadata.json")
    assert contract.default_runtime_image_metadata_json() == Path(
        "custom/metadata.json"
    )

    script_texts = {
        name: (REPO_ROOT / "scripts" / name).read_text()
        for name in (
            "gb10-list-vllm-release-assets.py",
            "gb10-write-runtime-image-provenance.py",
            "gb10-write-vllm-release-checksums.py",
            "gb10-validate-vllm-release-assets.py",
            "gb10-write-release-manifest.py",
        )
    }
    for name, script_text in script_texts.items():
        if name == "gb10-write-release-manifest.py":
            assert "default_release_manifest_json" in script_text
            continue
        assert "default_release_manifest_dir" in script_text
        assert "default_runtime_image_metadata_json" in script_text
        assert "def _default_manifest_dir" not in script_text
        assert "def _default_runtime_image_metadata_json" not in script_text
    assert "gb10-release-manifest/gb10-release-manifest.json" not in script_texts[
        "gb10-write-release-manifest.py"
    ]
    for name in (
        "gb10-list-vllm-release-assets.py",
        "gb10-write-vllm-release-checksums.py",
        "gb10-validate-vllm-release-assets.py",
    ):
        assert "DEFAULT_VLLM_RELEASE_DIST_DIR" in script_texts[name]


def test_gb10_release_evidence_default_provenance_paths_share_contract(
    monkeypatch,
):
    contract = _load_gb10_release_contract_module()
    for env_var in (
        "GB10_RELEASE_EVIDENCE_MANIFEST_JSON",
        "GB10_RELEASE_MANIFEST_JSON",
        "GB10_RELEASE_MANIFEST_DIR",
        "GB10_RELEASE_EVIDENCE_RUNTIME_IMAGE_METADATA_JSON",
        "GB10_RUNTIME_IMAGE_METADATA_JSON",
    ):
        monkeypatch.delenv(env_var, raising=False)

    assert contract.default_release_evidence_manifest_json() is None
    assert contract.default_release_evidence_runtime_image_metadata_json() is None

    monkeypatch.setenv("GB10_RELEASE_MANIFEST_DIR", "release-manifest-dir")
    assert contract.default_release_evidence_manifest_json() == (
        Path("release-manifest-dir") / "gb10-release-manifest.json"
    )

    monkeypatch.setenv("GB10_RELEASE_MANIFEST_JSON", "release/manifest.json")
    assert contract.default_release_evidence_manifest_json() == Path(
        "release/manifest.json"
    )

    monkeypatch.setenv("GB10_RELEASE_EVIDENCE_MANIFEST_JSON", "evidence/manifest.json")
    assert contract.default_release_evidence_manifest_json() == Path(
        "evidence/manifest.json"
    )

    monkeypatch.setenv("GB10_RUNTIME_IMAGE_METADATA_JSON", "release/metadata.json")
    assert contract.default_release_evidence_runtime_image_metadata_json() == Path(
        "release/metadata.json"
    )

    monkeypatch.setenv(
        "GB10_RELEASE_EVIDENCE_RUNTIME_IMAGE_METADATA_JSON",
        "evidence/metadata.json",
    )
    assert contract.default_release_evidence_runtime_image_metadata_json() == Path(
        "evidence/metadata.json"
    )

    bundler_script = (
        REPO_ROOT / "scripts" / "gb10-bundle-release-evidence.py"
    ).read_text()
    assert "default_release_evidence_manifest_json" in bundler_script
    assert "default_release_evidence_runtime_image_metadata_json" in bundler_script
    assert "def _default_release_manifest_json" not in bundler_script
    assert "def _default_runtime_image_metadata_json" not in bundler_script


def test_gb10_release_evidence_default_dirs_share_contract(monkeypatch):
    contract = _load_gb10_release_contract_module()
    monkeypatch.delenv("GB10_RELEASE_EVIDENCE_REPORT_DIR", raising=False)
    monkeypatch.delenv("GB10_RELEASE_EVIDENCE_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("GB10_RELEASE_PROVENANCE_DIR", raising=False)

    assert contract.default_release_evidence_report_dir() == Path(
        "gb10-smoke-reports"
    )
    assert contract.default_release_evidence_output_dir() == Path(
        "dist/gb10-release-evidence"
    )
    assert contract.default_release_provenance_dir() == Path(
        "gb10-release-provenance"
    )

    monkeypatch.setenv("GB10_RELEASE_EVIDENCE_REPORT_DIR", "custom-reports")
    monkeypatch.setenv("GB10_RELEASE_EVIDENCE_OUTPUT_DIR", "custom-evidence")
    monkeypatch.setenv("GB10_RELEASE_PROVENANCE_DIR", "custom-provenance")
    assert contract.default_release_evidence_report_dir() == Path("custom-reports")
    assert contract.default_release_evidence_output_dir() == Path("custom-evidence")
    assert contract.default_release_provenance_dir() == Path("custom-provenance")

    bundler_script = (
        REPO_ROOT / "scripts" / "gb10-bundle-release-evidence.py"
    ).read_text()
    assert "default_release_evidence_report_dir" in bundler_script
    assert "default_release_evidence_output_dir" in bundler_script
    assert "def _default_report_dir" not in bundler_script
    assert "def _default_output_dir" not in bundler_script

    orchestrator_script = (
        REPO_ROOT / "scripts" / "gb10-smoke-release-image.sh"
    ).read_text()
    assert (
        f'GB10_RELEASE_SMOKE_REPORT_DIR:-$PWD/'
        f'{contract.DEFAULT_RELEASE_EVIDENCE_REPORT_DIR.as_posix()}'
        in orchestrator_script
    )
    report_lister_script = (
        REPO_ROOT / "scripts" / "gb10-list-release-evidence-report-files.py"
    ).read_text()
    assert "default_release_evidence_report_dir" in report_lister_script
    assert "release_evidence_file_paths" in report_lister_script
    assert "scripts/gb10-list-release-evidence-report-files.py" in (
        orchestrator_script
    )
    assert (
        "Evidence report files listed by "
        "scripts/gb10-list-release-evidence-report-files.py."
        in orchestrator_script
    )
    assert (
        f"./{contract.DEFAULT_RELEASE_EVIDENCE_OUTPUT_DIR.as_posix()}"
        in orchestrator_script
    )
    assert (
        "Evidence bundle assets listed by "
        "scripts/gb10-list-evidence-release-assets.py."
        in orchestrator_script
    )
    assert (
        "${GB10_RELEASE_EVIDENCE_OUTPUT_DIR:-./"
        f"{contract.DEFAULT_RELEASE_EVIDENCE_OUTPUT_DIR.as_posix()}"
        not in orchestrator_script
    )


def test_gb10_github_artifact_names_share_contract():
    contract = _load_gb10_release_contract_module()
    provenance_lister = (
        _load_gb10_release_provenance_artifact_file_lister_module()
    )
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    smoke_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-smoke-release-image.yml"
    ).read_text()

    assert contract.GITHUB_RELEASE_MANIFEST_ARTIFACT_NAME == "gb10-release-manifest"
    assert contract.GITHUB_RELEASE_INPUTS_ARTIFACT_NAME == "gb10-release-inputs"
    assert contract.GITHUB_RELEASE_EVIDENCE_ARTIFACT_NAME == "gb10-release-evidence"
    assert (
        contract.DEFAULT_RELEASE_PROVENANCE_DIR.as_posix()
        == "gb10-release-provenance"
    )
    assert (
        "GB10_RELEASE_MANIFEST_ARTIFACT_NAME: "
        f"{contract.GITHUB_RELEASE_MANIFEST_ARTIFACT_NAME}"
        in gb10_workflow
    )
    assert "name: ${{ env.GB10_RELEASE_MANIFEST_ARTIFACT_NAME }}" in gb10_workflow
    assert (
        "GB10_RELEASE_INPUTS_ARTIFACT_NAME: "
        f"{contract.GITHUB_RELEASE_INPUTS_ARTIFACT_NAME}"
        in gb10_workflow
    )
    assert "name: ${{ env.GB10_RELEASE_INPUTS_ARTIFACT_NAME }}" in gb10_workflow
    assert (
        "GB10_RELEASE_MANIFEST_ARTIFACT_NAME: "
        f"{contract.GITHUB_RELEASE_MANIFEST_ARTIFACT_NAME}"
        in smoke_workflow
    )
    assert (
        "GB10_RELEASE_EVIDENCE_ARTIFACT_NAME: "
        f"{contract.GITHUB_RELEASE_EVIDENCE_ARTIFACT_NAME}"
        in smoke_workflow
    )
    assert (
        '--name "$GB10_RELEASE_MANIFEST_ARTIFACT_NAME"'
        in smoke_workflow
    )
    assert "name: ${{ env.GB10_RELEASE_EVIDENCE_ARTIFACT_NAME }}" in (
        smoke_workflow
    )
    assert (
        "GB10_RELEASE_PROVENANCE_DIR: ${{ github.workspace }}/"
        f"{contract.DEFAULT_RELEASE_PROVENANCE_DIR.as_posix()}"
        in smoke_workflow
    )

    provenance_paths = contract.release_provenance_artifact_paths(
        Path("$GB10_RELEASE_PROVENANCE_DIR")
    )
    assert provenance_lister.RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS == (
        contract.RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS
    )
    assert provenance_lister.list_release_provenance_artifact_files(
        Path("$GB10_RELEASE_PROVENANCE_DIR")
    ) == [
        provenance_paths[kind]
        for kind in contract.RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS
    ]
    assert "scripts/gb10-list-release-provenance-artifact-files.py" in (
        smoke_workflow
    )
    assert 'provenance_files_file="$(mktemp)"' in smoke_workflow
    assert "mapfile -t provenance_files" in smoke_workflow
    assert '"${#provenance_files[@]}" -ne 4' in smoke_workflow
    assert 'manifest="${provenance_files[0]}"' in smoke_workflow
    assert 'runtime_metadata="${provenance_files[1]}"' in smoke_workflow
    assert 'runtime_image_ref="${provenance_files[2]}"' in smoke_workflow
    assert 'runtime_image_digest="${provenance_files[3]}"' in smoke_workflow
    assert (
        'manifest="$GB10_RELEASE_PROVENANCE_DIR/gb10-release-manifest.json"'
        not in smoke_workflow
    )
    assert (
        "runtime_metadata="
        '"$GB10_RELEASE_PROVENANCE_DIR/buildx-runtime-image-metadata.json"'
        not in smoke_workflow
    )
    assert (
        "runtime_image_ref="
        '"$GB10_RELEASE_PROVENANCE_DIR/gb10-runtime-image-ref.txt"'
        not in smoke_workflow
    )
    assert (
        "runtime_image_digest="
        '"$GB10_RELEASE_PROVENANCE_DIR/gb10-runtime-image-digest.txt"'
        not in smoke_workflow
    )


def test_gb10_release_evidence_asset_names_share_contract(monkeypatch):
    contract = _load_gb10_release_contract_module()
    bundler = _load_gb10_release_bundle_module()
    validator = _load_gb10_release_asset_validator_module()
    lister = _load_gb10_evidence_release_asset_lister_module()
    smoke_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-smoke-release-image.yml"
    ).read_text()
    bundler_script = (
        REPO_ROOT / "scripts" / "gb10-bundle-release-evidence.py"
    ).read_text()
    validator_script = (
        REPO_ROOT / "scripts" / "gb10-validate-evidence-release-assets.py"
    ).read_text()
    orchestrator_script = (
        REPO_ROOT / "scripts" / "gb10-smoke-release-image.sh"
    ).read_text()

    monkeypatch.delenv("GB10_RELEASE_EVIDENCE_BUNDLE_NAME", raising=False)
    assert contract.default_release_evidence_bundle_name() == (
        "gb10-release-evidence"
    )
    monkeypatch.setenv("GB10_RELEASE_EVIDENCE_BUNDLE_NAME", "custom-evidence")
    assert contract.default_release_evidence_bundle_name() == "custom-evidence"

    assert contract.release_evidence_bundle_archive_name() == (
        "gb10-release-evidence.tar.gz"
    )
    assert contract.release_evidence_bundle_archive_checksum_name() == (
        "gb10-release-evidence.tar.gz.sha256"
    )
    assert [
        path.as_posix()
        for path in contract.release_evidence_asset_paths(Path("out"))
    ] == [
        "out/gb10-release-evidence.tar.gz",
        "out/gb10-release-evidence.tar.gz.sha256",
        "out/release-evidence-metadata.json",
        "out/SHA256SUMS",
    ]

    assert bundler.RELEASE_EVIDENCE_METADATA_FILE == (
        contract.RELEASE_EVIDENCE_METADATA_FILE
    )
    assert bundler.RELEASE_EVIDENCE_CHECKSUM_FILE == (
        contract.RELEASE_EVIDENCE_CHECKSUM_FILE
    )
    assert validator.RELEASE_EVIDENCE_METADATA_FILE == (
        contract.RELEASE_EVIDENCE_METADATA_FILE
    )
    assert validator.RELEASE_EVIDENCE_CHECKSUM_FILE == (
        contract.RELEASE_EVIDENCE_CHECKSUM_FILE
    )
    assert lister.default_release_evidence_bundle_name() == "custom-evidence"
    assert lister.release_evidence_asset_paths(Path("out")) == (
        contract.release_evidence_asset_paths(Path("out"))
    )

    assert "default_release_evidence_bundle_name" in bundler_script
    assert "release_evidence_asset_paths" in bundler_script
    assert "release_evidence_bundle_archive_name" in bundler_script
    assert "release_evidence_bundle_archive_checksum_name" in bundler_script
    assert "default_release_evidence_bundle_name" in validator_script
    assert "release_evidence_asset_paths" in validator_script
    assert "release_evidence_bundle_archive_checksum_name" in validator_script
    assert "scripts/gb10-list-evidence-release-assets.py" in orchestrator_script
    assert "GB10_RELEASE_EVIDENCE_BUNDLE_NAME" in orchestrator_script
    assert "GB10 release evidence assets:" in orchestrator_script
    assert "gb10-release-evidence.tar.gz" not in orchestrator_script
    assert "scripts/gb10-list-evidence-release-assets.py" in smoke_workflow
    assert "$GB10_RELEASE_EVIDENCE_OUTPUT_DIR/gb10-release-evidence.tar.gz" not in (
        smoke_workflow
    )
    assert (
        "$GB10_RELEASE_EVIDENCE_OUTPUT_DIR/gb10-release-evidence.tar.gz.sha256"
        not in smoke_workflow
    )


def test_gb10_vllm_release_checksum_writer_writes_release_assets(tmp_path):
    checksum_writer = _load_gb10_vllm_release_checksum_writer_module()
    validator = _load_gb10_vllm_release_asset_validator_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)
    checksum_file = manifest_dir / "gb10-vllm-release-SHA256SUMS"
    checksum_file.unlink()

    errors = checksum_writer.write_release_checksums(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
        repo_root=tmp_path,
    )

    assert errors == []
    checksum_lines = checksum_file.read_text(encoding="utf-8").splitlines()
    checksum_paths = {line.split(maxsplit=1)[1] for line in checksum_lines}
    assert checksum_paths == {
        "dist/vllm-0.22.1rc0+gb10.test-cp313-cp313-linux_aarch64.whl",
        "gb10-release-manifest/gb10-release-manifest.json",
        "gb10-release-manifest/buildx-runtime-image-metadata.json",
        "gb10-release-manifest/gb10-runtime-image-ref.txt",
        "gb10-release-manifest/gb10-runtime-image-digest.txt",
    }
    assert validator.validate_release_assets(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
        repo_root=tmp_path,
    ) == []


def test_gb10_vllm_release_checksum_writer_omits_absent_digest(tmp_path):
    checksum_writer = _load_gb10_vllm_release_checksum_writer_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)
    checksum_file = manifest_dir / "gb10-vllm-release-SHA256SUMS"
    checksum_file.unlink()
    (manifest_dir / "gb10-runtime-image-digest.txt").unlink()

    errors = checksum_writer.write_release_checksums(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
        repo_root=tmp_path,
    )

    assert errors == []
    checksum_text = checksum_file.read_text(encoding="utf-8")
    assert "gb10-runtime-image-ref.txt" in checksum_text
    assert "gb10-runtime-image-digest.txt" not in checksum_text


def test_gb10_vllm_release_checksum_writer_rejects_missing_wheel(tmp_path):
    checksum_writer = _load_gb10_vllm_release_checksum_writer_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)
    checksum_file = manifest_dir / "gb10-vllm-release-SHA256SUMS"
    for wheel in dist_dir.glob("vllm-*.whl"):
        wheel.unlink()
    checksum_file.write_text("stale\n", encoding="utf-8")

    errors = checksum_writer.write_release_checksums(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
        repo_root=tmp_path,
    )

    assert any("expects exactly one vLLM wheel" in error for error in errors)
    assert not checksum_file.exists()


def test_gb10_vllm_release_asset_lister_returns_publish_assets(tmp_path):
    asset_lister = _load_gb10_vllm_release_asset_lister_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)

    release_assets, errors = asset_lister.list_release_assets(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
    )

    assert errors == []
    assert release_assets == [
        dist_dir / "vllm-0.22.1rc0+gb10.test-cp313-cp313-linux_aarch64.whl",
        manifest_dir / "gb10-release-manifest.json",
        metadata,
        manifest_dir / "gb10-runtime-image-ref.txt",
        manifest_dir / "gb10-runtime-image-digest.txt",
        manifest_dir / "gb10-vllm-release-SHA256SUMS",
    ]


def test_gb10_vllm_release_asset_lister_rejects_bad_assets(tmp_path):
    asset_lister = _load_gb10_vllm_release_asset_lister_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)
    (dist_dir / "vllm-extra-0.0.0.whl").write_bytes(b"extra")
    (manifest_dir / "gb10-runtime-image-digest.txt").write_text("", encoding="utf-8")

    release_assets, errors = asset_lister.list_release_assets(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
    )

    assert len(release_assets) == 7
    assert any("expects exactly one vLLM wheel" in error for error in errors)
    assert any("missing or empty" in error for error in errors)


def _write_gb10_vllm_release_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    dist_dir = tmp_path / "dist"
    manifest_dir = tmp_path / "gb10-release-manifest"
    dist_dir.mkdir()
    manifest_dir.mkdir()
    wheel = dist_dir / "vllm-0.22.1rc0+gb10.test-cp313-cp313-linux_aarch64.whl"
    manifest = manifest_dir / "gb10-release-manifest.json"
    metadata = manifest_dir / "buildx-runtime-image-metadata.json"
    image_ref = manifest_dir / "gb10-runtime-image-ref.txt"
    image_digest = manifest_dir / "gb10-runtime-image-digest.txt"
    checksum_file = manifest_dir / "gb10-vllm-release-SHA256SUMS"
    manifest_data = _load_gb10_release_manifest_module().build_manifest(
        {
            "GITHUB_WORKFLOW": "GB10 vLLM wheel and image",
            "GITHUB_REPOSITORY": "gardner/vllm",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REF": "refs/tags/gb10-vllm-v0.22.1rc0-abcdef123",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
            "GITHUB_RUN_ID": "12345",
            "GB10_RELEASE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
            "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
            "GB10_IMAGE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
            "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
            "GB10_PREBUILT_WHEEL_URLS": " ".join(
                f"https://github.com/gardner/flashinfer/releases/download/"
                f"{FLASHINFER_RELEASE_TAG}/{wheel}"
                for wheel in FLASHINFER_RELEASE_WHEELS
            ),
            "GB10_FLASH_ATTN_REPO": (
                "https://github.com/gardner/vllm-flash-attention.git"
            ),
            "GB10_FLASH_ATTN_REF": VLLM_FLASH_ATTN_GIT_TAG,
            "GB10_PUSH_IMAGE": "true",
            "GB10_PREFLIGHT_ONLY": "false",
            "GB10_MAX_JOBS": "1",
            "GB10_NVCC_THREADS": "1",
            "GB10_NATIVE_CUDA_ARCHS_ONLY": "1",
            "GB10_RUNNER_LABELS": json.dumps(
                ["self-hosted", "linux", "aarch64", "cuda13", "dgx-spark", "sm121"]
            ),
            **GB10_RELEASE_CACHE_REF_ENV,
            "VLLM_USE_LOCAL_GB10_DEPS": "0",
        }
    )

    for path, content in (
        (wheel, b"wheel"),
        (
            manifest,
            json.dumps(manifest_data, indent=2, sort_keys=True).encode() + b"\n",
        ),
        (metadata, b'{"containerimage.digest":"sha256:' + b"a" * 64 + b'"}\n'),
        (image_ref, b"ghcr.io/gardner/vllm-gb10:gb10-test\n"),
        (image_digest, b"sha256:" + b"a" * 64 + b"\n"),
    ):
        path.write_bytes(content)

    checksum_assets = [wheel, manifest, metadata, image_ref, image_digest]
    checksum_lines = []
    for path in checksum_assets:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.relative_to(tmp_path).as_posix()}")
    checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return dist_dir, manifest_dir, metadata


def test_gb10_vllm_release_asset_validator_accepts_complete_assets(tmp_path):
    validator = _load_gb10_vllm_release_asset_validator_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)

    assert validator.validate_release_assets(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
        repo_root=tmp_path,
    ) == []


def test_gb10_vllm_release_asset_validator_rejects_invalid_manifest(tmp_path):
    validator = _load_gb10_vllm_release_asset_validator_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)
    manifest_path = manifest_dir / "gb10-release-manifest.json"
    manifest_path.write_text('{"schema_version":1}\n', encoding="utf-8")
    checksum_file = manifest_dir / "gb10-vllm-release-SHA256SUMS"
    checksum_lines = []
    for raw_line in checksum_file.read_text(encoding="utf-8").splitlines():
        _digest, raw_path = raw_line.split(maxsplit=1)
        asset = tmp_path / raw_path
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {raw_path}")
    checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    errors = validator.validate_release_assets(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
        repo_root=tmp_path,
    )

    assert any("release manifest validation failed" in error for error in errors)
    assert any("git.commit must be a full Git SHA" in error for error in errors)


def test_gb10_vllm_release_asset_validator_rejects_bad_assets(tmp_path):
    validator = _load_gb10_vllm_release_asset_validator_module()
    dist_dir, manifest_dir, metadata = _write_gb10_vllm_release_assets(tmp_path)

    (dist_dir / "vllm-extra-0.0.0.whl").write_bytes(b"extra")
    (manifest_dir / "gb10-runtime-image-digest.txt").write_text("", encoding="utf-8")
    (manifest_dir / "gb10-vllm-release-SHA256SUMS").write_text(
        "0" * 64
        + "  dist/vllm-0.22.1rc0+gb10.test-cp313-cp313-linux_aarch64.whl\n",
        encoding="utf-8",
    )

    errors = validator.validate_release_assets(
        dist_dir=dist_dir,
        release_manifest_dir=manifest_dir,
        runtime_image_metadata_json=metadata,
        repo_root=tmp_path,
    )

    assert any("expects exactly one vLLM wheel" in error for error in errors)
    assert any("missing or empty" in error for error in errors)
    assert any("checksum mismatch" in error for error in errors)
    assert any("checksums omit required assets" in error for error in errors)


def test_gb10_release_manifest_records_resolved_inputs(tmp_path):
    manifest = _load_gb10_release_manifest_module()
    env = {
        "GITHUB_WORKFLOW": "GB10 vLLM wheel and image",
        "GITHUB_REPOSITORY": "gardner/vllm",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REF": "refs/tags/gb10-vllm-v0.22.1rc0-abcdef123",
        "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        "GITHUB_RUN_ID": "12345",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GB10_RELEASE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
        "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
        "GB10_IMAGE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
        "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
        "GB10_PREBUILT_WHEEL_URLS": " ".join(
            f"https://github.com/gardner/flashinfer/releases/download/"
            f"{FLASHINFER_RELEASE_TAG}/{wheel}"
            for wheel in FLASHINFER_RELEASE_WHEELS
        ),
        "GB10_FLASH_ATTN_REPO": "https://github.com/gardner/vllm-flash-attention.git",
        "GB10_FLASH_ATTN_REF": VLLM_FLASH_ATTN_GIT_TAG,
        "GB10_PUSH_IMAGE": "true",
        "GB10_PREFLIGHT_ONLY": "true",
        "GB10_MAX_JOBS": "1",
        "GB10_NVCC_THREADS": "1",
        "GB10_NATIVE_CUDA_ARCHS_ONLY": "1",
        "GB10_RUNNER_LABELS": json.dumps(
            ["self-hosted", "linux", "aarch64", "cuda13", "dgx-spark", "sm121"]
        ),
        **GB10_RELEASE_CACHE_REF_ENV,
        "VLLM_USE_LOCAL_GB10_DEPS": "0",
    }

    output_path = tmp_path / "gb10-release-manifest.json"
    manifest.write_manifest(output_path, env=env)
    data = json.loads(output_path.read_text())

    assert data["schema_version"] == 1
    assert data["git"]["commit"] == env["GITHUB_SHA"]
    assert data["github"]["run_url"] == (
        "https://github.com/gardner/vllm/actions/runs/12345/attempts/2"
    )
    assert data["release"]["tag"] == env["GB10_RELEASE_TAG"]
    assert data["release"]["preflight_only"] is True
    assert data["image"] == {
        "name": "ghcr.io/gardner/vllm-gb10",
        "tag": "gb10-vllm-v0.22.1rc0-abcdef123",
        "push": True,
    }
    assert data["vllm"]["version"] == "0.22.1rc0+gb10.abcdef123456"
    assert data["dependencies"]["vllm_flash_attn"] == {
        "repository": "https://github.com/gardner/vllm-flash-attention.git",
        "ref": VLLM_FLASH_ATTN_GIT_TAG,
        "ref_is_full_git_sha": True,
    }
    assert data["build"]["parallelism"] == {"max_jobs": "1", "nvcc_threads": "1"}
    assert data["build"]["native_cuda_archs_only"] is True
    assert data["build"]["runner_labels"] == [
        "self-hosted",
        "linux",
        "aarch64",
        "cuda13",
        "dgx-spark",
        "sm121",
    ]
    assert data["build"]["local_gb10_dependency_checkouts"] is False
    assert data["build"]["cache_refs"]["runtime"] == (
        "ghcr.io/gardner/vllm-gb10-buildcache:runtime"
    )

    wheels = data["dependencies"]["flashinfer"]["wheels"]
    assert [wheel["component"] for wheel in wheels] == [
        "flashinfer_python",
        "flashinfer_cubin",
        "flashinfer_jit_cache",
    ]
    assert {wheel["release_repository"] for wheel in wheels} == {
        "gardner/flashinfer"
    }
    assert {wheel["release_tag"] for wheel in wheels} == {FLASHINFER_RELEASE_TAG}
    assert data["dependencies"]["flashinfer"]["all_required_components_present"] is True
    support_matrix = data["gb10_support_matrix"]
    assert support_matrix["architecture"] == "sm_121a"
    assert support_matrix["first_release_scope"] == "single_spark_first_path"
    assert support_matrix["entries"]["flashinfer_b12x_nvfp4_dense"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["flashinfer_cutlass_nvfp4_dense"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["modelopt_fp4_quantization"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["compressed_tensors_w4a4_nvfp4_dense_loading"][
        "status"
    ] == "supported_native"
    assert support_matrix["entries"]["compressed_tensors_w4a4_nvfp4_moe_loading"][
        "status"
    ] == "supported_native"
    assert support_matrix["entries"][
        "compressed_tensors_qutlass_nvfp4_transform_loading"
    ]["status"] == "not_supported"
    assert support_matrix["entries"]["flashmla_attention"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["flashmla_sparse_attention"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["flashinfer_mamba_ssu"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["flashinfer_gdn_prefill"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["public_flashattention_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["flashinfer_trtllm_nvfp4_dense"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["flashinfer_trtllm_mxfp4_moe"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["flashinfer_cutedsl_nvfp4_moe"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["triton_mla_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["flashinfer_trtllm_mla_attention"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["flashinfer_trtllm_sparse_mla_attention"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["public_flashattention_mla_runtime"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["cutlass_mla_sm100_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["tokenspeed_mla_cutedsl_fallback"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["triton_mamba_ssu_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mamba1_triton_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mamba2_triton_ssd_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["short_conv_triton_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["linear_attention_triton_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["speculative_decoding_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["pooling_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["reasoning_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["structured_outputs_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["openai_tool_calling_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["gdn_prefill_triton_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["gdn_prefill_cutedsl_backend"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mm_encoder_fp8_attention"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mm_encoder_public_flashattention_backend"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["mm_encoder_triton_attention_fallback"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["mm_encoder_torch_sdpa_attention_fallback"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["public_fp8_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["deepseek_v4_fp8_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["torchao_fp8_activation_quantization"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["torchao_weight_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["bitsandbytes_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["awq_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["gptq_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["inc_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["gguf_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["humming_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["humming_mxfp4_moe_backend"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["rocm_aiter_unquantized_moe"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["unquantized_moe_triton_fallback"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["rocm_aiter_fp8_moe"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["deep_gemm_fp8_moe"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["triton_fp8_moe"]["status"] == "not_supported"
    assert support_matrix["entries"]["vllm_cutlass_fp8_moe"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["rocm_aiter_mxfp4_moe"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["gpt_oss_triton_mxfp4_moe"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["modelopt_w4a16_nvfp4_checkpoint_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["modelopt_nvfp4_kv_cache_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["nvfp4_kv_cache_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["unvalidated_kv_cache_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["kv_events_runtime"]["status"] == "not_supported"
    assert support_matrix["entries"]["kv_offload_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["kv_transfer_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["ubatching_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["distributed_parallel_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["kv_sharing_fast_prefill_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["ec_transfer_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["weight_transfer_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["return_routed_experts_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["logprobs_logits_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["custom_logits_processors_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["io_processor_plugin_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["hf_overrides_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["transformers_model_impl_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["trust_remote_code_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["custom_scheduler_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["custom_worker_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["prompt_embeds_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["stock_torch_compile_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mamba_align_cache_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["marlin_mxfp4_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mxfp4_moe_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["public_mxfp4_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["fp8_w8a16_marlin_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["fp8_w8a16_moe_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["int8_moe_triton_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["wna16_moe_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["compressed_tensors_wna16_dense_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_wna16_moe_fallback"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["moe_wna16_legacy_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mxfp8_dense_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mxfp8_moe_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["modelopt_fp8_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["modelopt_mxfp8_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["modelopt_mixed_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["fbgemm_fp8_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["fbgemm_nvfp4_dense"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["experts_int8_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["fp_quant_fp4_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["online_fp8_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["online_mxfp8_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["online_mxfp4_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["online_int8_moe_quantization"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["compressed_tensors_w8a8_mxfp8_dense_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w8a8_mxfp8_moe_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["quark_nvfp4_checkpoint_loading"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["quark_ocp_mx_checkpoint_loading"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["quark_w4a8_mxfp4_fp8_checkpoint_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["quark_w4a8_fp8_moe_loading"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["quark_w8a8_fp8_checkpoint_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["quark_w8a8_int8_checkpoint_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["quark_w8a8_fp8_moe_loading"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["quark_w8a8_int8_moe_loading"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["compressed_tensors_fp4_kv_cache_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w4a8_fp8_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w4a8_int_dense_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w4a8_int_moe_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w8a16_fp8_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w8a8_fp8_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w8a8_fp8_moe_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w8a8_int_dense_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w8a8_int_moe_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w4a4_mxfp4_dense_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w4a4_mxfp4_moe_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w4a16_nvfp4_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["compressed_tensors_w4a16_nvfp4_moe_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["trtllm_gen_attention"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["flashinfer_cudnn_nvfp4_dense"]["status"] == (
        "deferred"
    )
    assert support_matrix["entries"]["deepseek_v4_deep_gemm_mega_moe"][
        "status"
    ] == "deferred"
    assert support_matrix["entries"]["multi_spark_ep_all2all_eplb"]["status"] == (
        "deferred"
    )

    source_dependencies = data["dependencies"]["source_dependencies"]
    assert source_dependencies["deepgemm"] == {
        "name": "DeepGEMM",
        "cmake_file": "cmake/external_projects/deepgemm.cmake",
        "repository": "https://github.com/gardner/DeepGEMM.git",
        "ref": DEEPGEMM_GIT_TAG,
        "ref_is_full_git_sha": True,
    }
    assert source_dependencies["flashmla"] == {
        "name": "FlashMLA",
        "cmake_file": "cmake/external_projects/flashmla.cmake",
        "repository": "https://github.com/gardner/FlashMLA.git",
        "ref": FLASHMLA_GIT_TAG,
        "ref_is_full_git_sha": True,
    }
    assert source_dependencies["triton_kernels"] == {
        "name": "triton_kernels",
        "cmake_file": "cmake/external_projects/triton_kernels.cmake",
        "repository": "https://github.com/gardner/triton.git",
        "ref": TRITON_KERNELS_GIT_TAG,
        "ref_is_full_git_sha": True,
    }


def test_gb10_release_manifest_rejects_invalid_boolean_inputs_before_write(
    tmp_path,
):
    manifest = _load_gb10_release_manifest_module()
    env = {
        "GITHUB_WORKFLOW": "GB10 vLLM wheel and image",
        "GITHUB_REPOSITORY": "gardner/vllm",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REF": "refs/tags/gb10-vllm-v0.22.1rc0-abcdef123",
        "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        "GITHUB_RUN_ID": "12345",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GB10_RELEASE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
        "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
        "GB10_IMAGE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
        "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
        "GB10_PREBUILT_WHEEL_URLS": " ".join(
            f"https://github.com/gardner/flashinfer/releases/download/"
            f"{FLASHINFER_RELEASE_TAG}/{wheel}"
            for wheel in FLASHINFER_RELEASE_WHEELS
        ),
        "GB10_FLASH_ATTN_REPO": "https://github.com/gardner/vllm-flash-attention.git",
        "GB10_FLASH_ATTN_REF": VLLM_FLASH_ATTN_GIT_TAG,
        "GB10_PUSH_IMAGE": "true",
        "GB10_PREFLIGHT_ONLY": "true",
        "GB10_MAX_JOBS": "1",
        "GB10_NVCC_THREADS": "1",
        "GB10_NATIVE_CUDA_ARCHS_ONLY": "1",
        "GB10_RUNNER_LABELS": json.dumps(
            ["self-hosted", "linux", "aarch64", "cuda13", "dgx-spark", "sm121"]
        ),
        **GB10_RELEASE_CACHE_REF_ENV,
        "VLLM_USE_LOCAL_GB10_DEPS": "0",
    }

    for name, bad_value in (
        ("GB10_PREFLIGHT_ONLY", "ture"),
        ("GB10_PUSH_IMAGE", "maybe"),
        ("GB10_NATIVE_CUDA_ARCHS_ONLY", "enabled"),
        ("VLLM_USE_LOCAL_GB10_DEPS", "flase"),
    ):
        output_path = tmp_path / f"{name}.json"
        invalid_env = {**env, name: bad_value}

        with pytest.raises(ValueError, match=f"{name}={bad_value}"):
            manifest.write_manifest(output_path, env=invalid_env)

        assert not output_path.exists()


def test_gb10_release_manifest_cli_rejects_invalid_boolean_without_traceback(
    tmp_path,
):
    manifest_script = REPO_ROOT / "scripts" / "gb10-write-release-manifest.py"
    output_path = tmp_path / "gb10-release-manifest.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(manifest_script),
            "--gb10-output-json",
            str(output_path),
            "--gb10-validate-release-inputs",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_PUSH_IMAGE": "maybe",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "Unsupported GB10 boolean setting GB10_PUSH_IMAGE=maybe" in proc.stdout
    assert "Traceback" not in proc.stdout
    assert not output_path.exists()


def test_gb10_release_manifest_cli_validates_release_inputs_before_write(
    tmp_path,
):
    manifest_script = REPO_ROOT / "scripts" / "gb10-write-release-manifest.py"
    output_path = tmp_path / "gb10-release-manifest.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(manifest_script),
            "--gb10-output-json",
            str(output_path),
            "--gb10-validate-release-inputs",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
            "GB10_RELEASE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
            "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
            "GB10_IMAGE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
            "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
            "GB10_PREBUILT_WHEEL_URLS": " ".join(
                f"https://github.com/gardner/flashinfer/releases/download/"
                f"{FLASHINFER_RELEASE_TAG}/{wheel}"
                for wheel in FLASHINFER_RELEASE_WHEELS
            ),
            "GB10_FLASH_ATTN_REPO": (
                "https://github.com/gardner/vllm-flash-attention.git"
            ),
            "GB10_FLASH_ATTN_REF": "main",
            "GB10_PUSH_IMAGE": "true",
            "GB10_PREFLIGHT_ONLY": "false",
            "GB10_NATIVE_CUDA_ARCHS_ONLY": "1",
            **GB10_RELEASE_CACHE_REF_ENV,
            "VLLM_USE_LOCAL_GB10_DEPS": "0",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10 release manifest validation failed" in proc.stdout
    assert "vLLM flash-attn ref must be a full Git SHA" in proc.stdout
    assert not output_path.exists()


def test_gb10_release_manifest_cli_removes_stale_output_before_validation(
    tmp_path,
):
    manifest_script = REPO_ROOT / "scripts" / "gb10-write-release-manifest.py"
    output_path = tmp_path / "gb10-release-manifest.json"
    output_path.write_text('{"stale": true}\n')

    proc = subprocess.run(
        [
            sys.executable,
            str(manifest_script),
            "--gb10-output-json",
            str(output_path),
            "--gb10-validate-release-inputs",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
            "GB10_RELEASE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
            "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
            "GB10_IMAGE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
            "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
            "GB10_PREBUILT_WHEEL_URLS": " ".join(
                f"https://github.com/gardner/flashinfer/releases/download/"
                f"{FLASHINFER_RELEASE_TAG}/{wheel}"
                for wheel in FLASHINFER_RELEASE_WHEELS
            ),
            "GB10_FLASH_ATTN_REPO": (
                "https://github.com/gardner/vllm-flash-attention.git"
            ),
            "GB10_FLASH_ATTN_REF": "main",
            "GB10_PUSH_IMAGE": "true",
            "GB10_PREFLIGHT_ONLY": "false",
            "GB10_NATIVE_CUDA_ARCHS_ONLY": "1",
            **GB10_RELEASE_CACHE_REF_ENV,
            "VLLM_USE_LOCAL_GB10_DEPS": "0",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10 release manifest validation failed" in proc.stdout
    assert "vLLM flash-attn ref must be a full Git SHA" in proc.stdout
    assert not output_path.exists()


def test_gb10_required_support_matrix_contract_is_shared():
    smoke = _load_gb10_smoke_module()
    manifest = _load_gb10_release_manifest_module()
    verifier = _load_gb10_release_evidence_module()
    bundler = _load_gb10_release_bundle_module()
    release_asset_validator = _load_gb10_release_asset_validator_module()

    assert manifest.REQUIRED_GB10_SUPPORT_MATRIX == GB10_REQUIRED_SUPPORT_MATRIX
    assert verifier.REQUIRED_GB10_SUPPORT_MATRIX == GB10_REQUIRED_SUPPORT_MATRIX
    assert bundler.REQUIRED_GB10_SUPPORT_MATRIX == GB10_REQUIRED_SUPPORT_MATRIX
    assert (
        release_asset_validator.REQUIRED_GB10_SUPPORT_MATRIX
        == GB10_REQUIRED_SUPPORT_MATRIX
    )
    required_not_supported_entries = {
        name
        for name, status in GB10_REQUIRED_SUPPORT_MATRIX.items()
        if status == "not_supported"
    }
    required_deferred_entries = {
        name
        for name, status in GB10_REQUIRED_SUPPORT_MATRIX.items()
        if status == "deferred"
    }
    required_supported_routed_entries = {
        name
        for name, status in GB10_REQUIRED_SUPPORT_MATRIX.items()
        if status == "supported_routed"
    }
    assert set(smoke.GB10_NOT_SUPPORTED_PATH_REASONS) == required_not_supported_entries
    assert set(smoke.GB10_DEFERRED_PATH_REASONS) == required_deferred_entries
    assert set(smoke.GB10_SUPPORTED_ROUTED_PATH_REASONS) == (
        required_supported_routed_entries
    )
    assert (
        verifier.GB10_NOT_SUPPORTED_PATH_REASONS
        == smoke.GB10_NOT_SUPPORTED_PATH_REASONS
    )
    assert verifier.GB10_DEFERRED_PATH_REASONS == smoke.GB10_DEFERRED_PATH_REASONS
    assert (
        verifier.GB10_SUPPORTED_ROUTED_PATH_REASONS
        == smoke.GB10_SUPPORTED_ROUTED_PATH_REASONS
    )


def test_gb10_release_evidence_classifies_flashinfer_cutlass_moe_selection():
    verifier = _load_gb10_release_evidence_module()

    assert verifier._support_matrix_entry_for_selection(
        {
            "path": "linear",
            "backend": "FlashInferB12xNvFp4LinearKernel",
            "is_fallback": False,
        }
    ) == "flashinfer_b12x_nvfp4_dense"
    assert verifier._support_matrix_entry_for_selection(
        {
            "path": "linear",
            "backend": "FlashInferCutlassNvFp4LinearKernel",
            "is_fallback": False,
        }
    ) == "flashinfer_cutlass_nvfp4_dense"
    assert verifier._support_matrix_entry_for_selection(
        {
            "path": "linear",
            "backend": "FlashInferTrtllmNvFp4LinearKernel",
            "is_fallback": False,
        }
    ) == "flashinfer_trtllm_nvfp4_dense"
    assert verifier._support_matrix_entry_for_selection(
        {
            "path": "linear",
            "backend": "FlashInferCudnnNvFp4LinearKernel",
            "is_fallback": False,
        }
    ) == "flashinfer_cudnn_nvfp4_dense"
    assert verifier._support_matrix_entry_for_selection(
        {
            "path": "moe",
            "backend": "FLASHINFER_CUTLASS",
            "is_fallback": False,
        }
    ) == "flashinfer_cutlass_non_ep_moe"


def test_gb10_release_provenance_contracts_are_shared():
    smoke = _load_gb10_smoke_module()
    manifest = _load_gb10_release_manifest_module()
    verifier = _load_gb10_release_evidence_module()

    assert manifest.REQUIRED_SOURCE_DEPENDENCIES == (
        GB10_REQUIRED_SOURCE_DEPENDENCIES
    )
    assert verifier.REQUIRED_SOURCE_DEPENDENCIES == (
        GB10_REQUIRED_SOURCE_DEPENDENCIES
    )
    assert smoke.FLASHINFER_RUNTIME_DISTRIBUTIONS == (
        GB10_FLASHINFER_RUNTIME_DISTRIBUTIONS
    )
    assert verifier.FLASHINFER_RUNTIME_DISTRIBUTIONS == (
        GB10_FLASHINFER_RUNTIME_DISTRIBUTIONS
    )


def test_gb10_evidence_upload_support_matrix_contract_is_shared():
    smoke_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-smoke-release-image.yml"
    ).read_text()
    release_asset_validator = _load_gb10_release_asset_validator_module()

    assert "scripts/gb10-validate-evidence-release-assets.py" in smoke_workflow
    assert "--gb10-output-dir" in smoke_workflow
    assert "--gb10-image-ref" in smoke_workflow
    assert "--gb10-image-digest" in smoke_workflow
    assert "--gb10-release-tag" in smoke_workflow
    assert "required_support_matrix = {" not in smoke_workflow
    assert '"flashinfer_nvfp4_dense": "supported_native"' not in smoke_workflow
    assert (
        release_asset_validator.REQUIRED_GB10_SUPPORT_MATRIX
        == GB10_REQUIRED_SUPPORT_MATRIX
    )


def test_gb10_release_manifest_validates_durable_inputs(tmp_path):
    manifest = _load_gb10_release_manifest_module()
    env = {
        "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        "GB10_RELEASE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
        "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
        "GB10_IMAGE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
        "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
        "GB10_PREBUILT_WHEEL_URLS": " ".join(
            f"https://github.com/gardner/flashinfer/releases/download/"
            f"{FLASHINFER_RELEASE_TAG}/{wheel}"
            for wheel in FLASHINFER_RELEASE_WHEELS
        ),
        "GB10_FLASH_ATTN_REPO": "https://github.com/gardner/vllm-flash-attention.git",
        "GB10_FLASH_ATTN_REF": VLLM_FLASH_ATTN_GIT_TAG,
        "GB10_PUSH_IMAGE": "true",
        "GB10_PREFLIGHT_ONLY": "false",
        "GB10_NATIVE_CUDA_ARCHS_ONLY": "1",
        **GB10_RELEASE_CACHE_REF_ENV,
        "VLLM_USE_LOCAL_GB10_DEPS": "0",
    }

    good_manifest = manifest.write_manifest(
        tmp_path / "gb10-release-manifest.json",
        env=env,
    )
    assert manifest.validate_manifest(good_manifest) == []

    bad_manifest = copy.deepcopy(good_manifest)
    bad_manifest["dependencies"]["vllm_flash_attn"]["ref"] = "main"
    bad_manifest["dependencies"]["vllm_flash_attn"]["ref_is_full_git_sha"] = False
    bad_manifest["dependencies"]["vllm_flash_attn"]["repository"] = "../flash-attn"
    bad_manifest["dependencies"]["source_dependencies"]["flashmla"][
        "ref_is_full_git_sha"
    ] = False
    bad_manifest["dependencies"]["source_dependencies"]["deepgemm"][
        "repository"
    ] = "file:///mnt/dgx-ssd/src/GB10/DeepGEMM"
    bad_manifest["build"]["local_gb10_dependency_checkouts"] = True
    bad_manifest["build"]["native_cuda_archs_only"] = False
    bad_manifest["dependencies"]["flashinfer"]["wheels"][0]["release_tag"] = None
    bad_manifest["dependencies"]["flashinfer"]["wheels"][0]["url"] = (
        "https://github.com/gardner/flashinfer/raw/main/"
        "flashinfer_python-0.6.12+cu130gb10-py3-none-any.whl"
    )
    bad_manifest["image"]["push"] = False
    bad_manifest["image"]["name"] = "docker.io/gardner/vllm-gb10"
    bad_manifest["image"]["tag"] = "latest"
    bad_manifest["vllm"]["version"] = "v0.22.1rc0-abcdef123"

    errors = manifest.validate_manifest(bad_manifest)

    assert any("vLLM flash-attn ref must be a full Git SHA" in err for err in errors)
    assert any(
        "vLLM flash-attn repository must be a GitHub HTTPS URL" in err
        for err in errors
    )
    assert any("FlashMLA ref must be a full Git SHA" in err for err in errors)
    assert any(
        "DeepGEMM repository must be a GitHub HTTPS URL" in err for err in errors
    )
    assert any(
        "GB10 local dependency checkouts are not allowed" in err for err in errors
    )
    assert any("native_cuda_archs_only=true" in err for err in errors)
    assert any(
        "FlashInfer wheel must come from a GitHub Release" in err for err in errors
    )
    assert any(
        "vLLM release manifest vllm.version must be a PEP 440 GB10 local version"
        in err
        for err in errors
    )
    assert any("tagged full release requires image.push=true" in err for err in errors)
    assert any(
        "GB10 tagged full release image.name must be a GHCR image" in err
        for err in errors
    )
    assert any(
        "GB10 tagged full release image.tag must match release.tag" in err
        for err in errors
    )

    missing_source_dependency_manifest = copy.deepcopy(good_manifest)
    del missing_source_dependency_manifest["dependencies"]["source_dependencies"][
        "flashmla"
    ]

    errors = manifest.validate_manifest(missing_source_dependency_manifest)

    assert any(
        "GB10 release manifest source_dependencies must include" in err
        and "flashmla" in err
        for err in errors
    )

    missing_support_entry_manifest = copy.deepcopy(good_manifest)
    del missing_support_entry_manifest["gb10_support_matrix"]["entries"][
        "public_flashattention_runtime"
    ]

    errors = manifest.validate_manifest(missing_support_entry_manifest)

    assert any(
        "GB10 release manifest support matrix must include" in err
        and "public_flashattention_runtime" in err
        for err in errors
    )

    wrong_support_status_manifest = copy.deepcopy(good_manifest)
    wrong_support_status_manifest["gb10_support_matrix"]["entries"][
        "trtllm_gen_attention"
    ]["status"] = "supported_native"

    errors = manifest.validate_manifest(wrong_support_status_manifest)

    assert any(
        "GB10 release manifest support matrix status mismatch" in err
        and "trtllm_gen_attention" in err
        and "supported_native" in err
        for err in errors
    )

    for malformed_image_name in (
        "ghcr.io/gardner/vllm-gb10:latest",
        "ghcr.io/gardner/vllm-gb10@sha256:" + "a" * 64,
        "ghcr.io/gardner/",
        "ghcr.io/Gardner/vllm-gb10",
        "ghcr.io/gardner/vllm+gb10",
    ):
        malformed_manifest = copy.deepcopy(good_manifest)
        malformed_manifest["image"]["name"] = malformed_image_name

        errors = manifest.validate_manifest(malformed_manifest)

        assert any(
            "GB10 tagged full release image.name must be a GHCR repository name"
            in err
            for err in errors
        )

    for malformed_image_tag in ("-bad", "bad/tag", "x" * 129):
        malformed_manifest = copy.deepcopy(good_manifest)
        malformed_manifest["release"]["tag"] = malformed_image_tag
        malformed_manifest["image"]["tag"] = malformed_image_tag

        errors = manifest.validate_manifest(malformed_manifest)

        assert any(
            "GB10 tagged full release image.tag must be a Docker-compatible tag"
            in err
            for err in errors
        )

    for cache_name, malformed_cache_ref in (
        ("preflight", "docker.io/gardner/vllm-gb10-buildcache:preflight"),
        ("wheel", "ghcr.io/gardner/vllm-gb10-buildcache"),
        ("runtime", "ghcr.io/Gardner/vllm-gb10-buildcache:runtime"),
    ):
        malformed_manifest = copy.deepcopy(good_manifest)
        malformed_manifest["build"]["cache_refs"][cache_name] = malformed_cache_ref

        errors = manifest.validate_manifest(malformed_manifest)

        assert any(
            f"GB10 release manifest build.cache_refs.{cache_name} "
            "must be a GHCR image ref"
            in err
            for err in errors
        )


def test_gb10_release_manifest_rejects_mixed_flashinfer_release_sets(tmp_path):
    manifest = _load_gb10_release_manifest_module()
    env = {
        "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
        "GB10_IMAGE_TAG": "gb10-abcdef123456",
        "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
        "GB10_PREBUILT_WHEEL_URLS": " ".join(
            f"https://github.com/gardner/flashinfer/releases/download/"
            f"{FLASHINFER_RELEASE_TAG}/{wheel}"
            for wheel in FLASHINFER_RELEASE_WHEELS
        ),
        "GB10_FLASH_ATTN_REPO": "https://github.com/gardner/vllm-flash-attention.git",
        "GB10_FLASH_ATTN_REF": VLLM_FLASH_ATTN_GIT_TAG,
        "GB10_PREFLIGHT_ONLY": "true",
        "GB10_NATIVE_CUDA_ARCHS_ONLY": "1",
        **GB10_RELEASE_CACHE_REF_ENV,
        "VLLM_USE_LOCAL_GB10_DEPS": "0",
    }

    good_manifest = manifest.write_manifest(
        tmp_path / "gb10-release-manifest.json",
        env=env,
    )
    assert manifest.validate_manifest(good_manifest) == []

    mixed_manifest = copy.deepcopy(good_manifest)
    mixed_manifest["dependencies"]["flashinfer"]["wheels"][1]["url"] = (
        "https://github.com/gardner/flashinfer/releases/download/"
        "gb10-flashinfer-v0.6.12-deadbeef/"
        "flashinfer_cubin-0.6.12+cu130gb10-py3-none-any.whl"
    )
    mixed_manifest["dependencies"]["flashinfer"]["wheels"][2]["url"] = (
        "https://github.com/other/flashinfer/releases/download/"
        f"{FLASHINFER_RELEASE_TAG}/"
        "flashinfer_jit_cache-0.6.12+cu130gb10-cp39-abi3-manylinux_2_28_aarch64.whl"
    )

    errors = manifest.validate_manifest(mixed_manifest)

    assert any(
        "FlashInfer GB10 wheels must come from one GitHub Release" in err
        for err in errors
    )


def test_gb10_release_manifest_rejects_duplicate_or_extra_flashinfer_wheels(
    tmp_path,
):
    manifest = _load_gb10_release_manifest_module()
    env = {
        "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
        "GB10_IMAGE_TAG": "gb10-abcdef123456",
        "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
        "GB10_PREBUILT_WHEEL_URLS": " ".join(
            f"https://github.com/gardner/flashinfer/releases/download/"
            f"{FLASHINFER_RELEASE_TAG}/{wheel}"
            for wheel in FLASHINFER_RELEASE_WHEELS
        ),
        "GB10_FLASH_ATTN_REPO": "https://github.com/gardner/vllm-flash-attention.git",
        "GB10_FLASH_ATTN_REF": VLLM_FLASH_ATTN_GIT_TAG,
        "GB10_PREFLIGHT_ONLY": "true",
        "GB10_NATIVE_CUDA_ARCHS_ONLY": "1",
        **GB10_RELEASE_CACHE_REF_ENV,
        "VLLM_USE_LOCAL_GB10_DEPS": "0",
    }

    good_manifest = manifest.write_manifest(
        tmp_path / "gb10-release-manifest.json",
        env=env,
    )
    assert manifest.validate_manifest(good_manifest) == []

    duplicate_manifest = copy.deepcopy(good_manifest)
    duplicate_manifest["dependencies"]["flashinfer"]["wheels"].append(
        copy.deepcopy(duplicate_manifest["dependencies"]["flashinfer"]["wheels"][0])
    )
    duplicate_errors = manifest.validate_manifest(duplicate_manifest)

    assert any(
        "FlashInfer GB10 wheel URLs must include exactly one" in err
        for err in duplicate_errors
    )

    url_duplicate_manifest = copy.deepcopy(good_manifest)
    url_duplicate_manifest["dependencies"]["flashinfer"]["wheels"][1]["url"] = (
        "https://github.com/gardner/flashinfer/releases/download/"
        f"{FLASHINFER_RELEASE_TAG}/"
        "flashinfer_python-0.6.12+cu130gb10-py3-none-any.whl"
    )
    url_duplicate_errors = manifest.validate_manifest(url_duplicate_manifest)

    assert any(
        "FlashInfer wheel component metadata does not match URL" in err
        for err in url_duplicate_errors
    )
    assert any(
        "FlashInfer GB10 wheel URLs must include exactly one" in err
        for err in url_duplicate_errors
    )

    extra_manifest = copy.deepcopy(good_manifest)
    extra_manifest["dependencies"]["flashinfer"]["wheels"].append(
        {
            "component": "flashinfer_debug",
            "filename": "flashinfer_debug-0.6.12+cu130gb10-py3-none-any.whl",
            "release_repository": "gardner/flashinfer",
            "release_tag": FLASHINFER_RELEASE_TAG,
            "url": (
                "https://github.com/gardner/flashinfer/releases/download/"
                f"{FLASHINFER_RELEASE_TAG}/"
                "flashinfer_debug-0.6.12+cu130gb10-py3-none-any.whl"
            ),
        }
    )
    extra_errors = manifest.validate_manifest(extra_manifest)

    assert any(
        "FlashInfer release manifest has unexpected wheel components" in err
        for err in extra_errors
    )


def test_gb10_release_manifest_rejects_non_gb10_flashinfer_wheels(tmp_path):
    manifest = _load_gb10_release_manifest_module()
    env = {
        "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
        "GB10_IMAGE_TAG": "gb10-abcdef123456",
        "GB10_VLLM_VERSION": "0.22.1rc0+gb10.abcdef123456",
        "GB10_PREBUILT_WHEEL_URLS": " ".join(
            [
                (
                    "https://github.com/gardner/flashinfer/releases/download/"
                    f"{FLASHINFER_RELEASE_TAG}/"
                    "flashinfer_python-0.6.12+cu130-py3-none-any.whl"
                ),
                (
                    "https://github.com/gardner/flashinfer/releases/download/"
                    f"{FLASHINFER_RELEASE_TAG}/"
                    "flashinfer_cubin-0.6.12+cu130-py3-none-any.whl"
                ),
                (
                    "https://github.com/gardner/flashinfer/releases/download/"
                    f"{FLASHINFER_RELEASE_TAG}/"
                    "flashinfer_jit_cache-0.6.12+cu130-cp39-abi3-"
                    "manylinux_2_28_aarch64.whl"
                ),
            ]
        ),
        "GB10_FLASH_ATTN_REPO": "https://github.com/gardner/vllm-flash-attention.git",
        "GB10_FLASH_ATTN_REF": VLLM_FLASH_ATTN_GIT_TAG,
        "GB10_PREFLIGHT_ONLY": "true",
        **GB10_RELEASE_CACHE_REF_ENV,
        "VLLM_USE_LOCAL_GB10_DEPS": "0",
    }

    non_gb10_manifest = manifest.write_manifest(
        tmp_path / "gb10-release-manifest.json",
        env=env,
    )

    errors = manifest.validate_manifest(non_gb10_manifest)

    assert any(
        "FlashInfer wheel filename must include a GB10 CUDA local version" in err
        for err in errors
    )


def test_gb10_image_smoke_workflow_publishes_durable_evidence():
    smoke_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-smoke-release-image.yml"
    ).read_text()

    assert "workflow_dispatch:" in smoke_workflow
    assert "image-ref:" in smoke_workflow
    assert "release-workflow-run-id:" in smoke_workflow
    assert "release-tag:" in smoke_workflow
    assert "publish-to-release:" in smoke_workflow
    assert "provenance-only:" in smoke_workflow
    assert "model:" in smoke_workflow
    assert "served-model-name:" in smoke_workflow
    assert "require-moe:" in smoke_workflow
    assert "require-openai-deterministic:" in smoke_workflow
    assert "allow-existing-vllm-containers:" in smoke_workflow
    assert "runs-on: [self-hosted, linux, aarch64, cuda13, dgx-spark, sm121]" in (
        smoke_workflow
    )
    assert "${{ runner.temp }}" not in smoke_workflow
    assert (
        "GB10_SMOKE_CACHE_DIR: "
        "${{ github.workspace }}/.gb10-hf-cache"
        in smoke_workflow
    )
    assert (
        "GB10_OPENAI_IMAGE_CACHE_DIR: "
        "${{ github.workspace }}/.gb10-hf-cache"
        in smoke_workflow
    )
    assert "permissions:" in smoke_workflow
    assert "contents: write" in smoke_workflow
    assert "packages: read" in smoke_workflow
    assert "actions: read" in smoke_workflow
    assert "Refuse concurrent vLLM service" in smoke_workflow
    assert "GB10_RELEASE_ALLOW_EXISTING_VLLM_CONTAINERS" in smoke_workflow
    assert "GB10_PROVENANCE_ONLY" in smoke_workflow
    assert "pre_smoke_resource_guard" in smoke_workflow
    assert (
        "inputs['allow-existing-vllm-containers'] != true && "
        "inputs['provenance-only'] != true"
        in smoke_workflow
    )
    assert smoke_workflow.index("Refuse concurrent vLLM service") < (
        smoke_workflow.index("Log in to GHCR")
    )
    assert smoke_workflow.index("Refuse concurrent vLLM service") < (
        smoke_workflow.index("Pull candidate image")
    )
    assert "docker/login-action@v4" in smoke_workflow
    assert smoke_workflow.index("Download release provenance artifact") < (
        smoke_workflow.index("Pull candidate image")
    )
    assert "docker pull \"$GB10_IMAGE_REF\"" in smoke_workflow
    assert "docker image inspect \"$GB10_IMAGE_REF\"" in smoke_workflow
    assert "GB10_IMAGE_DIGEST=$image_digest" in smoke_workflow
    assert "GB10_EXPECTED_IMAGE_DIGEST=$expected_digest" in smoke_workflow
    assert "gb10-provenance-check.json" in smoke_workflow
    assert "write_provenance_check_report" in smoke_workflow
    assert '"phase": phase' in smoke_workflow
    assert '"message": message' in smoke_workflow
    assert "GB10 release manifest image ref does not match input image-ref" in (
        smoke_workflow
    )
    assert "GB10 runtime image ref file does not match input image-ref" in (
        smoke_workflow
    )
    assert "GB10 pulled image digest does not match release provenance" in (
        smoke_workflow
    )
    assert '[[ ! "${expected_digest##*@}" =~ ^sha256:[0-9a-f]{64}$ ]]' in (
        smoke_workflow
    )
    assert "GB10 runtime image digest file is not a valid sha256 digest" in (
        smoke_workflow
    )
    assert '[[ ! "${image_digest##*@}" =~ ^sha256:[0-9a-f]{64}$ ]]' in (
        smoke_workflow
    )
    assert "GB10 pulled image digest is not a valid sha256 digest" in (
        smoke_workflow
    )
    assert "GB10 candidate image pull failed" in smoke_workflow
    assert "Verify candidate image GB10 packages" in smoke_workflow
    assert "scripts/gb10-verify-image-packages.py" in smoke_workflow
    assert "gb10-image-package-check.json" in smoke_workflow
    assert "docker run --rm" in smoke_workflow
    assert "--network none" in smoke_workflow
    assert smoke_workflow.index("Pull candidate image") < smoke_workflow.index(
        "Verify candidate image GB10 packages"
    )
    assert smoke_workflow.index("Verify candidate image GB10 packages") < (
        smoke_workflow.index("Record provenance-only evidence")
    )
    assert smoke_workflow.index("Verify candidate image GB10 packages") < (
        smoke_workflow.index("Run final-image smoke and evidence verifier")
    )
    report_lister_script = (
        REPO_ROOT / "scripts" / "gb10-list-release-evidence-report-files.py"
    ).read_text()
    assert "scripts/gb10-list-release-evidence-report-files.py" in smoke_workflow
    assert "--gb10-report-dir \"$GB10_RELEASE_SMOKE_REPORT_DIR\"" in smoke_workflow
    assert "mapfile -t release_evidence_files" in smoke_workflow
    assert '"${#release_evidence_files[@]}" -ne 4' in smoke_workflow
    assert 'smoked_image_digest_report="${release_evidence_files[3]}"' in (
        smoke_workflow
    )
    assert (
        '> "$GB10_RELEASE_SMOKE_REPORT_DIR/gb10-smoked-image-digest.txt"'
        not in smoke_workflow
    )
    assert "release_evidence_file_paths" in report_lister_script
    assert "scripts/gb10-smoke-release-image.sh \"$GB10_IMAGE_REF\"" in (
        smoke_workflow
    )
    assert "--gb10-require-path linear" in smoke_workflow
    assert "--gb10-expect-backend linear=FlashInferB12x" in smoke_workflow
    assert "--gb10-require-path moe" in smoke_workflow
    assert "--gb10-expect-backend moe=FLASHINFER_B12X" in smoke_workflow
    assert "--quantization modelopt" in smoke_workflow
    assert "--attention-backend flashinfer" in smoke_workflow
    assert "--kv-cache-dtype fp8" in smoke_workflow
    assert "GB10_RELEASE_SMOKE_REPORT_DIR" in smoke_workflow
    assert "GB10_RELEASE_EVIDENCE_OUTPUT_DIR" in smoke_workflow
    assert "GB10_RELEASE_WORKFLOW_RUN_ID" in smoke_workflow
    assert "GB10_RELEASE_PROVENANCE_DIR" in smoke_workflow
    assert "Download release provenance artifact" in smoke_workflow
    assert "gh run download \"$GB10_RELEASE_WORKFLOW_RUN_ID\"" in smoke_workflow
    assert '--name "$GB10_RELEASE_MANIFEST_ARTIFACT_NAME"' in smoke_workflow
    assert "--dir \"$GB10_RELEASE_PROVENANCE_DIR\"" in smoke_workflow
    provenance_lister_script = (
        REPO_ROOT / "scripts" / "gb10-list-release-provenance-artifact-files.py"
    ).read_text()
    assert "scripts/gb10-list-release-provenance-artifact-files.py" in (
        smoke_workflow
    )
    assert "--gb10-provenance-dir \"$GB10_RELEASE_PROVENANCE_DIR\"" in (
        smoke_workflow
    )
    assert "release_provenance_artifact_paths" in provenance_lister_script
    assert "RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS" in provenance_lister_script
    assert "GB10_RELEASE_MANIFEST_JSON=$manifest" in smoke_workflow
    assert "GB10_RUNTIME_IMAGE_METADATA_JSON=$runtime_metadata" in smoke_workflow
    assert "GB10 release manifest artifact is missing" in smoke_workflow
    assert "GB10 runtime image metadata artifact is missing" in smoke_workflow
    assert "GB10 runtime image ref artifact is missing" in smoke_workflow
    assert "GB10 runtime image digest artifact is missing" in smoke_workflow
    assert "GB10 runtime image ref file is empty" in smoke_workflow
    assert "GB10 runtime image digest file is empty" in smoke_workflow
    assert "Require release publication provenance" in smoke_workflow
    assert "GB10 release evidence publication requires release-workflow-run-id" in (
        smoke_workflow
    )
    assert "GB10 release evidence publication requires release-tag" in smoke_workflow
    assert "Reject provenance-only release publication" in smoke_workflow
    assert "GB10 provenance-only evidence cannot be published as release evidence" in (
        smoke_workflow
    )
    assert "Require provenance-only release workflow run" in smoke_workflow
    assert "GB10 provenance-only smoke requires release-workflow-run-id" in (
        smoke_workflow
    )
    assert "Require release tag provenance" in smoke_workflow
    assert "GB10 release-tag smoke requires release-workflow-run-id" in (
        smoke_workflow
    )
    assert smoke_workflow.index("Reject provenance-only release publication") < (
        smoke_workflow.index("Refuse concurrent vLLM service")
    )
    assert smoke_workflow.index("Require release publication provenance") < (
        smoke_workflow.index("Refuse concurrent vLLM service")
    )
    assert smoke_workflow.index("Require release publication tag") < (
        smoke_workflow.index("Refuse concurrent vLLM service")
    )
    assert smoke_workflow.index("Require release publication tag") < (
        smoke_workflow.index("Log in to GHCR")
    )
    assert smoke_workflow.index("Require provenance-only release workflow run") < (
        smoke_workflow.index("Refuse concurrent vLLM service")
    )
    assert smoke_workflow.index("Require provenance-only release workflow run") < (
        smoke_workflow.index("Log in to GHCR")
    )
    assert smoke_workflow.index("Require provenance-only release workflow run") < (
        smoke_workflow.index("Download release provenance artifact")
    )
    assert smoke_workflow.index("Require release tag provenance") < (
        smoke_workflow.index("Refuse concurrent vLLM service")
    )
    assert smoke_workflow.index("Require release tag provenance") < (
        smoke_workflow.index("Log in to GHCR")
    )
    assert smoke_workflow.index("Require release tag provenance") < (
        smoke_workflow.index("Download release provenance artifact")
    )
    assert smoke_workflow.index("Require release tag provenance") < (
        smoke_workflow.index("Pull candidate image")
    )
    missing_run_id_guard = (
        "inputs['publish-to-release'] && "
        "inputs['release-workflow-run-id'] == ''"
    )
    assert missing_run_id_guard in smoke_workflow
    assert "inputs['publish-to-release'] && inputs['release-tag'] == ''" in (
        smoke_workflow
    )
    assert "inputs['provenance-only'] && inputs['release-workflow-run-id'] == ''" in (
        smoke_workflow
    )
    assert "inputs['release-tag'] != '' && inputs['release-workflow-run-id'] == ''" in (
        smoke_workflow
    )
    assert "inputs['release-workflow-run-id'] != ''" in smoke_workflow
    assert "Record provenance-only evidence" in smoke_workflow
    assert "phase\": \"provenance_only\"" in smoke_workflow
    assert "final-image model smoke was skipped" in smoke_workflow
    assert "inputs['provenance-only'] == true" in smoke_workflow
    assert "inputs['provenance-only'] != true" in smoke_workflow
    assert "Bundle available evidence after failure" in smoke_workflow
    assert "scripts/gb10-bundle-release-evidence.py" in smoke_workflow
    assert "--gb10-allow-partial" in smoke_workflow
    manifest_arg = (
        'bundle_args+=(--gb10-release-manifest-json '
        '"$GB10_RELEASE_MANIFEST_JSON")'
    )
    metadata_arg = (
        'bundle_args+=(--gb10-runtime-image-metadata-json '
        '"$GB10_RUNTIME_IMAGE_METADATA_JSON")'
    )
    assert manifest_arg in smoke_workflow
    assert metadata_arg in smoke_workflow
    assert '"${bundle_args[@]}" || true' in smoke_workflow
    assert "actions/upload-artifact@v7" in smoke_workflow
    assert "name: ${{ env.GB10_RELEASE_EVIDENCE_ARTIFACT_NAME }}" in smoke_workflow
    assert "${{ env.GB10_RELEASE_SMOKE_REPORT_DIR }}/**" in smoke_workflow
    assert "${{ env.GB10_RELEASE_PROVENANCE_DIR }}/**" in smoke_workflow
    assert "${{ env.GB10_RELEASE_EVIDENCE_OUTPUT_DIR }}/**" in smoke_workflow
    assert "gb10-smoke-reports/**" not in smoke_workflow
    assert "gb10-release-provenance/**" not in smoke_workflow
    assert "dist/gb10-release-evidence/**" not in smoke_workflow
    assert "Validate GB10 evidence release assets" in smoke_workflow
    assert "id: validate_evidence_release_assets" in smoke_workflow
    validator_script = (
        REPO_ROOT / "scripts" / "gb10-validate-evidence-release-assets.py"
    ).read_text()
    lister_script = (
        REPO_ROOT / "scripts" / "gb10-list-evidence-release-assets.py"
    ).read_text()
    assert "scripts/gb10-list-evidence-release-assets.py" in smoke_workflow
    assert "GB10 evidence release upload asset is missing or empty" in lister_script
    assert "GB10 evidence release asset is missing or empty" in validator_script
    assert "sha256sum --check" not in smoke_workflow
    assert "--gb10-output-dir \"$GB10_RELEASE_EVIDENCE_OUTPUT_DIR\"" in (
        smoke_workflow
    )
    assert "scripts/gb10-validate-evidence-release-assets.py" in smoke_workflow
    assert "--gb10-metadata-json" not in smoke_workflow
    assert "release-evidence-metadata.json" not in smoke_workflow
    assert '--gb10-image-ref "$GB10_IMAGE_REF"' in smoke_workflow
    assert '--gb10-image-digest "$GB10_IMAGE_DIGEST"' in smoke_workflow
    assert '--gb10-release-tag "$GB10_RELEASE_TAG"' in smoke_workflow
    assert "does not prove a passed release" in validator_script
    assert "status={metadata.get('status')}" in validator_script
    assert 'metadata.get("status") != "complete"' in validator_script
    assert 'metadata.get("release_gate_passed") is not True' in validator_script
    image_digest_mismatch = (
        "GB10 evidence release metadata image digest does not match pulled digest"
    )
    assert image_digest_mismatch in validator_script
    image_ref_mismatch = (
        "GB10 evidence release metadata image ref does not match input image-ref"
    )
    assert image_ref_mismatch in validator_script
    assert "GB10 evidence release metadata tag does not match release-tag" in (
        validator_script
    )
    assert (
        "GB10 evidence release metadata is missing a GB10 support matrix summary"
        in validator_script
    )
    assert (
        "GB10 evidence release metadata does not prove a complete GB10 support matrix"
        in validator_script
    )
    assert 'metadata.get("support_matrix_complete") is not True' in validator_script
    assert 'support_matrix = metadata.get("support_matrix_summary")' in (
        validator_script
    )
    assert 'support_matrix.get("present") is not True' in validator_script
    assert (
        'support_matrix.get("release_manifest_present") is not True'
        in validator_script
    )
    assert 'support_matrix.get("architecture") != "sm_121a"' in validator_script
    assert (
        'support_matrix.get("first_release_scope") != "single_spark_first_path"'
        in validator_script
    )
    assert 'support_matrix.get("entry_count", 0) <= 0' in validator_script
    assert 'support_matrix.get("invalid_entries")' in validator_script
    assert "GB10 evidence release metadata support matrix does not match" in (
        validator_script
    )
    assert "from gb10_release_contract import" in validator_script
    assert "REQUIRED_GB10_SUPPORT_MATRIX" in validator_script
    assert '"flashinfer_nvfp4_dense": "supported_native"' not in validator_script
    assert 'entries = support_matrix.get("entries")' in validator_script
    assert "def _mismatched_support_entries" in validator_script
    assert validator_script.index(
        'support_matrix = metadata.get("support_matrix_summary")'
    ) < validator_script.index(
        'metadata.get("status") != "complete"'
    )
    assert "GB10 evidence release metadata is missing release provenance" in (
        validator_script
    )
    assert 'source = metadata.get("source")' in validator_script
    assert 'source.get("image_ref") != image_ref' in validator_script
    assert (
        'normalize_image_digest(source.get("image_digest"))'
        in validator_script
    )
    assert "pulled_digest = normalize_image_digest(image_digest)" in validator_script
    assert "SHA256_DIGEST_RE.fullmatch" in validator_script
    assert 'source.get("release_tag") != release_tag' in validator_script
    assert "REQUIRED_RELEASE_EVIDENCE_PROVENANCE" in validator_script
    assert "Attach evidence to GitHub Release" in smoke_workflow
    assert (
        "steps.validate_evidence_release_assets.outcome == 'success'"
        in smoke_workflow
    )
    assert "GB10 release tag does not exist; run the release workflow first" in (
        smoke_workflow
    )
    assert "gh release view \"$GB10_RELEASE_TAG\"" in smoke_workflow
    assert "gh release create" not in smoke_workflow
    assert "gh release upload \"$GB10_RELEASE_TAG\"" in smoke_workflow
    assert '"${evidence_assets[@]}"' in smoke_workflow
    assert "release_evidence_asset_paths" in lister_script


def test_gb10_workflows_use_node24_action_versions():
    release_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    smoke_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-smoke-release-image.yml"
    ).read_text()

    assert "actions/upload-artifact@v7" in release_workflow
    assert "docker/login-action@v4" in release_workflow
    assert "docker/setup-buildx-action@v4" in release_workflow

    assert "actions/upload-artifact@v7" in smoke_workflow
    assert "docker/login-action@v4" in smoke_workflow

    for workflow_text in (release_workflow, smoke_workflow):
        assert "actions/upload-artifact@v5" not in workflow_text
        assert "actions/upload-artifact@v4" not in workflow_text
        assert "docker/login-action@v3" not in workflow_text
        assert "docker/setup-buildx-action@v3" not in workflow_text


def test_gb10_evidence_release_asset_validator_accepts_complete_metadata():
    validator = _load_gb10_release_asset_validator_module()
    digest = "sha256:" + "a" * 64
    metadata = {
        "status": "complete",
        "release_gate_passed": True,
        "source": {
            "image_ref": "ghcr.io/gardner/vllm-gb10:gb10-test",
            "image_digest": f"ghcr.io/gardner/vllm-gb10@{digest}",
            "release_tag": "gb10-test",
        },
        "support_matrix_complete": True,
        "support_matrix_summary": {
            "present": True,
            "release_manifest_present": True,
            "architecture": "sm_121a",
            "first_release_scope": "single_spark_first_path",
            "entry_count": len(GB10_REQUIRED_SUPPORT_MATRIX),
            "invalid_entries": [],
            "entries": dict(GB10_REQUIRED_SUPPORT_MATRIX),
        },
        "included_provenance": [
            {"kind": "release_manifest"},
            {"kind": "runtime_image_metadata"},
        ],
    }

    assert validator.validate_metadata(
        metadata,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-test",
        image_digest=digest,
        release_tag="gb10-test",
    ) == []


def test_gb10_evidence_release_asset_validator_rejects_incomplete_metadata():
    validator = _load_gb10_release_asset_validator_module()
    digest = "sha256:" + "a" * 64
    wrong_digest = "sha256:" + "b" * 64
    incomplete_support_entries = dict(GB10_REQUIRED_SUPPORT_MATRIX)
    del incomplete_support_entries["flashinfer_b12x_non_ep_moe"]
    metadata = {
        "status": "partial",
        "release_gate_passed": False,
        "source": {
            "image_ref": "ghcr.io/gardner/vllm-gb10:gb10-test",
            "image_digest": digest,
            "release_tag": "gb10-test",
        },
        "support_matrix_complete": False,
        "support_matrix_summary": {
            "present": True,
            "release_manifest_present": True,
            "architecture": "sm_121a",
            "first_release_scope": "single_spark_first_path",
            "entry_count": len(incomplete_support_entries),
            "invalid_entries": [],
            "entries": incomplete_support_entries,
        },
        "included_provenance": [{"kind": "release_manifest"}],
    }

    errors = validator.validate_metadata(
        metadata,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-test",
        image_digest=wrong_digest,
        release_tag="gb10-test",
    )

    assert any("does not prove a complete GB10 support matrix" in err for err in errors)
    assert any("support matrix does not match" in err for err in errors)
    assert any("does not prove a passed release gate" in err for err in errors)
    assert any("image digest does not match pulled digest" in err for err in errors)
    assert any("missing release provenance" in err for err in errors)


def test_gb10_local_cached_build_script_defaults_to_serial_builds():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert "preflight|wheel|runtime" in script
    assert 'GB10_MAX_JOBS="${GB10_MAX_JOBS:-1}"' in script
    assert 'GB10_NVCC_THREADS="${GB10_NVCC_THREADS:-1}"' in script
    assert 'GB10_NATIVE_CUDA_ARCHS_ONLY="${GB10_NATIVE_CUDA_ARCHS_ONLY:-1}"' in script
    assert ".buildx-cache/gb10" in script
    assert 'GB10_USE_REGISTRY_CACHE="${GB10_USE_REGISTRY_CACHE:-0}"' in script
    assert 'GB10_BUILDX_BUILDER="${GB10_BUILDX_BUILDER:-gb10-builder}"' in script
    assert "--driver docker-container" in script
    assert '--cache-to "type=local,dest=$cache_next,mode=max"' in script
    assert "GB10_USE_REGISTRY_CACHE=1" in script
    assert 'scripts/gb10-run-with-heartbeat.sh "local ${cache_key} build"' in script
    assert "--builder \"$GB10_BUILDX_BUILDER\"" in script
    assert "--target \"$docker_target\"" in script
    assert (
        '--build-arg "vllm_native_cuda_archs_only=$GB10_NATIVE_CUDA_ARCHS_ONLY"'
        in script
    )


def test_gb10_local_cached_build_script_validates_manifest_before_buildx():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert 'GITHUB_SHA="${GITHUB_SHA:-$(git rev-parse HEAD)}"' in script
    assert "scripts/gb10-resolve-release-settings.py" in script
    assert "GB10_LOCAL_RELEASE_MANIFEST_DIR" in script
    assert "scripts/gb10-write-release-manifest.py" in script
    assert "--gb10-validate-release-inputs" in script
    assert "--gb10-output-json" in script
    assert "GB10_PREFLIGHT_ONLY" in script
    assert "GB10_PREFLIGHT_CACHE_REF" in script
    assert "GB10_WHEEL_CACHE_REF" in script
    assert "GB10_RUNTIME_CACHE_REF" in script
    assert script.index("--gb10-validate-release-inputs") < script.index(
        "docker buildx inspect \"$GB10_BUILDX_BUILDER\" --bootstrap"
    )
    assert script.index("scripts/gb10-resolve-release-settings.py") < script.index(
        "scripts/gb10-write-release-manifest.py"
    )
    assert FLASHINFER_RELEASE_TAG not in script


def test_gb10_local_cached_build_script_sets_image_metadata_args():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert 'VLLM_BUILD_COMMIT="${VLLM_BUILD_COMMIT:-$GITHUB_SHA}"' in script
    assert (
        'VLLM_BUILD_PIPELINE="${VLLM_BUILD_PIPELINE:-${GITHUB_WORKFLOW:-'
        "GB10 local cached build}}"
    ) in script
    assert 'VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-$GB10_IMAGE_TAG}"' in script
    assert '--build-arg "VLLM_BUILD_COMMIT=$VLLM_BUILD_COMMIT"' in script
    assert '--build-arg "VLLM_BUILD_PIPELINE=$VLLM_BUILD_PIPELINE"' in script
    assert '--build-arg "VLLM_BUILD_URL=$VLLM_BUILD_URL"' in script
    assert '--build-arg "VLLM_IMAGE_TAG=$VLLM_IMAGE_TAG"' in script


def test_gb10_local_cached_runtime_build_writes_image_provenance():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert "GB10_RUNTIME_IMAGE_METADATA_JSON" in script
    assert '--metadata-file "$GB10_RUNTIME_IMAGE_METADATA_JSON"' in script
    assert "scripts/gb10-write-runtime-image-provenance.py" in script
    assert '--gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON"' in (
        script
    )
    assert '--gb10-release-manifest-dir "$GB10_LOCAL_RELEASE_MANIFEST_DIR"' in script
    assert '--gb10-image-name "$GB10_IMAGE_NAME"' in script
    assert '--gb10-image-tag "$GB10_IMAGE_TAG"' in script
    assert '--gb10-push-image "$GB10_PUSH_IMAGE"' in script
    assert script.index('--metadata-file "$GB10_RUNTIME_IMAGE_METADATA_JSON"') < (
        script.index("build_status=$?")
    )
    assert script.index("build_status=$?") < script.index(
        "scripts/gb10-write-runtime-image-provenance.py"
    )


def test_gb10_local_cached_wheel_build_extracts_dist_artifact():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert "GB10_LOCAL_DIST_DIR" in script
    assert '[ "$docker_target" = "build" ]' in script
    assert '[ "$output_mode" = "load" ]' in script
    assert 'wheel_container="$(docker create vllm-gb10-wheel:local)"' in script
    assert 'docker cp "$wheel_container:/workspace/dist/." "$GB10_LOCAL_DIST_DIR/"' in (
        script
    )
    assert 'docker rm "$wheel_container"' in script
    assert "scripts/gb10-validate-vllm-wheel-artifact.py" in script
    assert '--gb10-context "GB10 local wheel build after artifact extraction"' in (
        script
    )
    assert '--gb10-vllm-version "$GB10_VLLM_VERSION"' in script
    assert script.index("build_status=$?") < script.index(
        'wheel_container="$(docker create vllm-gb10-wheel:local)"'
    )
    assert script.index('docker cp "$wheel_container:/workspace/dist/."') < (
        script.index(
            '--gb10-context "GB10 local wheel build after artifact extraction"'
        )
    )


def test_gb10_vllm_wheel_artifact_validator_enforces_exact_version(tmp_path):
    validator = REPO_ROOT / "scripts" / "gb10-validate-vllm-wheel-artifact.py"
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    expected_version = "0.22.1rc0+gb10.abcdef123456"

    def run_validator():
        return subprocess.run(
            [
                sys.executable,
                str(validator),
                "--gb10-dist-dir",
                str(dist_dir),
                "--gb10-vllm-version",
                expected_version,
                "--gb10-context",
                "GB10 local wheel build",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20,
        )

    missing_proc = run_validator()
    assert missing_proc.returncode != 0, missing_proc.stdout
    assert "requires exactly one vLLM wheel" in missing_proc.stdout

    stale_wheel = dist_dir / "vllm-0.22.1rc0+gb10.stale-cp38-abi3-linux_aarch64.whl"
    stale_wheel.write_bytes(b"stale")
    stale_proc = run_validator()
    assert stale_proc.returncode != 0, stale_proc.stdout
    assert "does not match GB10_VLLM_VERSION" in stale_proc.stdout

    stale_wheel.unlink()
    current_wheel = (
        dist_dir
        / "vllm-0.22.1rc0+gb10.abcdef123456-cp38-abi3-linux_aarch64.whl"
    )
    current_wheel.write_bytes(b"current")
    current_proc = run_validator()
    assert current_proc.returncode == 0, current_proc.stdout
    assert str(current_wheel) in current_proc.stdout

    (dist_dir / "vllm-0.22.1rc0+gb10.other-cp38-abi3-linux_aarch64.whl").write_bytes(
        b"other"
    )
    multiple_proc = run_validator()
    assert multiple_proc.returncode != 0, multiple_proc.stdout
    assert "requires exactly one vLLM wheel" in multiple_proc.stdout


def test_gb10_vllm_wheel_artifact_validator_allows_empty_when_requested(
    tmp_path,
):
    validator = REPO_ROOT / "scripts" / "gb10-validate-vllm-wheel-artifact.py"
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    empty_proc = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--gb10-dist-dir",
            str(dist_dir),
            "--gb10-vllm-version",
            "0.22.1rc0+gb10.abcdef123456",
            "--gb10-context",
            "GB10 local wheel build output directory before Docker/Buildx starts",
            "--gb10-allow-empty",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert empty_proc.returncode == 0, empty_proc.stdout
    assert "No local vLLM wheel artifacts found" in empty_proc.stdout


def test_gb10_local_cached_runtime_build_writes_release_checksums():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert "scripts/gb10-write-vllm-release-checksums.py" in script
    assert '--gb10-dist-dir "$GB10_LOCAL_DIST_DIR"' in script
    assert '--gb10-release-manifest-dir "$GB10_LOCAL_RELEASE_MANIFEST_DIR"' in script
    assert '--gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON"' in (
        script
    )
    assert "gb10-vllm-release-SHA256SUMS" in script
    assert "scripts/gb10-validate-vllm-release-assets.py" in script
    assert '[ -n "$GB10_RELEASE_TAG" ]' in script
    assert script.index("scripts/gb10-write-runtime-image-provenance.py") < (
        script.index("scripts/gb10-write-vllm-release-checksums.py")
    )
    assert script.index("scripts/gb10-write-vllm-release-checksums.py") < (
        script.index("scripts/gb10-validate-vllm-release-assets.py")
    )


def test_gb10_local_cached_runtime_cacheonly_skips_release_asset_writes():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    post_build_runtime_guard = (
        'if [ "$docker_target" = "vllm-openai" ] '
        '&& [ "$output_mode" != "cacheonly" ]; then'
    )

    assert script.rindex(post_build_runtime_guard) > script.index(
        'if [ "$build_status" -ne 0 ]; then'
    )
    assert script.rindex(post_build_runtime_guard) < script.index(
        "scripts/gb10-write-runtime-image-provenance.py"
    )
    assert script.index("scripts/gb10-write-runtime-image-provenance.py") < (
        script.rindex("fi")
    )


def test_gb10_local_cached_runtime_build_requires_wheel_before_docker():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert "scripts/gb10-validate-vllm-wheel-artifact.py" in script
    assert '--gb10-dist-dir "$GB10_LOCAL_DIST_DIR"' in script
    assert '--gb10-vllm-version "$GB10_VLLM_VERSION"' in script
    assert (
        '--gb10-context "GB10 local runtime build before Docker/Buildx starts"'
        in script
    )
    assert "Run scripts/gb10-build-cached.sh wheel first" in script
    assert '[ "$docker_target" = "vllm-openai" ]' in script
    assert '[ "$output_mode" != "cacheonly" ]' in script
    assert script.index(
        '--gb10-context "GB10 local runtime build before Docker/Buildx starts"'
    ) < script.index('mkdir -p "$cache_root"')
    assert script.index(
        '--gb10-context "GB10 local runtime build before Docker/Buildx starts"'
    ) < script.index('docker buildx inspect "$GB10_BUILDX_BUILDER" --bootstrap')


def test_gb10_local_cached_runtime_missing_wheel_fails_before_cache_or_docker(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    dist_dir = tmp_path / "dist"

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_LOCAL_DIST_DIR": str(dist_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "requires exactly one vLLM wheel" in proc.stdout
    assert "before Docker/Buildx starts" in proc.stdout
    assert "Run scripts/gb10-build-cached.sh wheel first" in proc.stdout
    assert not cache_dir.exists()


def test_gb10_local_cached_runtime_stale_wheel_fails_before_cache_or_docker(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "vllm-0.22.1rc0+gb10.stale-cp38-abi3-linux_aarch64.whl").write_bytes(
        b"stale"
    )

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_LOCAL_DIST_DIR": str(dist_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "does not match GB10_VLLM_VERSION" in proc.stdout
    assert "0.22.1rc0+gb10.abcdef123456" in proc.stdout
    assert "Run scripts/gb10-build-cached.sh wheel first" in proc.stdout
    assert not cache_dir.exists()


def test_gb10_local_cached_wheel_stale_output_fails_before_cache_or_docker(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "vllm-0.22.1rc0+gb10.stale-cp38-abi3-linux_aarch64.whl").write_bytes(
        b"stale"
    )

    proc = subprocess.run(
        ["bash", str(script), "wheel"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_LOCAL_DIST_DIR": str(dist_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "output directory before Docker/Buildx starts" in proc.stdout
    assert "does not match GB10_VLLM_VERSION" in proc.stdout
    assert "0.22.1rc0+gb10.abcdef123456" in proc.stdout
    assert "Remove stale vLLM wheels" in proc.stdout
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_invalid_dry_run_flag_before_manifest(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    dist_dir = tmp_path / "dist"

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "ture",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_LOCAL_DIST_DIR": str(dist_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "Unsupported GB10_DRY_RUN=ture" in proc.stdout
    assert "Use 1, 0, true, false, yes, no, on, or off" in proc.stdout
    assert "requires exactly one vLLM wheel" not in proc.stdout
    assert not (manifest_dir / "gb10-release-manifest.json").exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_invalid_registry_cache_flag_before_manifest(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_USE_REGISTRY_CACHE": "maybe",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "Unsupported GB10_USE_REGISTRY_CACHE=maybe" in proc.stdout
    assert "Use 1, 0, true, false, yes, no, on, or off" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not (manifest_dir / "gb10-release-manifest.json").exists()
    assert not cache_dir.exists()


@pytest.mark.parametrize(
    ("cache_env_name", "cache_ref"),
    (
        (
            "GB10_PREFLIGHT_CACHE_REF",
            "docker.io/gardner/vllm-gb10-buildcache:preflight",
        ),
        ("GB10_WHEEL_CACHE_REF", "ghcr.io/gardner/vllm-gb10-buildcache"),
        ("GB10_RUNTIME_CACHE_REF", "ghcr.io/Gardner/vllm-gb10-buildcache:runtime"),
    ),
)
def test_gb10_local_cached_build_rejects_invalid_cache_ref_before_manifest(
    tmp_path,
    cache_env_name,
    cache_ref,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            cache_env_name: cache_ref,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert f"GB10 local cached build cache ref {cache_env_name}" in proc.stdout
    assert "must be a GHCR image ref with a Docker-compatible tag" in proc.stdout
    assert cache_ref in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not (manifest_dir / "gb10-release-manifest.json").exists()
    assert not manifest_dir.exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_output_outside_manifest_dir(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    protected_output = tmp_path / "outside-runtime-metadata.json"
    protected_output.write_text("keep\n")

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_RUNTIME_IMAGE_METADATA_JSON": str(protected_output),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10 generated release output path GB10_RUNTIME_IMAGE_METADATA_JSON" in (
        proc.stdout
    )
    assert "must stay under GB10_LOCAL_RELEASE_MANIFEST_DIR" in proc.stdout
    assert protected_output.read_text() == "keep\n"
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not manifest_dir.exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_duplicate_release_output_paths(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    manifest_json = manifest_dir / "gb10-release-manifest.json"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_RUNTIME_IMAGE_METADATA_JSON": str(manifest_json),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10 generated release output paths must be unique" in proc.stdout
    assert "GB10_RELEASE_MANIFEST_JSON" in proc.stdout
    assert "GB10_RUNTIME_IMAGE_METADATA_JSON" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not manifest_json.exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_release_output_directory(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    metadata_dir = manifest_dir / "buildx-runtime-image-metadata.json"
    metadata_dir.mkdir(parents=True)

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_RUNTIME_IMAGE_METADATA_JSON": str(metadata_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10 generated release output path GB10_RUNTIME_IMAGE_METADATA_JSON" in (
        proc.stdout
    )
    assert "must be a file path" in proc.stdout
    assert "existing target is a directory" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert metadata_dir.is_dir()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_manifest_dir_file(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    manifest_dir.write_text("keep\n")

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10_LOCAL_RELEASE_MANIFEST_DIR must be a directory path" in proc.stdout
    assert "existing target is not a directory" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert manifest_dir.read_text() == "keep\n"
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_cache_root_file_before_manifest(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    cache_dir.write_text("keep\n")

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10_LOCAL_CACHE_DIR must be a directory path" in proc.stdout
    assert "existing target is not a directory" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert cache_dir.read_text() == "keep\n"
    assert not manifest_dir.exists()


@pytest.mark.parametrize("target", ("runtime", "wheel"))
def test_gb10_local_cached_build_rejects_dist_dir_file_before_manifest(
    tmp_path,
    target,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    dist_dir = tmp_path / "dist"
    dist_dir.write_text("keep\n")

    proc = subprocess.run(
        ["bash", str(script), target],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_LOCAL_DIST_DIR": str(dist_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10_LOCAL_DIST_DIR must be a directory path" in proc.stdout
    assert "existing target is not a directory" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert "requires exactly one vLLM wheel" not in proc.stdout
    assert dist_dir.read_text() == "keep\n"
    assert not manifest_dir.exists()
    assert not cache_dir.exists()


@pytest.mark.parametrize("builder_name", ("bad builder", "--bootstrap"))
def test_gb10_local_cached_build_rejects_invalid_buildx_builder_before_manifest(
    tmp_path,
    builder_name,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_BUILDX_BUILDER": builder_name,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10_BUILDX_BUILDER must be a Docker-compatible builder name" in (
        proc.stdout
    )
    assert builder_name in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not manifest_dir.exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_pushed_non_ghcr_image_before_manifest(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_OUTPUT": "push",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "requires a GHCR image-name" in proc.stdout
    assert "got vllm-gb10" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not (manifest_dir / "gb10-release-manifest.json").exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_pushed_stage_target_before_manifest(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_OUTPUT": "push",
            "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10_OUTPUT=push is only supported for runtime image builds" in (
        proc.stdout
    )
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not (manifest_dir / "gb10-release-manifest.json").exists()
    assert not cache_dir.exists()


@pytest.mark.parametrize(
    ("output_mode", "push_image", "expected_error"),
    (
        (
            "load",
            "true",
            "GB10_OUTPUT=load requires GB10_PUSH_IMAGE=false",
        ),
        (
            "push",
            "false",
            "GB10_OUTPUT=push requires GB10_PUSH_IMAGE=true",
        ),
    ),
)
def test_gb10_local_cached_build_rejects_output_push_image_mismatch_before_manifest(
    tmp_path,
    output_mode,
    push_image,
    expected_error,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_OUTPUT": output_mode,
            "GB10_PUSH_IMAGE": push_image,
            "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert expected_error in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not manifest_dir.exists()
    assert not cache_dir.exists()


@pytest.mark.parametrize(
    ("image_env", "expected_error"),
    (
        (
            {"GB10_IMAGE_TAG": "bad tag"},
            "GB10 local runtime image tag must be a Docker-compatible tag",
        ),
        (
            {"GB10_IMAGE_NAME": "bad image"},
            "GB10 local runtime image name must be a Docker-compatible repository name",
        ),
    ),
)
def test_gb10_local_cached_runtime_rejects_invalid_local_image_ref_before_manifest(
    tmp_path,
    image_env,
    expected_error,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_OUTPUT": "load",
            **image_env,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert expected_error in proc.stdout
    assert next(iter(image_env.values())) in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not manifest_dir.exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_invalid_native_arch_flag_before_manifest(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_NATIVE_CUDA_ARCHS_ONLY": "maybe",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "Unsupported GB10_NATIVE_CUDA_ARCHS_ONLY=maybe" in proc.stdout
    assert "Use 1, 0, true, false, yes, no, on, or off" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not (manifest_dir / "gb10-release-manifest.json").exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_rejects_disabled_native_arch_flag_before_manifest(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_NATIVE_CUDA_ARCHS_ONLY": "false",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "GB10_NATIVE_CUDA_ARCHS_ONLY must be enabled" in proc.stdout
    assert "got false" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not (manifest_dir / "gb10-release-manifest.json").exists()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_accepts_boolean_flag_aliases(tmp_path):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "true",
            "GB10_USE_REGISTRY_CACHE": "on",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stdout
    assert "GB10 local cached build dry run" in proc.stdout
    assert "registry_cache_enabled=1" in proc.stdout
    assert not cache_dir.exists()


def test_gb10_local_cached_build_script_dry_run_validates_before_cache_or_docker(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stdout
    assert "GB10 local cached build dry run" in proc.stdout
    assert "docker_target=gb10-flashinfer-preflight" in proc.stdout
    assert "cache_key=preflight" in proc.stdout
    assert f"cache_root={cache_dir}" in proc.stdout
    assert f"cache_dir={cache_dir / 'preflight'}" in proc.stdout
    assert f"cache_next={cache_dir / 'preflight.next'}" in proc.stdout
    assert f"cache_failed={cache_dir / 'preflight.failed'}" in proc.stdout
    assert (
        "registry_cache_refs="
        "ghcr.io/gardner/vllm-gb10-buildcache:preflight"
    ) in proc.stdout
    assert "GB10_PREFLIGHT_ONLY=true" in proc.stdout
    assert "GB10_BUILDX_BUILDER=gb10-builder" in proc.stdout
    assert (manifest_dir / "gb10-release-manifest.json").is_file()
    assert not cache_dir.exists()


def test_gb10_local_cached_build_dry_run_prints_resolved_env_keys():
    resolver = _load_gb10_release_settings_resolver_module()
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    missing_keys = [
        key
        for key in resolver.RESOLVED_ENV_KEYS
        if f'echo "{key}=$' not in script
    ]

    assert missing_keys == []


def test_gb10_local_cached_runtime_dry_run_uses_resolved_release_settings(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    github_sha = "abcdef1234567890abcdef1234567890abcdef12"

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": github_sha,
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stdout
    assert "docker_target=vllm-openai" in proc.stdout
    assert "cache_key=runtime" in proc.stdout
    assert f"cache_root={cache_dir}" in proc.stdout
    assert f"cache_dir={cache_dir / 'runtime'}" in proc.stdout
    assert f"cache_next={cache_dir / 'runtime.next'}" in proc.stdout
    assert f"cache_failed={cache_dir / 'runtime.failed'}" in proc.stdout
    assert (
        "registry_cache_refs="
        "ghcr.io/gardner/vllm-gb10-buildcache:wheel "
        "ghcr.io/gardner/vllm-gb10-buildcache:runtime"
    ) in proc.stdout
    assert "GB10_PREFLIGHT_ONLY=false" in proc.stdout
    assert "GB10_PUSH_IMAGE=false" in proc.stdout
    assert "GB10_IMAGE_NAME=vllm-gb10" in proc.stdout
    assert "GB10_IMAGE_TAG=gb10-abcdef123456" in proc.stdout
    assert "GB10_VLLM_VERSION=0.22.1rc0+gb10.abcdef123456" in proc.stdout
    assert (
        "GB10_FLASH_ATTN_REPO="
        "https://github.com/gardner/vllm-flash-attention.git"
    ) in proc.stdout
    assert f"GB10_FLASH_ATTN_REF={VLLM_FLASH_ATTN_GIT_TAG}" in proc.stdout
    assert (
        'GB10_RUNNER_LABELS=["self-hosted","linux","aarch64",'
        '"cuda13","dgx-spark","sm121"]'
    ) in proc.stdout
    assert "DEEPGEMM_GIT_REPOSITORY=https://github.com/gardner/DeepGEMM.git" in (
        proc.stdout
    )
    assert f"DEEPGEMM_GIT_TAG={DEEPGEMM_GIT_TAG}" in proc.stdout
    assert "FLASH_MLA_GIT_REPOSITORY=https://github.com/gardner/FlashMLA.git" in (
        proc.stdout
    )
    assert f"FLASH_MLA_GIT_TAG={FLASHMLA_GIT_TAG}" in proc.stdout
    assert "TRITON_KERNELS_GIT_REPOSITORY=https://github.com/gardner/triton.git" in (
        proc.stdout
    )
    assert f"TRITON_KERNELS_GIT_TAG={TRITON_KERNELS_GIT_TAG}" in proc.stdout
    assert (
        f"GB10_RELEASE_MANIFEST_JSON="
        f"{manifest_dir / 'gb10-release-manifest.json'}"
    ) in proc.stdout
    assert (
        f"GB10_RELEASE_CHECKSUMS="
        f"{manifest_dir / 'gb10-vllm-release-SHA256SUMS'}"
    ) in proc.stdout
    assert (
        f"GB10_RUNTIME_IMAGE_REF={manifest_dir / 'gb10-runtime-image-ref.txt'}"
    ) in proc.stdout
    assert (
        f"GB10_RUNTIME_IMAGE_DIGEST="
        f"{manifest_dir / 'gb10-runtime-image-digest.txt'}"
    ) in proc.stdout
    assert not cache_dir.exists()

    manifest_path = manifest_dir / "gb10-release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["git"]["commit"] == github_sha
    assert manifest["image"] == {
        "name": "vllm-gb10",
        "tag": "gb10-abcdef123456",
        "push": False,
    }
    assert manifest["release"]["preflight_only"] is False
    assert manifest["vllm"]["version"] == "0.22.1rc0+gb10.abcdef123456"


def test_gb10_local_cached_release_dry_run_prints_release_tag(tmp_path):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    github_sha = "abcdef1234567890abcdef1234567890abcdef12"
    release_tag = "gb10-vllm-v0.22.1rc0-abcdef123"

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_OUTPUT": "push",
            "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
            "GB10_RELEASE_TAG": release_tag,
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": github_sha,
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stdout
    assert "GB10 local cached build dry run" in proc.stdout
    assert "GB10_OUTPUT=push" in proc.stdout
    assert f"GB10_RELEASE_TAG={release_tag}" in proc.stdout
    assert "GB10_PUSH_IMAGE=true" in proc.stdout
    assert f"GB10_IMAGE_TAG={release_tag}" in proc.stdout
    assert "GB10_VLLM_VERSION=0.22.1rc0+gb10.abcdef123456" in proc.stdout
    assert not cache_dir.exists()

    manifest = json.loads((manifest_dir / "gb10-release-manifest.json").read_text())
    assert manifest["release"]["tag"] == release_tag
    assert manifest["image"] == {
        "name": "ghcr.io/gardner/vllm-gb10",
        "tag": release_tag,
        "push": True,
    }


def test_gb10_local_cached_release_dry_run_rejects_unpushed_tagged_image(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_OUTPUT": "load",
            "GB10_RELEASE_TAG": "gb10-vllm-v0.22.1rc0-abcdef123",
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "requires push-image=true" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not cache_dir.exists()
    assert not (manifest_dir / "gb10-release-manifest.json").exists()


def test_gb10_local_cached_release_dry_run_rejects_invalid_release_tag_before_manifest(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-build-cached.sh"
    manifest_dir = tmp_path / "manifest"
    cache_dir = tmp_path / "cache"
    stale_outputs = (
        manifest_dir / "gb10-release-manifest.json",
        manifest_dir / "gb10-vllm-release-SHA256SUMS",
        manifest_dir / "gb10-runtime-image-ref.txt",
        manifest_dir / "gb10-runtime-image-digest.txt",
        manifest_dir / "buildx-runtime-image-metadata.json",
    )
    manifest_dir.mkdir()
    for stale_output in stale_outputs:
        stale_output.write_text("stale\n")

    proc = subprocess.run(
        ["bash", str(script), "runtime"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GB10_DRY_RUN": "1",
            "GB10_LOCAL_RELEASE_MANIFEST_DIR": str(manifest_dir),
            "GB10_LOCAL_CACHE_DIR": str(cache_dir),
            "GB10_OUTPUT": "push",
            "GB10_IMAGE_NAME": "ghcr.io/gardner/vllm-gb10",
            "GB10_RELEASE_TAG": "bad-tag",
            "GITHUB_EVENT_NAME": "",
            "GITHUB_REF": "",
            "GITHUB_SHA": "abcdef1234567890abcdef1234567890abcdef12",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )

    assert proc.returncode != 0, proc.stdout
    assert "release-tag must match" in proc.stdout
    assert "GB10 local cached build dry run" not in proc.stdout
    assert not cache_dir.exists()
    assert not any(stale_output.exists() for stale_output in stale_outputs)


def test_gb10_local_cached_build_script_preserves_failed_cache_exports():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert 'cache_failed="$cache_root/${cache_key}.failed"' in script
    assert '[ -f "$cache_failed/index.json" ]' in script
    assert '--cache-from "type=local,src=$cache_failed"' in script
    assert "build_status=$?" in script
    assert '[ "$build_status" -ne 0 ]' in script
    assert 'mv "$cache_next" "$cache_failed"' in script
    assert "GB10 BuildKit cache export from failed build preserved" in script
    assert 'exit "$build_status"' in script


def test_gb10_release_workflow_overrides_pep440_wheel_version():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    release_settings_resolver = (
        REPO_ROOT / "scripts" / "gb10-resolve-release-settings.py"
    ).read_text()

    assert 'DEFAULT_VLLM_VERSION_BASE = "0.22.1rc0"' in release_settings_resolver
    assert 'f"{_vllm_version_base(release_tag)}+gb10.{github_sha[:12]}"' in (
        release_settings_resolver
    )
    assert '"GB10_VLLM_VERSION"' in release_settings_resolver
    assert '--build-arg vllm_version_override="$GB10_VLLM_VERSION"' in gb10_workflow
    assert 'ARG vllm_version_override=""' in dockerfile
    assert 'export VLLM_VERSION_OVERRIDE="${vllm_version_override}"' in dockerfile
    assert dockerfile.index('ARG vllm_version_override=""') > dockerfile.index(
        "FROM base AS build"
    )


def test_gb10_dockerfile_exports_cuda_home_for_extension_builds():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    cuda_home_env = "ENV CUDA_HOME=/usr/local/cuda"

    assert dockerfile.count(cuda_home_env) >= 2
    assert dockerfile.index(cuda_home_env) < dockerfile.index(
        "FROM base AS extensions-build"
    )
    assert dockerfile.index(cuda_home_env) < dockerfile.index("FROM base AS build")
    assert dockerfile.index("FROM ${FINAL_BASE_IMAGE} AS vllm-base") < (
        dockerfile.rindex(cuda_home_env)
    )


def test_gb10_ep_kernel_helper_exports_cuda_home_for_uv_build():
    install_script = (
        REPO_ROOT / "tools" / "ep_kernels" / "install_python_libraries.sh"
    ).read_text()

    assert "export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}" in install_script


def test_gb10_ep_kernel_build_uses_native_torch_cuda_arch_list():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    versions_json = (REPO_ROOT / "docker" / "versions.json").read_text()

    assert "export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\"" in dockerfile
    assert "export TORCH_CUDA_ARCH_LIST='9.0a 10.0a'" not in dockerfile
    assert '"TORCH_CUDA_ARCH_LIST": {\n      "default": "12.1a"\n    }' in versions_json


def test_gb10_prebuilt_flashinfer_wheels_do_not_preserve_unpinned_torch():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    prebuilt_block = dockerfile.split(
        'if [ -n "${gb10_prebuilt_wheel_urls}" ]; then',
        1,
    )[1].split('if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "12" ]; then', 1)[0]

    assert 'uv pip install --python /opt/venv/bin/python3 --no-deps "$wheel_url"' in (
        prebuilt_block
    )
    assert "use_existing_torch.py" not in prebuilt_block
    assert "GB10 build requires CUDA-enabled PyTorch" in dockerfile


def test_gb10_runtime_image_uses_published_flashinfer_wheels():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    runtime_stage = dockerfile.split(
        "FROM ${FINAL_BASE_IMAGE} AS vllm-base",
        1,
    )[1].split("FROM vllm-base AS test", 1)[0]

    assert 'ARG gb10_prebuilt_wheel_urls=""' in runtime_stage
    assert 'ARG gb10_require_flashinfer_wheels=true' in runtime_stage
    assert "Skipping public FlashInfer JIT cache install" in runtime_stage
    assert "Installing GB10 runtime FlashInfer wheels" in runtime_stage
    assert 'uv pip install --system --no-deps "$wheel_url"' in runtime_stage
    assert "GB10 runtime FlashInfer wheels are required" in runtime_stage
    assert "verify_gb10_flashinfer_jit_cache.py" in runtime_stage
    assert "Skipping FlashInfer remote cubin download for GB10" in runtime_stage
    assert runtime_stage.index("Installing GB10 runtime FlashInfer wheels") < (
        runtime_stage.index("python3 /tmp/verify_gb10_flashinfer_jit_cache.py")
    )
    assert runtime_stage.index("Skipping FlashInfer remote cubin download for GB10") < (
        runtime_stage.index("flashinfer download-cubin")
    )


def test_gb10_runtime_flashinfer_jit_cache_validator_uses_cuobjdump():
    validator = (
        REPO_ROOT / "docker" / "verify_gb10_flashinfer_jit_cache.py"
    ).read_text()

    assert 'ALLOWED_CUDA_IMAGES = {"sm_121a", "compute_121a"}' in validator
    assert '"cuobjdump", "--list-elf"' in validator
    assert "unexpected CUDA images" in validator
    assert "GB10 FlashInfer JIT cache has no CUDA images" in validator


def test_nvfp4_swiglu_limit_uses_sm12x_capable_flashinfer_cutlass():
    nvfp4_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "fused_moe" / "oracle" / "nvfp4.py"
    ).read_text()

    clamp_allowlist = nvfp4_oracle.split("NVFP4_BACKENDS_WITH_CLAMP = {", 1)[1].split(
        "}", 1
    )[0]

    assert "NvFp4MoeBackend.FLASHINFER_TRTLLM" in clamp_allowlist
    assert "NvFp4MoeBackend.FLASHINFER_CUTLASS" in clamp_allowlist
    assert "NvFp4MoeBackend.FLASHINFER_B12X" not in clamp_allowlist


def test_gb10_nvfp4_linear_fallbacks_are_reported():
    linear_selector = (
        REPO_ROOT / "vllm" / "model_executor" / "kernels" / "linear" /
        "__init__.py"
    ).read_text()
    flashinfer_nvfp4_linear = (
        REPO_ROOT / "vllm" / "model_executor" / "kernels" / "linear" /
        "nvfp4" / "flashinfer.py"
    ).read_text()
    modelopt_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "modelopt.py"
    ).read_text()
    compressed_tensors_w4a16 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_w4a16_nvfp4.py"
    ).read_text()
    compressed_tensors_w4a4_nvfp4 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_w4a4_nvfp4.py"
    ).read_text()
    compressed_tensors_w4a4_mxfp4 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_w4a4_mxfp4.py"
    ).read_text()
    compressed_tensors_w4a4_nvfp4_moe = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "compressed_tensors_moe" /
        "compressed_tensors_moe_w4a4_nvfp4.py"
    ).read_text()
    compressed_tensors_transform_linear = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "transform" / "linear.py"
    ).read_text()
    compressed_tensors_qutlass_nvfp4_transform = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "transform" / "schemes" /
        "linear_qutlass_nvfp4.py"
    ).read_text()
    compressed_tensors_w4a8_int = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_w4a8_int.py"
    ).read_text()
    compressed_tensors_w8a8_int8 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_w8a8_int8.py"
    ).read_text()
    compressed_tensors_w8a8_mxfp8 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_w8a8_mxfp8.py"
    ).read_text()
    compressed_tensors_wna16 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_wNa16.py"
    ).read_text()
    compressed_tensors_w8a16_fp8 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_w8a16_fp8.py"
    ).read_text()
    compressed_tensors_w8a8_fp8 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "schemes" /
        "compressed_tensors_w8a8_fp8.py"
    ).read_text()
    compressed_tensors_mxfp4_moe = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "compressed_tensors_moe" /
        "compressed_tensors_moe_w4a4_mxfp4.py"
    ).read_text()
    compressed_tensors_moe = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "compressed_tensors_moe" /
        "compressed_tensors_moe.py"
    ).read_text()
    compressed_tensors = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "compressed_tensors.py"
    ).read_text()
    compressed_tensors_utils = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "utils.py"
    ).read_text()
    online_quant_base = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "online" / "base.py"
    ).read_text()
    fp8_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "fp8.py"
    ).read_text()
    deepseek_v4_quant = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "quant_config.py"
    ).read_text()
    torchao_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "torchao.py"
    ).read_text()
    bitsandbytes_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "bitsandbytes.py"
    ).read_text()
    awq_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "awq.py"
    ).read_text()
    awq_marlin_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "awq_marlin.py"
    ).read_text()
    auto_gptq_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "auto_gptq.py"
    ).read_text()
    inc_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "inc.py"
    ).read_text()
    gguf_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "gguf.py"
    ).read_text()
    humming_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "humming.py"
    ).read_text()
    fbgemm_fp8_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "fbgemm_fp8.py"
    ).read_text()
    experts_int8_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "experts_int8.py"
    ).read_text()
    fp_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "fp_quant.py"
    ).read_text()
    mxfp4_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "mxfp4.py"
    ).read_text()
    moe_wna16 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "moe_wna16.py"
    ).read_text()
    quark_w4a8_mxfp4_fp8 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "quark" / "schemes" / "quark_w4a8_mxfp4_fp8.py"
    ).read_text()
    quark_w8a8_fp8 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "quark" / "schemes" / "quark_w8a8_fp8.py"
    ).read_text()
    quark_w8a8_int8 = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "quark" / "schemes" / "quark_w8a8_int8.py"
    ).read_text()
    quark_utils = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "quark" / "utils.py"
    ).read_text()
    quark_moe = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "quark" / "quark_moe.py"
    ).read_text()
    mxfp4_moe_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" /
        "oracle" / "mxfp4.py"
    ).read_text()
    mxfp8_moe_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" /
        "oracle" / "mxfp8.py"
    ).read_text()
    wna16_moe_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" /
        "oracle" / "int_wna16.py"
    ).read_text()
    unquantized_moe_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" /
        "oracle" / "unquantized.py"
    ).read_text()
    fp8_moe_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" /
        "oracle" / "fp8.py"
    ).read_text()
    int8_moe_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" /
        "oracle" / "int8.py"
    ).read_text()

    assert "_log_nvfp4_linear_kernel_selection" in linear_selector
    assert "record_nvfp4_backend_selection" in linear_selector
    assert "record_nvfp4_fallback" in linear_selector
    assert "NVFP4 linear selected fallback backend " in linear_selector
    assert "not the native " in linear_selector
    assert "GB10 W4A4 FP4 Tensor Core path" in linear_selector
    assert "verify " in linear_selector
    assert "this fallback is intentional " in linear_selector
    assert "before publishing GB10 artifacts" in linear_selector
    assert "MarlinNvFp4LinearKernel" in linear_selector
    assert "EmulationNvFp4LinearKernel" in linear_selector
    assert "FbgemmNvFp4LinearKernel" in linear_selector
    assert "MarlinMxFp4LinearKernel" in linear_selector
    assert "MarlinFP8ScaledMMLinearKernel" in linear_selector
    assert "MarlinMxfp8LinearKernel" in linear_selector
    assert "EmulationMxfp8LinearKernel" in linear_selector
    assert "_gb10_wfp8a16_linear_fallback_unsupported_reason" in linear_selector
    assert "_gb10_mxfp4_linear_fallback_unsupported_reason" in linear_selector
    assert "_gb10_mxfp8_linear_fallback_unsupported_reason" in linear_selector
    assert "FP8 W8A16 Marlin fallback" in linear_selector
    assert "non-native MXFP4 dense fallback" in linear_selector
    assert "non-native MXFP8 dense fallback" in linear_selector
    assert "FBGEMM NVFP4 dense backend" in linear_selector
    assert "_gb10_w8a8_mxfp8_dense_unsupported_reason" in (
        compressed_tensors_w8a8_mxfp8
    )
    assert "CompressedTensors W8A8 MXFP8 dense loading" in (
        compressed_tensors_w8a8_mxfp8
    )
    assert "MXFP8 dense kernel selection" in compressed_tensors_w8a8_mxfp8
    assert "not supported on GB10/SM12x" in compressed_tensors_w8a8_mxfp8
    assert "_gb10_online_fp8_quantization_unsupported_reason" in online_quant_base
    assert "Online FP8 quantization" in online_quant_base
    assert "FP8 scaled-mm dense kernels" in online_quant_base
    assert "generic FP8 MoE backend selection" in online_quant_base
    assert "not supported on GB10/SM12x" in online_quant_base
    assert "_gb10_online_mxfp8_quantization_unsupported_reason" in (
        online_quant_base
    )
    assert "Online MXFP8 quantization" in online_quant_base
    assert "FlashInfer CUTLASS MXFP8 dense" in online_quant_base
    assert "generic MXFP8 MoE backend selection" in online_quant_base
    assert "not supported on GB10/SM12x" in online_quant_base
    assert "_gb10_online_mxfp4_quantization_unsupported_reason" in (
        online_quant_base
    )
    assert "Online MXFP4 quantization" in online_quant_base
    assert "weight='mxfp4'" in online_quant_base
    assert "no online MXFP4 method is wired" in online_quant_base
    assert "native GB10 online MXFP4 correctness evidence" in online_quant_base
    assert "not supported on GB10/SM12x" in online_quant_base
    assert "_gb10_online_int8_moe_quantization_unsupported_reason" in (
        online_quant_base
    )
    assert "Online Int8 MoE quantization" in online_quant_base
    assert "backend selection today" in online_quant_base
    assert "not supported on GB10/SM12x" in online_quant_base
    assert "_gb10_fbgemm_fp8_quantization_unsupported_reason" in fbgemm_fp8_quant
    assert "FBGEMM FP8 quantization" in fbgemm_fp8_quant
    assert "deprecated public quantization method" in fbgemm_fp8_quant
    assert "generic FP8 linear kernel selection" in fbgemm_fp8_quant
    assert "not supported on GB10/SM12x" in fbgemm_fp8_quant
    assert (
        "_gb10_experts_int8_quantization_unsupported_reason"
        in experts_int8_quant
    )
    assert "ExpertsInt8 quantization" in experts_int8_quant
    assert "backward-compatible public quantization method" in experts_int8_quant
    assert "online Int8 MoE backend selection" in experts_int8_quant
    assert "not supported on GB10/SM12x" in experts_int8_quant
    assert "_gb10_fp_quant_fp4_quantization_unsupported_reason" in fp_quant
    assert "FPQuant FP4 quantization" in fp_quant
    assert "deprecated public quantization method" in fp_quant
    assert "MXFP4/NVFP4 FPQuant linear kernels" in fp_quant
    assert "not supported on GB10/SM12x" in fp_quant
    assert "_gb10_public_mxfp4_quantization_unsupported_reason" in mxfp4_quant
    assert "Public MXFP4 quantization" in mxfp4_quant
    assert "mxfp4 and gpt_oss_mxfp4" in mxfp4_quant
    assert "unquantized linear/attention handling" in mxfp4_quant
    assert "MXFP4 MoE backend selection" in mxfp4_quant
    assert "not supported on GB10/SM12x" in mxfp4_quant
    assert "_gb10_public_fp8_quantization_unsupported_reason" in fp8_quant
    assert "Public FP8 quantization" in fp8_quant
    assert "fp8 quantization method" in fp8_quant
    assert "FP8 scaled-mm dense kernel selection" in fp8_quant
    assert "FP8 MoE backend selection" in fp8_quant
    assert "online FP8 quantization path" in fp8_quant
    assert "not supported on GB10/SM12x" in fp8_quant
    assert "_gb10_deepseek_v4_fp8_quantization_unsupported_reason" in (
        deepseek_v4_quant
    )
    assert "DeepSeek V4 FP8 quantization" in deepseek_v4_quant
    assert "deepseek_v4_fp8 quantization method" in deepseek_v4_quant
    assert "FP8 block-quantized linear/attention layers" in deepseek_v4_quant
    assert "FP8, MXFP4, or ModelOpt NVFP4 MoE dispatch" in deepseek_v4_quant
    assert "not supported on GB10/SM12x" in deepseek_v4_quant
    assert "_gb10_torchao_fp8_activation_unsupported_reason" in torchao_quant
    assert "TorchAO FP8 activation quantization" in torchao_quant
    assert "torchao quantization method" in torchao_quant
    assert "Float8" in torchao_quant
    assert "Activation" in torchao_quant
    assert "torchao.quantization.quantize_" in torchao_quant
    assert "convert_to_packed_tensor_based_on_current_hardware" in torchao_quant
    assert "not supported on GB10/SM12x" in torchao_quant
    assert "_gb10_torchao_weight_quantization_unsupported_reason" in torchao_quant
    assert "TorchAO weight quantization" in torchao_quant
    assert "native GB10 TorchAO weight correctness evidence" in torchao_quant
    assert "_gb10_bitsandbytes_quantization_unsupported_reason" in (
        bitsandbytes_quant
    )
    assert "BitsAndBytes quantization" in bitsandbytes_quant
    assert "bitsandbytes quantization method" in bitsandbytes_quant
    assert "bitsandbytes 4-bit linear kernels" in bitsandbytes_quant
    assert "bitsandbytes 8-bit matmul kernels" in bitsandbytes_quant
    assert "BitsAndBytesMoEMethod" in bitsandbytes_quant
    assert "not supported on GB10/SM12x" in bitsandbytes_quant
    assert "_gb10_awq_quantization_unsupported_reason" in awq_quant
    assert "AWQ quantization" in awq_quant
    assert "awq and awq_marlin quantization methods" in awq_quant
    assert "AWQLinearMethod" in awq_quant
    assert "AWQMarlinLinearMethod" in awq_quant
    assert "AWQMarlinMoEMethod" in awq_quant
    assert "MoeWNA16Config" in awq_quant
    assert "not supported on GB10/SM12x" in awq_quant
    assert "_gb10_awq_quantization_unsupported_reason" in awq_marlin_quant
    assert "AWQMarlinConfig" in awq_marlin_quant
    assert "_gb10_gptq_quantization_unsupported_reason" in auto_gptq_quant
    assert "GPTQ quantization" in auto_gptq_quant
    assert "auto_gptq, gptq, and gptq_marlin quantization methods" in (
        auto_gptq_quant
    )
    assert "get_linear_quant_method" in auto_gptq_quant
    assert "choose_mp_linear_kernel" in auto_gptq_quant
    assert "AutoGPTQMoEMethod" in auto_gptq_quant
    assert "MoeWNA16Config" in auto_gptq_quant
    assert "not supported on GB10/SM12x" in auto_gptq_quant
    assert "_gb10_inc_quantization_unsupported_reason" in inc_quant
    assert "INC/AutoRound quantization" in inc_quant
    assert "inc and auto-round quantization methods" in inc_quant
    assert "apply_awq_quant_layer" in inc_quant
    assert "apply_gptq_quant_layer" in inc_quant
    assert "AWQMarlinLinearMethod" in inc_quant
    assert "AutoGPTQLinearMethod" in inc_quant
    assert "MoeWNA16Config" in inc_quant
    assert "not supported on GB10/SM12x" in inc_quant
    assert "_gb10_gguf_quantization_unsupported_reason" in gguf_quant
    assert "GGUF quantization" in gguf_quant
    assert "gguf quantization method" in gguf_quant
    assert "GGUFLinearMethod" in gguf_quant
    assert "GGUFEmbeddingMethod" in gguf_quant
    assert "GGUFMoEMethod" in gguf_quant
    assert "ggml_mul_mat_vec_a8" in gguf_quant
    assert "ggml_mul_mat_a8" in gguf_quant
    assert "ggml_dequantize" in gguf_quant
    assert "not supported on GB10/SM12x" in gguf_quant
    assert "_gb10_humming_quantization_unsupported_reason" in humming_quant
    assert "Humming quantization" in humming_quant
    assert "humming quantization method" in humming_quant
    assert "HummingLinearMethod" in humming_quant
    assert "HummingMoEMethod" in humming_quant
    assert "HummingMethod.prepare_layer_meta" in humming_quant
    assert "HummingMethod.transform_humming_layer" in humming_quant
    assert "HummingMethod.forward_layer" in humming_quant
    assert "not supported on GB10/SM12x" in humming_quant
    assert "FlashInfer TRTLLM NVFP4 dense is not supported on GB10/SM12x" in (
        flashinfer_nvfp4_linear
    )
    assert "FlashInfer cuDNN NVFP4 dense is deferred on GB10/SM12x" in (
        flashinfer_nvfp4_linear
    )

    assert "W4A16_NVFP4 linear selected MarlinNvFp4LinearKernel" in modelopt_quant
    assert "_gb10_w4a16_nvfp4_marlin_unsupported_reason" in modelopt_quant
    assert "_gb10_w4a16_nvfp4_moe_unsupported_reason" in modelopt_quant
    assert "not supported on GB10/SM12x" in modelopt_quant
    assert "record_nvfp4_backend_selection" in modelopt_quant
    assert "record_nvfp4_fallback" in modelopt_quant
    assert '"linear_w4a16"' in modelopt_quant
    assert "weight-only fallback path" in modelopt_quant
    assert "not the native GB10 W4A4 FP4 Tensor " in modelopt_quant
    assert "Core path; verify this fallback is intentional " in modelopt_quant
    assert "_gb10_modelopt_mxfp8_quantization_unsupported_reason" in (
        modelopt_quant
    )
    assert "_gb10_modelopt_fp8_quantization_unsupported_reason" in modelopt_quant
    assert "ModelOpt FP8 quantization" in modelopt_quant
    assert "FP8 dense kernel selection" in modelopt_quant
    assert "FP8 MoE backend selection" in modelopt_quant
    assert "FP8_PER_CHANNEL_PER_TOKEN" in modelopt_quant
    assert "FP8_PB_WO" in modelopt_quant
    assert "not supported on GB10/SM12x" in modelopt_quant
    assert "ModelOpt MXFP8 quantization" in modelopt_quant
    assert "MXFP8 dense kernel selection" in modelopt_quant
    assert "MXFP8 MoE backend selection" in modelopt_quant
    assert "not supported on GB10/SM12x" in modelopt_quant
    assert "_gb10_modelopt_mixed_quantization_unsupported_reason" in modelopt_quant
    assert "_gb10_modelopt_nvfp4_kv_cache_unsupported_reason" in modelopt_quant
    assert "ModelOpt NVFP4 KV-cache loading" in modelopt_quant
    assert "kv_cache_quant_algo=NVFP4" in modelopt_quant
    assert "kv_cache_dtype='nvfp4'" in modelopt_quant
    assert "FP8 E4M3 KV cache" in modelopt_quant
    assert "native SM12x NVFP4 KV-cache correctness evidence" in modelopt_quant
    assert "ModelOpt mixed precision quantization" in modelopt_quant
    assert "FP8 dense or MoE selection" in modelopt_quant
    assert "NVFP4 dense or MoE selection" in modelopt_quant
    assert "W4A16 NVFP4 fallback selection" in modelopt_quant
    assert "not supported on GB10/SM12x" in modelopt_quant

    assert "_gb10_w4a16_nvfp4_marlin_unsupported_reason" in (
        compressed_tensors_w4a16
    )
    assert "CompressedTensors W4A16 NVFP4 loading would select" in (
        compressed_tensors_w4a16
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_w4a16
    assert 'if type_ != "float" or num_bits != 8:' in compressed_tensors
    assert "Currently supported kv cache quantization" in compressed_tensors
    assert "init_nvfp4_linear_kernel" in compressed_tensors_w4a4_nvfp4
    assert "self.kernel = init_nvfp4_linear_kernel()" in (
        compressed_tensors_w4a4_nvfp4
    )
    assert "select_nvfp4_moe_backend" in compressed_tensors_w4a4_nvfp4_moe
    assert "make_nvfp4_moe_kernel" in compressed_tensors_w4a4_nvfp4_moe
    assert "_gb10_w4a16_nvfp4_moe_loading_unsupported_reason" in (
        compressed_tensors_moe
    )
    assert "CompressedTensors W4A16 NVFP4 MoE loading" in compressed_tensors_moe
    assert "weight-only NVFP4 MoE handling" in compressed_tensors_moe
    assert "not supported on GB10/SM12x" in compressed_tensors_moe
    assert "_gb10_qutlass_nvfp4_transform_unsupported_reason" in (
        compressed_tensors_transform_linear
    )
    assert "_gb10_qutlass_nvfp4_transform_unsupported_reason" in (
        compressed_tensors_qutlass_nvfp4_transform
    )
    assert "CompressedTensors Qutlass NVFP4 transform loading" in (
        compressed_tensors_qutlass_nvfp4_transform
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_qutlass_nvfp4_transform
    assert "_gb10_w4a4_mxfp4_dense_unsupported_reason" in (
        compressed_tensors_w4a4_mxfp4
    )
    assert "CompressedTensors W4A4 MXFP4 dense loading is not validated" in (
        compressed_tensors_w4a4_mxfp4
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_w4a4_mxfp4
    assert "_gb10_mxfp4_moe_marlin_unsupported_reason" in (
        compressed_tensors_mxfp4_moe
    )
    assert "CompressedTensors W4A4 MXFP4 MoE would select" in (
        compressed_tensors_mxfp4_moe
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_mxfp4_moe
    assert "_gb10_compressed_tensors_wna16_moe_unsupported_reason" in (
        compressed_tensors_moe
    )
    assert "CompressedTensors WNA16 MoE legacy fused-experts fallback" in (
        compressed_tensors_moe
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_moe
    assert "_gb10_moe_wna16_legacy_unsupported_reason" in moe_wna16
    assert "MoeWNA16 legacy fused-experts fallback" in moe_wna16
    assert "not supported on GB10/SM12x" in moe_wna16
    assert "gb10_quark_w4a8_mxfp4_fp8_unsupported_reason" in (
        quark_w4a8_mxfp4_fp8
    )
    assert "gb10_quark_w4a8_mxfp4_fp8_unsupported_reason" in quark_utils
    assert "gb10_quark_w4a8_fp8_moe_unsupported_reason" in quark_moe
    assert "gb10_quark_w4a8_fp8_moe_unsupported_reason" in quark_utils
    assert "Quark W4A8 FP8 MoE checkpoint loading" in quark_utils
    assert "ROCm AITER fused MoE support" in quark_utils
    assert "gb10_quark_w8a8_fp8_unsupported_reason" in quark_w8a8_fp8
    assert "gb10_quark_w8a8_fp8_unsupported_reason" in quark_utils
    assert "gb10_quark_w8a8_int8_unsupported_reason" in quark_w8a8_int8
    assert "gb10_quark_w8a8_int8_unsupported_reason" in quark_utils
    assert "Quark W8A8 FP8 checkpoint loading" in quark_utils
    assert "Quark W8A8 Int8 checkpoint loading" in quark_utils
    assert "gb10_quark_w8a8_fp8_moe_unsupported_reason" in quark_moe
    assert "gb10_quark_w8a8_fp8_moe_unsupported_reason" in quark_utils
    assert "gb10_quark_w8a8_int8_moe_unsupported_reason" in quark_moe
    assert "gb10_quark_w8a8_int8_moe_unsupported_reason" in quark_utils
    assert "Quark W8A8 FP8 MoE checkpoint loading" in quark_utils
    assert "Quark W8A8 Int8 MoE checkpoint loading" in quark_utils
    assert "not supported on GB10/SM12x" in quark_utils
    assert "gb10_compressed_tensors_w4a8_fp8_unsupported_reason" in (
        compressed_tensors
    )
    assert "gb10_compressed_tensors_w4a8_fp8_unsupported_reason" in (
        compressed_tensors_moe
    )
    assert "_gb10_w8a8_fp8_moe_loading_unsupported_reason" in (
        compressed_tensors_moe
    )
    assert "CompressedTensors W8A8 FP8 MoE loading" in compressed_tensors_moe
    assert "generic FP8 W8A8 MoE backend selection" in compressed_tensors_moe
    assert "not supported on GB10/SM12x" in compressed_tensors_moe
    assert "_gb10_w8a8_mxfp8_moe_loading_unsupported_reason" in (
        compressed_tensors_moe
    )
    assert "CompressedTensors W8A8 MXFP8 MoE loading" in compressed_tensors_moe
    assert "generic MXFP8 MoE backend selection" in compressed_tensors_moe
    assert "not supported on GB10/SM12x" in compressed_tensors_moe
    assert "_gb10_w8a8_int_moe_loading_unsupported_reason" in (
        compressed_tensors_moe
    )
    assert "CompressedTensors W8A8 Int8 MoE loading" in compressed_tensors_moe
    assert "generic Int8 W8A8 MoE backend selection" in compressed_tensors_moe
    assert "not supported on GB10/SM12x" in compressed_tensors_moe
    assert "CompressedTensors W4A8 FP8 checkpoint loading" in (
        compressed_tensors_utils
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_utils
    assert "_gb10_w4a8_int_dense_unsupported_reason" in compressed_tensors_w4a8_int
    assert "CompressedTensors W4A8 Int dense loading" in compressed_tensors_w4a8_int
    assert "generic mixed-precision" in compressed_tensors_w4a8_int
    assert "not supported on GB10/SM12x" in compressed_tensors_w4a8_int
    assert "_gb10_w4a8_int_moe_loading_unsupported_reason" in (
        compressed_tensors_moe
    )
    assert "CompressedTensors W4A8 Int8 MoE loading" in compressed_tensors_moe
    assert "CPU-only W4A8 Int8 MoE backend selection" in compressed_tensors_moe
    assert "not supported on GB10/SM12x" in compressed_tensors_moe
    assert "_gb10_wna16_dense_unsupported_reason" in compressed_tensors_wna16
    assert "CompressedTensors WNA16 dense loading" in compressed_tensors_wna16
    assert "generic mixed-precision" in compressed_tensors_wna16
    assert "not supported on GB10/SM12x" in compressed_tensors_wna16
    assert "_gb10_w8a16_fp8_loading_unsupported_reason" in (
        compressed_tensors_w8a16_fp8
    )
    assert "CompressedTensors W8A16 FP8 loading" in compressed_tensors_w8a16_fp8
    assert "FP8 W8A16 Marlin fallback" in compressed_tensors_w8a16_fp8
    assert "not supported on GB10/SM12x" in compressed_tensors_w8a16_fp8
    assert "_gb10_w8a8_fp8_loading_unsupported_reason" in (
        compressed_tensors_w8a8_fp8
    )
    assert "CompressedTensors W8A8 FP8 loading" in compressed_tensors_w8a8_fp8
    assert "scaled-mm W8A8 FP8 kernels" in compressed_tensors_w8a8_fp8
    assert "not supported on GB10/SM12x" in compressed_tensors_w8a8_fp8
    assert "_gb10_w8a8_int_dense_unsupported_reason" in compressed_tensors_w8a8_int8
    assert "CompressedTensors W8A8 Int dense loading" in compressed_tensors_w8a8_int8
    assert "Cutlass/Triton W8A8 Int8 scaled-mm kernels" in (
        compressed_tensors_w8a8_int8
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_w8a8_int8
    assert "weight_quant.type == QuantizationType.INT" in compressed_tensors
    assert "_gb10_mxfp4_moe_fallback_unsupported_reason" in mxfp4_moe_oracle
    assert "_gb10_mxfp4_moe_trtllm_unsupported_reason" in mxfp4_moe_oracle
    assert "_gb10_mxfp4_moe_humming_unsupported_reason" in mxfp4_moe_oracle
    assert "_gb10_mxfp4_moe_aiter_unsupported_reason" in mxfp4_moe_oracle
    assert (
        "_gb10_mxfp4_moe_gpt_oss_triton_unsupported_reason"
        in mxfp4_moe_oracle
    )
    assert "_MXFP4_MOE_FALLBACK_BACKENDS" in mxfp4_moe_oracle
    assert "_MXFP4_MOE_TRTLLM_BACKENDS" in mxfp4_moe_oracle
    assert "_MXFP4_MOE_HUMMING_BACKENDS" in mxfp4_moe_oracle
    assert "_MXFP4_MOE_AITER_BACKENDS" in mxfp4_moe_oracle
    assert "_MXFP4_MOE_GPT_OSS_TRITON_BACKENDS" in mxfp4_moe_oracle
    assert "FlashInfer TRTLLM MXFP4 MoE backend" in mxfp4_moe_oracle
    assert "Humming MXFP4 MoE backend" in mxfp4_moe_oracle
    assert "AITER MXFP4 MoE backend" in mxfp4_moe_oracle
    assert "GPT-OSS Triton MXFP4 MoE backend" in mxfp4_moe_oracle
    assert "Humming Mixed Precision kernels" in mxfp4_moe_oracle
    assert "ROCm-specific paths" in mxfp4_moe_oracle
    assert "triton_kernels MXFP4/SwiGLU" in mxfp4_moe_oracle
    assert "not native GB10 " in mxfp4_moe_oracle
    assert "GPT-OSS MXFP4 MoE correctness evidence" in mxfp4_moe_oracle
    assert "CPU fallback" in mxfp4_moe_oracle
    assert "not supported on GB10/SM12x" in mxfp4_moe_oracle
    assert "_gb10_mxfp8_moe_fallback_unsupported_reason" in mxfp8_moe_oracle
    assert "_MXFP8_MOE_FALLBACK_BACKENDS" in mxfp8_moe_oracle
    assert "MXFP8 MoE fallback backend" in mxfp8_moe_oracle
    assert "not supported on GB10/SM12x" in mxfp8_moe_oracle
    assert "_gb10_wna16_moe_fallback_unsupported_reason" in wna16_moe_oracle
    assert "_WNA16_MOE_FALLBACK_BACKENDS" in wna16_moe_oracle
    assert (
        "if reason := _gb10_wna16_moe_fallback_unsupported_reason(backend)"
        in wna16_moe_oracle
    )
    assert "WNA16 MoE fallback backend" in wna16_moe_oracle
    assert "not supported on GB10/SM12x" in wna16_moe_oracle
    assert "_gb10_fp8_moe_fallback_unsupported_reason" in fp8_moe_oracle
    assert "_gb10_aiter_fp8_moe_unsupported_reason" in fp8_moe_oracle
    assert "_gb10_deep_gemm_fp8_moe_unsupported_reason" in fp8_moe_oracle
    assert "_gb10_triton_fp8_moe_unsupported_reason" in fp8_moe_oracle
    assert "_gb10_vllm_cutlass_fp8_moe_unsupported_reason" in fp8_moe_oracle
    assert "_FP8_MOE_FALLBACK_BACKENDS" in fp8_moe_oracle
    assert "_FP8_MOE_DEEP_GEMM_BACKENDS" in fp8_moe_oracle
    assert "_FP8_MOE_TRITON_BACKENDS" in fp8_moe_oracle
    assert "_FP8_MOE_VLLM_CUTLASS_BACKENDS" in fp8_moe_oracle
    assert "FP8 MoE fallback backend" in fp8_moe_oracle
    assert "AITER FP8 MoE backend" in fp8_moe_oracle
    assert "DeepGEMM FP8 MoE backend" in fp8_moe_oracle
    assert "Triton FP8 MoE backend" in fp8_moe_oracle
    assert "vLLM CUTLASS FP8 MoE backend" in fp8_moe_oracle
    assert "ROCm-specific backend" in fp8_moe_oracle
    assert "Marlin and CPU W8A16 fallbacks" in fp8_moe_oracle
    assert "native GB10 DeepGEMM FP8 MoE" in fp8_moe_oracle
    assert "native GB10 Triton FP8 MoE" in fp8_moe_oracle
    assert "native GB10 vLLM CUTLASS FP8 MoE" in fp8_moe_oracle
    assert "not supported on GB10/SM12x" in fp8_moe_oracle
    assert "_gb10_int8_moe_triton_unsupported_reason" in int8_moe_oracle
    assert "Int8 MoE Triton fallback backend" in int8_moe_oracle
    assert "not supported on GB10/SM12x" in int8_moe_oracle
    assert "_gb10_aiter_unquantized_moe_unsupported_reason" in (
        unquantized_moe_oracle
    )
    assert "_gb10_unquantized_moe_triton_unsupported_reason" in (
        unquantized_moe_oracle
    )
    assert "_UNQUANTIZED_MOE_TRITON_FALLBACK_BACKENDS" in unquantized_moe_oracle
    assert "AITER unquantized MoE backend" in unquantized_moe_oracle
    assert "generic Triton unquantized MoE fallback backend" in (
        unquantized_moe_oracle
    )
    assert "TRITON" in unquantized_moe_oracle
    assert "BATCHED_TRITON" in unquantized_moe_oracle
    assert "ROCm-specific backend" in unquantized_moe_oracle
    assert "not supported on GB10/SM12x" in unquantized_moe_oracle
    assert "before publishing " in modelopt_quant
    assert "GB10 artifacts" in modelopt_quant


def test_gb10_deepseek_v4_deep_gemm_mega_moe_is_deferred_explicit_ep_only():
    deepseek_v4_model = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "nvidia" / "model.py"
    ).read_text()

    assert 'moe_backend == "deep_gemm_mega_moe"' in deepseek_v4_model
    assert "not vllm_config.parallel_config.enable_expert_parallel" in (
        deepseek_v4_model
    )
    assert "DeepSeek V4 MegaMoE currently requires expert parallel" in (
        deepseek_v4_model
    )
    assert "arch_major not in (10, 12)" in deepseek_v4_model
    assert "DeepGEMM MegaMoE requires SM100 or SM120-family GPUs" in (
        deepseek_v4_model
    )


def test_gb10_unquantized_moe_triton_fallback_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle import unquantized

    monkeypatch.setattr(
        unquantized,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    backend = unquantized.UnquantizedMoeBackend.TRITON
    reason = unquantized._gb10_unquantized_moe_triton_unsupported_reason(backend)
    assert reason is not None
    assert "generic Triton unquantized MoE fallback backend" in reason
    assert "native GB10 unquantized MoE correctness evidence" in reason

    for config in (
        SimpleNamespace(
            moe_backend="triton",
            is_lora_enabled=False,
            moe_parallel_config=SimpleNamespace(
                use_batched_activation_format=False,
                dp_size=1,
            ),
        ),
        SimpleNamespace(
            moe_backend="auto",
            is_lora_enabled=True,
            moe_parallel_config=SimpleNamespace(
                use_batched_activation_format=False,
                dp_size=1,
            ),
        ),
        SimpleNamespace(
            moe_backend="triton",
            is_lora_enabled=False,
            moe_parallel_config=SimpleNamespace(
                use_batched_activation_format=True,
                dp_size=1,
            ),
        ),
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            unquantized.select_unquantized_moe_backend(config)

        assert "generic Triton unquantized MoE fallback backend" in str(
            exc_info.value
        )

    monkeypatch.setattr(
        unquantized,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert unquantized._gb10_unquantized_moe_triton_unsupported_reason(backend) is None
    assert unquantized.map_unquantized_backend("triton") == backend


def test_gb10_wna16_moe_fallbacks_reject_sm12x(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle import int_wna16

    monkeypatch.setattr(int_wna16, "_is_sm12x_device", lambda: True)

    for backend in (
        int_wna16.WNA16MoEBackend.MARLIN,
        int_wna16.WNA16MoEBackend.BATCHED_MARLIN,
    ):
        reason = int_wna16._gb10_wna16_moe_fallback_unsupported_reason(backend)
        assert reason is not None
        assert backend.value in reason
        assert "not supported on GB10/SM12x" in reason
        assert "not native GB10 WNA16/MXINT MoE evidence" in reason

    assert (
        int_wna16._gb10_wna16_moe_fallback_unsupported_reason(
            int_wna16.WNA16MoEBackend.FLASHINFER_TRTLLM
        )
        is None
    )

    monkeypatch.setattr(int_wna16, "_is_sm12x_device", lambda: False)
    assert (
        int_wna16._gb10_wna16_moe_fallback_unsupported_reason(
            int_wna16.WNA16MoEBackend.MARLIN
        )
        is None
    )


def test_gb10_compressed_tensors_wna16_moe_rejects_legacy_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            return None

        def get_scheme_dict(self, layer, name):
            return {
                "weights": SimpleNamespace(
                    num_bits=4,
                    group_size=64,
                    strategy="group",
                    actorder=None,
                    symmetric=True,
                ),
                "input_activations": None,
                "format": CompressionFormat.pack_quantized.value,
            }

        def _is_mxfp4(self, weight_quant):
            return False

        def _is_mxfp8(self, weight_quant):
            return False

        def _is_wNa16_group_channel(self, weight_quant, input_quant):
            return True

    monkeypatch.setattr(compressed_tensors_moe, "_is_sm12x_device", lambda: True)
    monkeypatch.setattr(
        compressed_tensors_moe,
        "check_moe_marlin_supports_layer",
        lambda layer, group_size: False,
    )

    layer = SimpleNamespace(moe_config=SimpleNamespace())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe.CompressedTensorsMoEMethod.get_moe_method(
            FakeCompressedTensorsConfig(),
            layer,
            "model.layers.0.mlp.experts",
        )

    assert "CompressedTensors WNA16 MoE legacy fused-experts fallback" in str(
        exc_info.value
    )


def test_gb10_moe_wna16_rejects_legacy_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import moe_wna16

    class FakeRoutedExperts:
        def __init__(self):
            self.moe_config = SimpleNamespace()

    monkeypatch.setattr(moe_wna16, "_is_sm12x_device", lambda: True, raising=False)
    monkeypatch.setattr(moe_wna16, "RoutedExperts", FakeRoutedExperts)

    config = moe_wna16.MoeWNA16Config.from_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 64,
            "sym": True,
        }
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        config.get_quant_method(FakeRoutedExperts(), "model.layers.0.mlp.experts")

    assert "MoeWNA16 legacy fused-experts fallback" in str(exc_info.value)


def test_gb10_int8_moe_triton_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle import int8

    monkeypatch.setattr(int8, "_is_sm12x_device", lambda: True, raising=False)

    config = SimpleNamespace(
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )

    with pytest.raises(NotImplementedError, match="not supported on GB10/SM12x"):
        int8.select_int8_moe_backend(config)

    reason = int8._gb10_int8_moe_triton_unsupported_reason(
        int8.Int8MoeBackend.TRITON
    )
    assert reason is not None
    assert "Int8 MoE Triton fallback backend" in reason
    assert "not native GB10 Int8 MoE evidence" in reason

    monkeypatch.setattr(int8, "_is_sm12x_device", lambda: False, raising=False)
    assert (
        int8._gb10_int8_moe_triton_unsupported_reason(int8.Int8MoeBackend.TRITON)
        is None
    )


def test_gb10_compressed_tensors_w4a8_fp8_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors import utils
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW4A8Int,
    )

    monkeypatch.setattr(utils, "_is_sm12x_device", lambda: True, raising=False)

    weight_quant = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )

    assert not CompressedTensorsConfig._is_dynamic_token_w4a8_int(
        weight_quant, input_quant
    )
    reason = utils.gb10_compressed_tensors_w4a8_fp8_unsupported_reason(
        weight_quant, input_quant
    )
    assert reason is not None
    assert "CompressedTensors W4A8 FP8 checkpoint loading" in reason
    assert "not supported on GB10/SM12x" in reason
    assert "exact-SM90 CUTLASS W4A8" in reason

    config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format=CompressionFormat.float_quantized.value,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        config._get_scheme_from_parts(
            weight_quant,
            input_quant,
            format=CompressionFormat.float_quantized.value,
        )

    int_weight_quant = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    int_input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )
    assert CompressedTensorsConfig._is_dynamic_token_w4a8_int(
        int_weight_quant, int_input_quant
    )
    monkeypatch.setattr(utils, "_is_sm12x_device", lambda: False, raising=False)
    assert isinstance(
        config._get_scheme_from_parts(
            int_weight_quant,
            int_input_quant,
            format=CompressionFormat.float_quantized.value,
        ),
        CompressedTensorsW4A8Int,
    )


def test_gb10_compressed_tensors_w4a8_fp8_sm90_dense_rejects_first(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors,
        utils,
    )

    weight_quant = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )
    config = compressed_tensors.CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format=CompressionFormat.float_quantized.value,
    )

    constructed = []

    class ConstructedW4A8Fp8Scheme:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(utils, "_is_sm12x_device", lambda: True, raising=False)
    monkeypatch.setattr(
        compressed_tensors.CompressedTensorsConfig,
        "_is_fp8_w4a8_sm90",
        staticmethod(lambda weight_quant, input_quant: True),
    )
    monkeypatch.setattr(
        compressed_tensors,
        "CompressedTensorsW4A8Fp8",
        ConstructedW4A8Fp8Scheme,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        config._get_scheme_from_parts(
            weight_quant,
            input_quant,
            format=CompressionFormat.float_quantized.value,
        )

    assert not constructed
    assert "CompressedTensors W4A8 FP8 checkpoint loading" in str(exc_info.value)
    assert "exact-SM90 CUTLASS W4A8" in str(exc_info.value)


def test_gb10_compressed_tensors_w4a8_int_rejects_dense_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a8_int,
    )

    monkeypatch.setattr(
        compressed_tensors_w4a8_int,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_w4a8_int.CompressedTensorsW4A8Int(
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W4A8 Int dense loading" in reason
    assert "generic mixed-precision" in reason
    assert "not native GB10 W4A8 Int dense evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_w4a8_int,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_w4a8_int._gb10_w4a8_int_dense_unsupported_reason()
        is None
    )
    assert isinstance(
        compressed_tensors_w4a8_int.CompressedTensorsW4A8Int(
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
        ),
        compressed_tensors_w4a8_int.CompressedTensorsW4A8Int,
    )


def test_gb10_compressed_tensors_w8a8_mxfp8_rejects_dense_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_mxfp8,
    )

    monkeypatch.setattr(
        compressed_tensors_w8a8_mxfp8,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_w8a8_mxfp8.CompressedTensorsW8A8Mxfp8()

    reason = str(exc_info.value)
    assert "CompressedTensors W8A8 MXFP8 dense loading" in reason
    assert "MXFP8 dense kernel selection" in reason
    assert "native GB10 W8A8 MXFP8 dense correctness evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_w8a8_mxfp8,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_w8a8_mxfp8._gb10_w8a8_mxfp8_dense_unsupported_reason()
        is None
    )


def test_gb10_online_mxfp8_quantization_rejects_sm12x(monkeypatch):
    from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
    from vllm.model_executor.layers.quantization.online import base as online_base
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
        kMxfp8Dynamic,
    )

    monkeypatch.setattr(
        online_base,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    args = QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp8Dynamic),
        moe=QuantSpec(weight=kMxfp8Dynamic),
    )
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        online_base.OnlineQuantizationConfig(args)

    reason = str(exc_info.value)
    assert "Online MXFP8 quantization" in reason
    assert "FlashInfer CUTLASS MXFP8 dense" in reason
    assert "generic MXFP8 MoE backend selection" in reason
    assert "native GB10 online MXFP8 correctness evidence" in reason

    monkeypatch.setattr(
        online_base,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert online_base._gb10_online_mxfp8_quantization_unsupported_reason(args) is None
    assert isinstance(
        online_base.OnlineQuantizationConfig(
            QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticTensorSym))
        ),
        online_base.OnlineQuantizationConfig,
    )


def test_gb10_online_mxfp4_quantization_rejects_sm12x(monkeypatch):
    from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
    from vllm.model_executor.layers.quantization.online import base as online_base
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
        kMxfp4Dynamic,
    )

    monkeypatch.setattr(
        online_base,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for args in (
        QuantizationConfigArgs(linear=QuantSpec(weight=kMxfp4Dynamic)),
        QuantizationConfigArgs(moe=QuantSpec(weight=kMxfp4Dynamic)),
        QuantizationConfigArgs(linear="mxfp4"),
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            online_base.OnlineQuantizationConfig(args)

        reason = str(exc_info.value)
        assert "Online MXFP4 quantization" in reason
        assert "weight='mxfp4'" in reason
        assert "no online MXFP4 method is wired" in reason
        assert "native GB10 online MXFP4 correctness evidence" in reason

    monkeypatch.setattr(
        online_base,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    args = QuantizationConfigArgs(linear=QuantSpec(weight=kMxfp4Dynamic))
    assert online_base._gb10_online_mxfp4_quantization_unsupported_reason(args) is None
    assert isinstance(
        online_base.OnlineQuantizationConfig(
            QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticTensorSym))
        ),
        online_base.OnlineQuantizationConfig,
    )


def test_gb10_online_fp8_quantization_rejects_sm12x(monkeypatch):
    from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
    from vllm.model_executor.layers.quantization.online import base as online_base
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Static128BlockSym,
        kFp8StaticTensorSym,
        kInt8StaticChannelSym,
    )

    monkeypatch.setattr(
        online_base,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    args = QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticTensorSym),
        moe=QuantSpec(weight=kFp8Static128BlockSym),
    )
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        online_base.OnlineQuantizationConfig(args)

    reason = str(exc_info.value)
    assert "Online FP8 quantization" in reason
    assert "FP8 scaled-mm dense kernels" in reason
    assert "generic FP8 MoE backend selection" in reason
    assert "native GB10 online FP8 correctness evidence" in reason

    monkeypatch.setattr(
        online_base,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert online_base._gb10_online_fp8_quantization_unsupported_reason(args) is None
    assert isinstance(
        online_base.OnlineQuantizationConfig(
            QuantizationConfigArgs(moe=QuantSpec(weight=kInt8StaticChannelSym))
        ),
        online_base.OnlineQuantizationConfig,
    )


def test_gb10_online_int8_moe_quantization_rejects_sm12x(monkeypatch):
    from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
    from vllm.model_executor.layers.quantization.online import base as online_base
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8StaticChannelSym,
        kMxfp8Dynamic,
    )

    monkeypatch.setattr(
        online_base,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    args = QuantizationConfigArgs(moe=QuantSpec(weight=kInt8StaticChannelSym))
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        online_base.OnlineQuantizationConfig(args)

    reason = str(exc_info.value)
    assert "Online Int8 MoE quantization" in reason
    assert "int8_per_channel_weight_only" in reason
    assert "Int8 MoE backend selection" in reason
    assert "native GB10 online Int8 MoE correctness evidence" in reason

    monkeypatch.setattr(
        online_base,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        online_base._gb10_online_int8_moe_quantization_unsupported_reason(args)
        is None
    )
    assert isinstance(
        online_base.OnlineQuantizationConfig(
            QuantizationConfigArgs(moe=QuantSpec(weight=kMxfp8Dynamic))
        ),
        online_base.OnlineQuantizationConfig,
    )


def test_gb10_fbgemm_fp8_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import fbgemm_fp8

    monkeypatch.setattr(
        fbgemm_fp8,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        fbgemm_fp8.FBGEMMFp8Config(ignore_list=[], input_scale_ub=1200.0)

    reason = str(exc_info.value)
    assert "FBGEMM FP8 quantization" in reason
    assert "deprecated public quantization method" in reason
    assert "generic FP8 linear kernel selection" in reason
    assert "native GB10 FBGEMM FP8 correctness evidence" in reason

    monkeypatch.setattr(
        fbgemm_fp8,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert fbgemm_fp8._gb10_fbgemm_fp8_quantization_unsupported_reason() is None
    assert isinstance(
        fbgemm_fp8.FBGEMMFp8Config(ignore_list=[], input_scale_ub=1200.0),
        fbgemm_fp8.FBGEMMFp8Config,
    )


def test_gb10_fp_quant_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import fp_quant

    monkeypatch.setattr(
        fp_quant,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for forward_dtype in ("mxfp4", "nvfp4"):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            fp_quant.FPQuantConfig(
                hadamard_group_size=32,
                forward_dtype=forward_dtype,
                forward_method="abs_max",
            )

        reason = str(exc_info.value)
        assert "FPQuant FP4 quantization" in reason
        assert "deprecated public quantization method" in reason
        assert "MXFP4/NVFP4 FPQuant linear kernels" in reason
        assert "native GB10 FPQuant FP4 correctness evidence" in reason

    monkeypatch.setattr(
        fp_quant,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert fp_quant._gb10_fp_quant_fp4_quantization_unsupported_reason() is None
    assert isinstance(
        fp_quant.FPQuantConfig(
            hadamard_group_size=32,
            forward_dtype="mxfp4",
            forward_method="abs_max",
        ),
        fp_quant.FPQuantConfig,
    )


def test_gb10_experts_int8_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import experts_int8

    monkeypatch.setattr(
        experts_int8,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        experts_int8.ExpertsInt8Config()

    reason = str(exc_info.value)
    assert "ExpertsInt8 quantization" in reason
    assert "backward-compatible public quantization method" in reason
    assert "online Int8 MoE backend selection" in reason
    assert "native GB10 online Int8 MoE correctness evidence" in reason

    monkeypatch.setattr(
        experts_int8,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        experts_int8._gb10_experts_int8_quantization_unsupported_reason()
        is None
    )
    assert isinstance(
        experts_int8.ExpertsInt8Config(),
        experts_int8.ExpertsInt8Config,
    )


def test_gb10_public_mxfp4_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import mxfp4

    monkeypatch.setattr(
        mxfp4,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for config_cls in (mxfp4.Mxfp4Config, mxfp4.GptOssMxfp4Config):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            config_cls()

        reason = str(exc_info.value)
        assert "Public MXFP4 quantization" in reason
        assert "mxfp4 and gpt_oss_mxfp4" in reason
        assert "unquantized linear/attention handling" in reason
        assert "MXFP4 MoE backend selection" in reason
        assert "native GB10 public MXFP4 correctness evidence" in reason

    monkeypatch.setattr(
        mxfp4,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert mxfp4._gb10_public_mxfp4_quantization_unsupported_reason() is None
    assert isinstance(mxfp4.Mxfp4Config(), mxfp4.Mxfp4Config)
    assert isinstance(mxfp4.GptOssMxfp4Config(), mxfp4.GptOssMxfp4Config)


def test_gb10_public_fp8_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import fp8

    monkeypatch.setattr(
        fp8,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for kwargs in (
        {},
        {
            "is_checkpoint_fp8_serialized": True,
            "activation_scheme": "static",
        },
        {
            "is_checkpoint_fp8_serialized": True,
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            fp8.Fp8Config(**kwargs)

        reason = str(exc_info.value)
        assert "Public FP8 quantization" in reason
        assert "fp8 quantization method" in reason
        assert "FP8 scaled-mm dense kernel selection" in reason
        assert "FP8 MoE backend selection" in reason
        assert "online FP8 quantization path" in reason
        assert "native GB10 public FP8 correctness evidence" in reason

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        fp8.Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
            }
        )

    monkeypatch.setattr(
        fp8,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert fp8._gb10_public_fp8_quantization_unsupported_reason() is None
    assert isinstance(fp8.Fp8Config(), fp8.Fp8Config)


def test_gb10_deepseek_v4_fp8_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import fp8
    from vllm.models.deepseek_v4 import quant_config

    monkeypatch.setattr(
        quant_config,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        quant_config.DeepseekV4FP8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

    reason = str(exc_info.value)
    assert "DeepSeek V4 FP8 quantization" in reason
    assert "deepseek_v4_fp8 quantization method" in reason
    assert "FP8 block-quantized linear/attention layers" in reason
    assert "FP8, MXFP4, or ModelOpt NVFP4 MoE dispatch" in reason
    assert "native GB10 DeepSeek V4 correctness evidence" in reason
    assert "Public FP8 quantization" not in reason

    assert quant_config.DeepseekV4FP8Config.override_quantization_method(
        {"quant_method": "fp8"},
        None,
        type("DeepseekV4HfConfig", (), {"model_type": "deepseek_v4"})(),
    ) == "deepseek_v4_fp8"
    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        quant_config.DeepseekV4FP8Config.from_config(
            {
                "quant_method": "deepseek_v4_fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            }
        )

    monkeypatch.setattr(
        quant_config,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        fp8,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert quant_config._gb10_deepseek_v4_fp8_quantization_unsupported_reason() is None
    assert isinstance(
        quant_config.DeepseekV4FP8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        ),
        quant_config.DeepseekV4FP8Config,
    )


def test_gb10_torchao_fp8_activation_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import torchao

    class Float8DynamicActivationFloat8WeightConfig:
        pass

    class Int8WeightOnlyConfig:
        pass

    monkeypatch.setattr(
        torchao,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    fp8_config = Float8DynamicActivationFloat8WeightConfig()
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        torchao.TorchAOConfig(fp8_config)

    reason = str(exc_info.value)
    assert "TorchAO FP8 activation quantization" in reason
    assert "torchao quantization method" in reason
    assert "Float8DynamicActivationFloat8WeightConfig" in reason
    assert "torchao.quantization.quantize_" in reason
    assert "convert_to_packed_tensor_based_on_current_hardware" in reason
    assert "native GB10 TorchAO FP8 activation correctness evidence" in reason

    assert torchao._gb10_torchao_fp8_activation_unsupported_reason(
        Int8WeightOnlyConfig()
    ) is None

    monkeypatch.setattr(
        torchao,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert torchao._gb10_torchao_fp8_activation_unsupported_reason(fp8_config) is None
    assert isinstance(torchao.TorchAOConfig(fp8_config), torchao.TorchAOConfig)


def test_gb10_torchao_weight_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import torchao

    class Float8DynamicActivationFloat8WeightConfig:
        pass

    class Float8WeightOnlyConfig:
        pass

    class Int4WeightOnlyConfig:
        pass

    class Int8WeightOnlyConfig:
        pass

    monkeypatch.setattr(
        torchao,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for weight_config in (
        Float8WeightOnlyConfig(),
        Int4WeightOnlyConfig(),
        Int8WeightOnlyConfig(),
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            torchao.TorchAOConfig(weight_config)

        reason = str(exc_info.value)
        assert "TorchAO weight quantization" in reason
        assert type(weight_config).__name__ in reason
        assert "torchao.quantization.quantize_" in reason
        assert "convert_to_packed_tensor_based_on_current_hardware" in reason
        assert "native GB10 TorchAO weight correctness evidence" in reason

    fp8_activation_config = Float8DynamicActivationFloat8WeightConfig()
    assert (
        torchao._gb10_torchao_weight_quantization_unsupported_reason(
            fp8_activation_config
        )
        is None
    )

    monkeypatch.setattr(
        torchao,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        torchao._gb10_torchao_weight_quantization_unsupported_reason(
            Int8WeightOnlyConfig()
        )
        is None
    )
    assert isinstance(
        torchao.TorchAOConfig(Int8WeightOnlyConfig()),
        torchao.TorchAOConfig,
    )


def test_gb10_bitsandbytes_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import bitsandbytes

    monkeypatch.setattr(
        bitsandbytes,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for kwargs in (
        {},
        {"load_in_4bit": True},
        {"load_in_8bit": True, "load_in_4bit": False},
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            bitsandbytes.BitsAndBytesConfig(**kwargs)

        reason = str(exc_info.value)
        assert "BitsAndBytes quantization" in reason
        assert "bitsandbytes quantization method" in reason
        assert "bitsandbytes 4-bit linear kernels" in reason
        assert "bitsandbytes 8-bit matmul kernels" in reason
        assert "BitsAndBytesMoEMethod" in reason
        assert "native GB10 BitsAndBytes correctness evidence" in reason

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        bitsandbytes.BitsAndBytesConfig.from_config(
            {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            }
        )

    monkeypatch.setattr(
        bitsandbytes,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert bitsandbytes._gb10_bitsandbytes_quantization_unsupported_reason() is None
    assert isinstance(
        bitsandbytes.BitsAndBytesConfig(),
        bitsandbytes.BitsAndBytesConfig,
    )
    assert isinstance(
        bitsandbytes.BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False),
        bitsandbytes.BitsAndBytesConfig,
    )


def test_gb10_awq_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import awq, awq_marlin

    monkeypatch.setattr(
        awq,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for construct_config in (
        lambda: awq.AWQConfig(weight_bits=4, group_size=128, zero_point=True),
        lambda: awq.AWQConfig.from_config(
            {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            }
        ),
        lambda: awq_marlin.AWQMarlinConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=[],
            full_config={
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            },
        ),
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            construct_config()

        reason = str(exc_info.value)
        assert "AWQ quantization" in reason
        assert "awq and awq_marlin quantization methods" in reason
        assert "AWQLinearMethod" in reason
        assert "AWQMarlinLinearMethod" in reason
        assert "AWQMarlinMoEMethod" in reason
        assert "MoeWNA16Config" in reason
        assert "native GB10 AWQ correctness evidence" in reason

    monkeypatch.setattr(
        awq,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert awq._gb10_awq_quantization_unsupported_reason() is None
    assert isinstance(
        awq.AWQConfig(weight_bits=4, group_size=128, zero_point=True),
        awq.AWQConfig,
    )


def test_gb10_gptq_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import auto_gptq

    monkeypatch.setattr(
        auto_gptq,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    config = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "lm_head": False,
    }

    for construct_config in (
        lambda: auto_gptq.AutoGPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config=config,
        ),
        lambda: auto_gptq.AutoGPTQConfig.from_config(config),
        lambda: auto_gptq.AutoGPTQConfig(
            weight_bits=8,
            group_size=128,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={**config, "bits": 8},
        ),
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            construct_config()

        reason = str(exc_info.value)
        assert "GPTQ quantization" in reason
        assert "auto_gptq, gptq, and gptq_marlin quantization methods" in reason
        assert "get_linear_quant_method" in reason
        assert "choose_mp_linear_kernel" in reason
        assert "AutoGPTQMoEMethod" in reason
        assert "MoeWNA16Config" in reason
        assert "native GB10 GPTQ correctness evidence" in reason

    monkeypatch.setattr(
        auto_gptq,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert auto_gptq._gb10_gptq_quantization_unsupported_reason() is None
    assert isinstance(
        auto_gptq.AutoGPTQConfig.from_config(config),
        auto_gptq.AutoGPTQConfig,
    )


def test_gb10_inc_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import inc

    monkeypatch.setattr(
        inc,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    configs = (
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "packing_format": "auto_round:auto_gptq",
            "backend": "auto",
        },
        {
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "packing_format": "auto_round:auto_awq",
            "backend": "awq:marlin",
        },
    )

    for config in configs:
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            inc.INCConfig.from_config(config)

        reason = str(exc_info.value)
        assert "INC/AutoRound quantization" in reason
        assert "inc and auto-round quantization methods" in reason
        assert "apply_awq_quant_layer" in reason
        assert "apply_gptq_quant_layer" in reason
        assert "AWQMarlinLinearMethod" in reason
        assert "AutoGPTQLinearMethod" in reason
        assert "MoeWNA16Config" in reason
        assert "native GB10 INC/AutoRound correctness evidence" in reason

    monkeypatch.setattr(
        inc,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert inc._gb10_inc_quantization_unsupported_reason() is None
    assert isinstance(
        inc.INCConfig.from_config(configs[0]),
        inc.INCConfig,
    )


def test_gb10_gguf_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import gguf

    monkeypatch.setattr(
        gguf,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for construct_config in (
        lambda: gguf.GGUFConfig(),
        lambda: gguf.GGUFConfig.from_config({}),
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            construct_config()

        reason = str(exc_info.value)
        assert "GGUF quantization" in reason
        assert "gguf quantization method" in reason
        assert "GGUFLinearMethod" in reason
        assert "GGUFEmbeddingMethod" in reason
        assert "GGUFMoEMethod" in reason
        assert "ggml_mul_mat_vec_a8" in reason
        assert "ggml_mul_mat_a8" in reason
        assert "ggml_dequantize" in reason
        assert "native GB10 GGUF correctness evidence" in reason

    monkeypatch.setattr(
        gguf,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert gguf._gb10_gguf_quantization_unsupported_reason() is None
    assert isinstance(gguf.GGUFConfig.from_config({}), gguf.GGUFConfig)


def test_gb10_humming_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import humming

    monkeypatch.setattr(
        humming,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for construct_config in (
        lambda: humming.HummingConfig(),
        lambda: humming.HummingConfig.from_config({"quant_method": "humming"}),
    ):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            construct_config()

        reason = str(exc_info.value)
        assert "Humming quantization" in reason
        assert "humming quantization method" in reason
        assert "HummingLinearMethod" in reason
        assert "HummingMoEMethod" in reason
        assert "HummingMethod.prepare_layer_meta" in reason
        assert "HummingMethod.transform_humming_layer" in reason
        assert "HummingMethod.forward_layer" in reason
        assert "native GB10 Humming correctness evidence" in reason

    monkeypatch.setattr(
        humming,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert humming._gb10_humming_quantization_unsupported_reason() is None
    assert isinstance(
        humming.HummingConfig.from_config({"quant_method": "humming"}),
        humming.HummingConfig,
    )


def test_gb10_humming_mxfp4_moe_backend_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4

    monkeypatch.setattr(
        mxfp4,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    backend = mxfp4.Mxfp4MoeBackend.HUMMING
    reason = mxfp4._gb10_unsupported_backend_reason(backend)
    assert reason is not None
    assert "Humming MXFP4 MoE backend" in reason
    assert "Humming Mixed Precision kernels" in reason
    assert "native GB10 Humming MXFP4 MoE correctness evidence" in reason

    config = SimpleNamespace(
        moe_backend="humming",
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )
    monkeypatch.setattr(
        mxfp4,
        "_resolve_activation_key",
        lambda activation_key: None,
        raising=False,
    )
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        mxfp4.select_mxfp4_moe_backend(config)

    assert "Humming MXFP4 MoE backend" in str(exc_info.value)

    monkeypatch.setattr(
        mxfp4,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert mxfp4._gb10_unsupported_backend_reason(backend) is None
    assert mxfp4.map_mxfp4_backend("humming") == [backend]


def test_gb10_quark_w4a8_fp8_moe_loading_rejects_sm12x(monkeypatch):
    import torch

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.quantization.quark import quark_moe, utils

    monkeypatch.setattr(
        utils.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
        raising=False,
    )
    monkeypatch.setattr(
        quark_moe.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: True,
        raising=False,
    )

    moe_config = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size_per_partition=256,
        num_local_experts=4,
        num_logical_experts=4,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.float16,
    )

    weight_config = [
        {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
        {
            "dtype": "int4",
            "qscheme": "per_channel",
            "is_dynamic": False,
            "symmetric": True,
            "ch_axis": 0,
        },
    ]
    input_config = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
    }

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
        exc_info
    ):
        quark_moe.QuarkW4A8Fp8MoEMethod(weight_config, input_config, moe_config)

    reason = str(exc_info.value)
    assert "Quark W4A8 FP8 MoE checkpoint loading" in reason
    assert "ROCm AITER fused MoE support" in reason
    assert "native GB10 W4A8 FP8 MoE correctness evidence" in reason

    monkeypatch.setattr(
        utils.current_platform,
        "is_device_capability_family",
        lambda family: False,
        raising=False,
    )
    assert utils.gb10_quark_w4a8_fp8_moe_unsupported_reason() is None


def test_gb10_quark_w8a8_checkpoint_loading_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization.quark import quark, utils

    monkeypatch.setattr(
        utils.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
        raising=False,
    )

    config = quark.QuarkConfig(
        {
            "global_quant_config": {},
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "exclude": [],
        }
    )
    monkeypatch.setattr(
        config,
        "_check_scheme_supported",
        lambda min_capability, error=True: True,
        raising=False,
    )

    fp8_w8a8_config = {
        "weight": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
        "output_tensors": None,
        "bias": None,
    }
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
        exc_info
    ):
        config._get_scheme_from_config(fp8_w8a8_config)

    reason = str(exc_info.value)
    assert "Quark W8A8 FP8 checkpoint loading" in reason
    assert "FP8 scaled-mm dense kernel selection" in reason
    assert "native GB10 W8A8 FP8 dense correctness evidence" in reason

    int8_w8a8_static_config = {
        "weight": {
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": False,
            "symmetric": True,
        },
        "input_tensors": {
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
        "output_tensors": None,
        "bias": None,
    }
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
        exc_info
    ):
        config._get_scheme_from_config(int8_w8a8_static_config)

    reason = str(exc_info.value)
    assert "Quark W8A8 Int8 checkpoint loading" in reason
    assert "Int8 scaled-mm dense kernel selection" in reason
    assert "native GB10 W8A8 Int8 dense correctness evidence" in reason

    int8_w8a8_dynamic_config = copy.deepcopy(int8_w8a8_static_config)
    int8_w8a8_dynamic_config["input_tensors"]["qscheme"] = "per_channel"
    int8_w8a8_dynamic_config["input_tensors"]["is_dynamic"] = True
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
        exc_info
    ):
        config._get_scheme_from_config(int8_w8a8_dynamic_config)
    assert "Quark W8A8 Int8 checkpoint loading" in str(exc_info.value)

    monkeypatch.setattr(
        utils.current_platform,
        "is_device_capability_family",
        lambda family: False,
        raising=False,
    )
    assert utils.gb10_quark_w8a8_fp8_unsupported_reason() is None
    assert utils.gb10_quark_w8a8_int8_unsupported_reason() is None


def test_gb10_quark_w8a8_moe_loading_rejects_sm12x(monkeypatch):
    import torch

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.quantization.quark import quark_moe, utils

    monkeypatch.setattr(
        utils.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
        raising=False,
    )

    moe_config = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size_per_partition=256,
        num_local_experts=4,
        num_logical_experts=4,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.float16,
    )

    fp8_weight_config = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
    }
    fp8_input_config = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
    }
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
        exc_info
    ):
        quark_moe.QuarkW8A8Fp8MoEMethod(
            fp8_weight_config, fp8_input_config, moe_config
        )

    reason = str(exc_info.value)
    assert "Quark W8A8 FP8 MoE checkpoint loading" in reason
    assert "generic FP8 W8A8 MoE backend selection" in reason
    assert "native GB10 W8A8 FP8 MoE correctness evidence" in reason

    int8_weight_config = {
        "dtype": "int8",
        "qscheme": "per_channel",
        "is_dynamic": False,
        "symmetric": True,
    }
    int8_input_config = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
        exc_info
    ):
        quark_moe.QuarkW8A8Int8MoEMethod(
            int8_weight_config, int8_input_config, moe_config
        )

    reason = str(exc_info.value)
    assert "Quark W8A8 Int8 MoE checkpoint loading" in reason
    assert "generic Int8 W8A8 MoE backend selection" in reason
    assert "native GB10 W8A8 Int8 MoE correctness evidence" in reason

    monkeypatch.setattr(
        utils.current_platform,
        "is_device_capability_family",
        lambda family: False,
        raising=False,
    )
    assert utils.gb10_quark_w8a8_fp8_moe_unsupported_reason() is None
    assert utils.gb10_quark_w8a8_int8_moe_unsupported_reason() is None


def test_gb10_modelopt_fp8_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import modelopt

    monkeypatch.setattr(
        modelopt,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for quant_method in ("FP8", "FP8_PER_CHANNEL_PER_TOKEN", "FP8_PB_WO"):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            modelopt.ModelOptFp8Config(
                quant_method=quant_method,
                is_checkpoint_fp8_serialized=True,
                kv_cache_quant_method=None,
                exclude_modules=[],
            )

        reason = str(exc_info.value)
        assert "ModelOpt FP8 quantization" in reason
        assert quant_method in reason
        assert "FP8 dense kernel selection" in reason
        assert "FP8 MoE backend selection" in reason
        assert "native GB10 ModelOpt FP8 correctness evidence" in reason

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptFp8Config.from_config(
            {
                "quantization": {
                    "quant_algo": "FP8",
                    "kv_cache_quant_algo": None,
                    "exclude_modules": [],
                },
            }
        )

    monkeypatch.setattr(
        modelopt,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert modelopt._gb10_modelopt_fp8_quantization_unsupported_reason() is None
    assert isinstance(
        modelopt.ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        ),
        modelopt.ModelOptFp8Config,
    )


def test_gb10_modelopt_mxfp8_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import modelopt

    monkeypatch.setattr(
        modelopt,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        modelopt.ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )

    reason = str(exc_info.value)
    assert "ModelOpt MXFP8 quantization" in reason
    assert "serialized checkpoint" in reason
    assert "MXFP8 dense kernel selection" in reason
    assert "MXFP8 MoE backend selection" in reason
    assert "native GB10 ModelOpt MXFP8 correctness evidence" in reason

    monkeypatch.setattr(
        modelopt,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert modelopt._gb10_modelopt_mxfp8_quantization_unsupported_reason() is None
    assert isinstance(
        modelopt.ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        modelopt.ModelOptMxFp8Config,
    )


def test_gb10_modelopt_mixed_quantization_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import modelopt

    hf_quant_config = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "group_size": 16,
            "quantized_layers": {
                "model.layers.0.fp8_proj": {"quant_algo": "FP8"},
                "model.layers.0.nvfp4_proj": {"quant_algo": "NVFP4"},
                "model.layers.0.w4a16_proj": {"quant_algo": "W4A16_NVFP4"},
            },
        },
    }

    monkeypatch.setattr(
        modelopt,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        modelopt.ModelOptMixedPrecisionConfig.from_config(hf_quant_config)

    reason = str(exc_info.value)
    assert "ModelOpt mixed precision quantization" in reason
    assert "MIXED_PRECISION" in reason
    assert "FP8 dense or MoE selection" in reason
    assert "NVFP4 dense or MoE selection" in reason
    assert "W4A16 NVFP4 fallback selection" in reason
    assert "native GB10 ModelOpt mixed precision correctness evidence" in reason

    monkeypatch.setattr(
        modelopt,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert modelopt._gb10_modelopt_mixed_quantization_unsupported_reason() is None
    assert isinstance(
        modelopt.ModelOptMixedPrecisionConfig.from_config(hf_quant_config),
        modelopt.ModelOptMixedPrecisionConfig,
    )


def test_gb10_modelopt_nvfp4_kv_cache_loading_rejects_sm12x(monkeypatch):
    from vllm.model_executor.layers.quantization import modelopt

    monkeypatch.setattr(
        modelopt,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    for kv_cache_quant_algo in ("NVFP4", "nvfp4"):
        with pytest.raises(ValueError, match="not supported on GB10/SM12x") as (
            exc_info
        ):
            modelopt.ModelOptNvFp4Config.from_config(
                {
                    "quantization": {
                        "quant_algo": "NVFP4",
                        "kv_cache_quant_algo": kv_cache_quant_algo,
                        "exclude_modules": [],
                        "group_size": 16,
                    },
                }
            )

        reason = str(exc_info.value)
        assert "ModelOpt NVFP4 KV-cache loading" in reason
        assert "kv_cache_quant_algo=NVFP4" in reason
        assert "kv_cache_dtype='nvfp4'" in reason
        assert "FP8 E4M3 KV cache" in reason
        assert "native SM12x NVFP4 KV-cache correctness evidence" in reason

    with pytest.raises(ValueError, match="not supported on GB10/SM12x"):
        modelopt.ModelOptNvFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "quant_method": "modelopt_fp4",
                "kv_cache_scheme": {
                    "type": "float",
                    "num_bits": 4,
                },
                "ignore": [],
                "group_size": 16,
            }
        )

    monkeypatch.setattr(
        modelopt,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert modelopt._gb10_modelopt_nvfp4_kv_cache_unsupported_reason("NVFP4") is None
    config = modelopt.ModelOptNvFp4Config.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": "NVFP4",
                "exclude_modules": [],
                "group_size": 16,
            },
        }
    )
    assert isinstance(config, modelopt.ModelOptNvFp4Config)
    assert config.kv_cache_quant_algo == "NVFP4"


def test_gb10_compressed_tensors_wna16_dense_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16,
    )

    monkeypatch.setattr(
        compressed_tensors_wNa16,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    weight_quant = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format=CompressionFormat.pack_quantized.value,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        config._get_scheme_from_parts(
            weight_quant,
            None,
            format=CompressionFormat.pack_quantized.value,
        )

    reason = str(exc_info.value)
    assert "CompressedTensors WNA16 dense loading" in reason
    assert "generic mixed-precision" in reason
    assert "not native GB10 WNA16/MXINT dense evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_wNa16,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert compressed_tensors_wNa16._gb10_wna16_dense_unsupported_reason() is None
    assert isinstance(
        config._get_scheme_from_parts(
            weight_quant,
            None,
            format=CompressionFormat.pack_quantized.value,
        ),
        compressed_tensors_wNa16.CompressedTensorsWNA16,
    )


def test_gb10_compressed_tensors_w8a16_fp8_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a16_fp8,
    )

    monkeypatch.setattr(
        compressed_tensors_w8a16_fp8,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format=CompressionFormat.float_quantized.value,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        config._get_scheme_from_parts(
            weight_quant,
            None,
            format=CompressionFormat.float_quantized.value,
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W8A16 FP8 loading" in reason
    assert "FP8 W8A16 Marlin fallback" in reason
    assert "not native GB10 FP8 W8A16 dense evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_w8a16_fp8,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_w8a16_fp8._gb10_w8a16_fp8_loading_unsupported_reason()
        is None
    )


def test_gb10_compressed_tensors_w8a8_fp8_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_fp8,
    )

    monkeypatch.setattr(
        CompressedTensorsConfig,
        "_check_scheme_supported",
        classmethod(
            lambda cls, min_capability, error=True, match_exact=False: True
        ),
    )
    monkeypatch.setattr(
        compressed_tensors_w8a8_fp8,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format=CompressionFormat.float_quantized.value,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        config._get_scheme_from_parts(
            weight_quant,
            input_quant,
            format=CompressionFormat.float_quantized.value,
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W8A8 FP8 loading" in reason
    assert "scaled-mm W8A8 FP8 kernels" in reason
    assert "not native GB10 W8A8 FP8 dense evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_w8a8_fp8,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_w8a8_fp8._gb10_w8a8_fp8_loading_unsupported_reason()
        is None
    )


def test_gb10_compressed_tensors_w8a8_int_dense_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_int8,
    )

    monkeypatch.setattr(
        compressed_tensors_w8a8_int8,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format=CompressionFormat.int_quantized.value,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        config._get_scheme_from_parts(
            weight_quant,
            input_quant,
            format=CompressionFormat.int_quantized.value,
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W8A8 Int dense loading" in reason
    assert "Cutlass/Triton W8A8 Int8 scaled-mm kernels" in reason
    assert "native GB10 W8A8 Int8 dense correctness evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_w8a8_int8,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_w8a8_int8._gb10_w8a8_int_dense_unsupported_reason()
        is None
    )


def test_gb10_compressed_tensors_w4a8_fp8_moe_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors import utils
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            return None

        @staticmethod
        def _is_mxfp4(weight_quant):
            return False

        @staticmethod
        def _is_mxfp8(weight_quant):
            return False

        @staticmethod
        def _is_wNa16_group_channel(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_nvfp4_format(quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm90(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm100(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_dynamic_token_w8a8(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w4a8_sm90(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_dynamic_token_w4a8_int(weight_quant, input_quant):
            return True

        def get_scheme_dict(self, layer, name):
            return {
                "weights": QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.GROUP,
                    symmetric=True,
                    dynamic=False,
                    group_size=128,
                ),
                "input_activations": QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TOKEN,
                    symmetric=True,
                    dynamic=True,
                ),
                "format": CompressionFormat.float_quantized.value,
            }

    monkeypatch.setattr(utils, "_is_sm12x_device", lambda: True, raising=False)
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe.CompressedTensorsMoEMethod.get_moe_method(
            FakeCompressedTensorsConfig(),
            layer,
            "model.layers.0.mlp.experts",
        )

    assert "CompressedTensors W4A8 FP8 checkpoint loading" in str(exc_info.value)
    assert "exact-SM90 CUTLASS W4A8" in str(exc_info.value)


def test_gb10_compressed_tensors_w4a8_fp8_sm90_moe_rejects_first(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors import utils
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe,
        compressed_tensors_moe_w4a8_fp8,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            return None

        @staticmethod
        def _is_mxfp4(weight_quant):
            return False

        @staticmethod
        def _is_mxfp8(weight_quant):
            return False

        @staticmethod
        def _is_wNa16_group_channel(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_nvfp4_format(quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm90(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm100(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_dynamic_token_w8a8(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w4a8_sm90(weight_quant, input_quant):
            return True

        @staticmethod
        def _is_dynamic_token_w4a8_int(weight_quant, input_quant):
            return False

        def get_scheme_dict(self, layer, name):
            return {
                "weights": QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.GROUP,
                    symmetric=True,
                    dynamic=False,
                    group_size=128,
                ),
                "input_activations": QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TOKEN,
                    symmetric=True,
                    dynamic=True,
                ),
                "format": CompressionFormat.float_quantized.value,
            }

    constructed = []

    class ConstructedW4A8Fp8MoEMethod:
        def __init__(self, *args):
            constructed.append(args)

    monkeypatch.setattr(utils, "_is_sm12x_device", lambda: True, raising=False)
    monkeypatch.setattr(
        compressed_tensors_moe_w4a8_fp8,
        "CompressedTensorsW4A8Fp8MoEMethod",
        ConstructedW4A8Fp8MoEMethod,
    )
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe.CompressedTensorsMoEMethod.get_moe_method(
            FakeCompressedTensorsConfig(),
            layer,
            "model.layers.0.mlp.experts",
        )

    assert not constructed
    assert "CompressedTensors W4A8 FP8 checkpoint loading" in str(exc_info.value)
    assert "exact-SM90 CUTLASS W4A8" in str(exc_info.value)


def test_gb10_compressed_tensors_w8a8_fp8_moe_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            return None

        @staticmethod
        def _is_mxfp4(weight_quant):
            return False

        @staticmethod
        def _is_mxfp8(weight_quant):
            return False

        @staticmethod
        def _is_wNa16_group_channel(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_nvfp4_format(quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm90(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm100(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8(weight_quant, input_quant):
            return True

        def get_scheme_dict(self, layer, name):
            return {
                "weights": QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
                "input_activations": QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
                "format": CompressionFormat.float_quantized.value,
            }

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe.CompressedTensorsMoEMethod.get_moe_method(
            FakeCompressedTensorsConfig(),
            layer,
            "model.layers.0.mlp.experts",
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W8A8 FP8 MoE loading" in reason
    assert "generic FP8 W8A8 MoE backend selection" in reason
    assert "native GB10 W8A8 FP8 MoE correctness evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_moe._gb10_w8a8_fp8_moe_loading_unsupported_reason()
        is None
    )


def test_gb10_compressed_tensors_w8a8_mxfp8_moe_rejects_sm12x(monkeypatch):
    import torch
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            return None

        @staticmethod
        def _is_mxfp4(weight_quant):
            return False

        @staticmethod
        def _is_mxfp8(weight_quant):
            return True

        def get_scheme_dict(self, layer, name):
            return {
                "weights": QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.GROUP,
                    symmetric=True,
                    dynamic=False,
                    group_size=32,
                    scale_dtype=torch.uint8,
                ),
                "input_activations": None,
                "format": CompressionFormat.float_quantized.value,
            }

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe.CompressedTensorsMoEMethod.get_moe_method(
            FakeCompressedTensorsConfig(),
            layer,
            "model.layers.0.mlp.experts",
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W8A8 MXFP8 MoE loading" in reason
    assert "generic MXFP8 MoE backend selection" in reason
    assert "native GB10 W8A8 MXFP8 MoE correctness evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_moe._gb10_w8a8_mxfp8_moe_loading_unsupported_reason()
        is None
    )


def test_gb10_compressed_tensors_w8a8_int_moe_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            return None

        @staticmethod
        def _is_mxfp4(weight_quant):
            return False

        @staticmethod
        def _is_mxfp8(weight_quant):
            return False

        @staticmethod
        def _is_wNa16_group_channel(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_nvfp4_format(quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm90(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm100(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_dynamic_token_w8a8(weight_quant, input_quant):
            return True

        def get_scheme_dict(self, layer, name):
            return {
                "weights": QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.CHANNEL,
                    symmetric=True,
                    dynamic=False,
                ),
                "input_activations": QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.TOKEN,
                    symmetric=True,
                    dynamic=True,
                ),
                "format": CompressionFormat.int_quantized.value,
            }

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe.CompressedTensorsMoEMethod.get_moe_method(
            FakeCompressedTensorsConfig(),
            layer,
            "model.layers.0.mlp.experts",
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W8A8 Int8 MoE loading" in reason
    assert "generic Int8 W8A8 MoE backend selection" in reason
    assert "native GB10 W8A8 Int8 MoE correctness evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_moe._gb10_w8a8_int_moe_loading_unsupported_reason()
        is None
    )


def test_gb10_compressed_tensors_w4a8_int_moe_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            return None

        @staticmethod
        def _is_mxfp4(weight_quant):
            return False

        @staticmethod
        def _is_mxfp8(weight_quant):
            return False

        @staticmethod
        def _is_wNa16_group_channel(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_nvfp4_format(quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm90(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8_sm100(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w8a8(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_dynamic_token_w8a8(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_fp8_w4a8_sm90(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_dynamic_token_w4a8_int(weight_quant, input_quant):
            return True

        def get_scheme_dict(self, layer, name):
            return {
                "weights": QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.CHANNEL,
                    symmetric=True,
                    dynamic=False,
                ),
                "input_activations": QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.TOKEN,
                    symmetric=True,
                    dynamic=True,
                ),
                "format": CompressionFormat.int_quantized.value,
            }

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe.CompressedTensorsMoEMethod.get_moe_method(
            FakeCompressedTensorsConfig(),
            layer,
            "model.layers.0.mlp.experts",
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W4A8 Int8 MoE loading" in reason
    assert "CPU-only W4A8 Int8 MoE backend selection" in reason
    assert "native GB10 W4A8 Int8 MoE correctness evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_moe._gb10_w4a8_int_moe_loading_unsupported_reason()
        is None
    )


def test_gb10_compressed_tensors_w4a4_mxfp4_moe_rejects_marlin_sm12x(
    monkeypatch,
):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_w4a4_mxfp4,
    )

    monkeypatch.setattr(
        compressed_tensors_moe_w4a4_mxfp4.CutlassExpertsMxfp4,
        "_supports_current_device",
        staticmethod(lambda: False),
    )
    monkeypatch.setattr(
        compressed_tensors_moe_w4a4_mxfp4.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe_w4a4_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod(
            SimpleNamespace()
        )

    reason = str(exc_info.value)
    assert "CompressedTensors W4A4 MXFP4 MoE would select" in reason
    assert "FP4 Marlin fallback" in reason
    assert "native SM12x MXFP4 MoE backend" in reason

    monkeypatch.setattr(
        compressed_tensors_moe_w4a4_mxfp4.current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )
    assert (
        compressed_tensors_moe_w4a4_mxfp4._gb10_mxfp4_moe_marlin_unsupported_reason()
        is None
    )


def test_gb10_compressed_tensors_w4a16_nvfp4_moe_rejects_sm12x(monkeypatch):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe,
        compressed_tensors_moe_w4a4_nvfp4,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            return None

        @staticmethod
        def _is_mxfp4(weight_quant):
            return False

        @staticmethod
        def _is_mxfp8(weight_quant):
            return False

        @staticmethod
        def _is_wNa16_group_channel(weight_quant, input_quant):
            return False

        @staticmethod
        def _is_nvfp4_format(quant):
            return quant is not None

        def get_scheme_dict(self, layer, name):
            return {
                "weights": QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR_GROUP,
                    symmetric=True,
                    dynamic=False,
                    group_size=16,
                ),
                "input_activations": None,
                "format": CompressionFormat.float_quantized.value,
            }

    constructed = []

    class ConstructedW4A4Nvfp4MoEMethod:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        compressed_tensors_moe_w4a4_nvfp4,
        "CompressedTensorsW4A4Nvfp4MoEMethod",
        ConstructedW4A4Nvfp4MoEMethod,
    )
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        compressed_tensors_moe.CompressedTensorsMoEMethod.get_moe_method(
            FakeCompressedTensorsConfig(),
            layer,
            "model.layers.0.mlp.experts",
        )

    reason = str(exc_info.value)
    assert not constructed
    assert "CompressedTensors W4A16 NVFP4 MoE loading" in reason
    assert "weight-only NVFP4 MoE handling" in reason
    assert "native GB10 W4A16 NVFP4 MoE correctness evidence" in reason

    monkeypatch.setattr(
        compressed_tensors_moe,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        compressed_tensors_moe._gb10_w4a16_nvfp4_moe_loading_unsupported_reason()
        is None
    )


def test_gb10_unvalidated_attention_fallbacks_are_reported():
    triton_attn = (
        REPO_ROOT / "vllm" / "v1" / "attention" / "backends" / "triton_attn.py"
    ).read_text()
    flex_attn = (
        REPO_ROOT / "vllm" / "v1" / "attention" / "backends" /
        "flex_attention.py"
    ).read_text()
    turboquant_attn = (
        REPO_ROOT / "vllm" / "v1" / "attention" / "backends" /
        "turboquant_attn.py"
    ).read_text()

    assert "_gb10_triton_attention_unsupported_reason" in triton_attn
    assert "Triton attention backend" in triton_attn
    assert "generic Triton attention fallback" in triton_attn
    assert "not supported on GB10/SM12x" in triton_attn
    assert "_gb10_flex_attention_unsupported_reason" in flex_attn
    assert "FlexAttention backend" in flex_attn
    assert "PyTorch FlexAttention fallback" in flex_attn
    assert "not supported on GB10/SM12x" in flex_attn
    assert "_gb10_turboquant_attention_unsupported_reason" in turboquant_attn
    assert "TurboQuant attention backend" in turboquant_attn
    assert "TurboQuant KV-cache compression" in turboquant_attn
    assert "not supported on GB10/SM12x" in turboquant_attn


def test_gb10_mla_backend_selection_is_reported():
    cuda_platform = (REPO_ROOT / "vllm" / "platforms" / "cuda.py").read_text()
    flashmla = (
        REPO_ROOT / "vllm" / "v1" / "attention" / "backends" / "mla" /
        "flashmla.py"
    ).read_text()
    flashmla_sparse = (
        REPO_ROOT / "vllm" / "v1" / "attention" / "backends" / "mla" /
        "flashmla_sparse.py"
    ).read_text()
    triton_mla = (
        REPO_ROOT / "vllm" / "v1" / "attention" / "backends" / "mla" /
        "triton_mla.py"
    ).read_text()
    flashmla_ops = (
        REPO_ROOT / "vllm" / "v1" / "attention" / "ops" / "flashmla.py"
    ).read_text()
    sm12x_mla_priorities = cuda_platform.split(
        "if device_capability.major == 12:", 1
    )[1].split("if device_capability.major == 10:", 1)[0]

    assert "_GB10_UNVALIDATED_MLA_BACKEND_NAMES" in cuda_platform
    assert "_gb10_mla_backend_unsupported_reason" in cuda_platform
    assert "FLASHMLA_SPARSE" in cuda_platform
    assert "FLASHINFER_MLA" not in sm12x_mla_priorities
    assert "TRITON_MLA" not in sm12x_mla_priorities
    assert "FlashInfer TRT-LLM MLA" in cuda_platform
    assert "TokenSpeed CuTe DSL MLA" in cuda_platform
    assert "MLA reachability" in cuda_platform
    assert "not supported on GB10/SM12x" in cuda_platform
    assert "capability.major in [9, 10, 12]" in flashmla
    assert "capability.major in [9, 10, 12]" in flashmla_sparse
    assert "Triton MLA backend is not supported on GB10/SM12x" in triton_mla
    assert "current_platform.is_device_capability_family(120)" in flashmla_ops


def test_gb10_gdn_prefill_fallbacks_are_reported():
    gdn_prefill = (
        REPO_ROOT
        / "vllm"
        / "model_executor"
        / "layers"
        / "mamba"
        / "gdn"
        / "qwen_gdn_linear_attn.py"
    ).read_text()

    assert "_gb10_gdn_prefill_unsupported_reason" in gdn_prefill
    assert "GDN prefill requires native FlashInfer SM12x support" in gdn_prefill
    assert "Triton/FLA backend is not supported on GB10/SM12x" in gdn_prefill
    assert "CuteDSL backend is not supported on GB10/SM12x" in gdn_prefill
    assert "not native GB10 GDN prefill correctness evidence" in gdn_prefill


def test_gb10_mamba_ssu_backend_selection_is_reported():
    mamba_ssu = (
        REPO_ROOT
        / "vllm"
        / "model_executor"
        / "layers"
        / "mamba"
        / "ops"
        / "ssu_dispatch.py"
    ).read_text()

    assert "_gb10_mamba_ssu_unsupported_reason" in mamba_ssu
    assert "Triton Mamba SSU backend is not supported on GB10/SM12x" in mamba_ssu
    assert "validated FlashInfer SM12x Mamba SSU" in mamba_ssu
    assert "MambaAttentionBackendEnum.MAMBA1" in mamba_ssu
    assert "MambaAttentionBackendEnum.MAMBA2" in mamba_ssu


def test_gb10_mamba_attention_backend_selection_is_reported():
    selector = (REPO_ROOT / "vllm" / "v1" / "attention" / "selector.py").read_text()

    assert "_gb10_mamba_attn_unsupported_reason" in selector
    assert "Mamba attention backend is not supported on " in selector
    assert "GB10/SM12x" in selector
    assert "FlashInfer Mamba SSU is native GB10 evidence" in selector
    assert "MambaAttentionBackendEnum.MAMBA1" in selector
    assert "MambaAttentionBackendEnum.MAMBA2" in selector
    assert "MambaAttentionBackendEnum.SHORT_CONV" in selector
    assert "MambaAttentionBackendEnum.LINEAR" in selector
    assert "MambaAttentionBackendEnum.GDN_ATTN" not in selector.split(
        "_GB10_UNVALIDATED_MAMBA_BACKENDS", 1
    )[1].split("}", 1)[0]


def test_gb10_speculative_decoding_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_SPECULATIVE_DECODING_MESSAGE" in vllm_config
    assert "_is_gb10_sm12x_cuda_platform" in vllm_config
    assert 'self.model_config.runner_type == "draft"' in vllm_config
    assert "speculative decoding is not supported on GB10/SM12x" in vllm_config
    assert "MTP/EAGLE/draft/ngram speculative runtime paths" in vllm_config
    assert "native first-path NVFP4 release" in vllm_config


def test_gb10_pooling_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_POOLING_RUNTIME_MESSAGE" in vllm_config
    assert "pooling runtime is not supported on GB10/SM12x" in vllm_config
    assert 'self.model_config.runner_type == "pooling"' in vllm_config
    assert "embedding, classification, reward, and scoring outputs" in vllm_config
    assert "native SM12x pooling correctness" in vllm_config


def test_gb10_reasoning_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_REASONING_RUNTIME_MESSAGE" in vllm_config
    assert "reasoning runtime is not supported on GB10/SM12x" in vllm_config
    assert "self.reasoning_config is not None" in vllm_config
    assert "reasoning token parsing and output extraction" in vllm_config
    assert "native SM12x reasoning correctness" in vllm_config


def test_gb10_structured_outputs_runtime_is_reported():
    request = (
        REPO_ROOT / "vllm" / "v1" / "structured_output" / "request.py"
    ).read_text()
    worker = (
        REPO_ROOT / "vllm" / "v1" / "worker" / "gpu" / "structured_outputs.py"
    ).read_text()

    assert "_GB10_STRUCTURED_OUTPUTS_RUNTIME_MESSAGE" in request
    assert "_is_gb10_sm12x_cuda_platform" in request
    assert (
        "structured outputs runtime is not supported on GB10/SM12x" in request
    )
    assert "def from_sampling_params" in request
    assert "request-level structured_outputs/grammar constraints" in request
    assert "native SM12x structured-output correctness" in request
    assert "apply_grammar_bitmask" in worker
    assert "_apply_grammar_bitmask_kernel" in worker


def test_gb10_openai_tool_calling_runtime_is_reported():
    gb10_runtime = (
        REPO_ROOT / "vllm" / "entrypoints" / "openai" / "gb10_runtime.py"
    ).read_text()
    cli_args = (
        REPO_ROOT / "vllm" / "entrypoints" / "openai" / "cli_args.py"
    ).read_text()
    chat_serving = (
        REPO_ROOT
        / "vllm"
        / "entrypoints"
        / "openai"
        / "chat_completion"
        / "serving.py"
    ).read_text()
    responses_serving = (
        REPO_ROOT
        / "vllm"
        / "entrypoints"
        / "openai"
        / "responses"
        / "serving.py"
    ).read_text()
    tool_parser = (
        REPO_ROOT / "vllm" / "tool_parsers" / "abstract_tool_parser.py"
    ).read_text()
    engine_serving = (
        REPO_ROOT / "vllm" / "entrypoints" / "openai" / "engine" / "serving.py"
    ).read_text()

    assert "_GB10_OPENAI_TOOL_CALLING_RUNTIME_MESSAGE" in gb10_runtime
    assert "OpenAI tool-calling runtime is not supported on GB10/SM12x" in (
        gb10_runtime
    )
    assert "--enable-auto-tool-choice" in gb10_runtime
    assert "--tool-call-parser" in gb10_runtime
    assert "--tool-parser-plugin" in gb10_runtime
    assert "--tool-server" in gb10_runtime
    assert "request-level tools/tool_choice" in gb10_runtime
    assert "native SM12x tool-calling correctness" in gb10_runtime
    assert "reject_gb10_openai_tool_calling_server_args" in cli_args
    assert "gb10_openai_tool_calling_request_error" in chat_serving
    assert "gb10_openai_tool_calling_request_error" in responses_serving
    assert "ToolParser.adjust_request" in gb10_runtime
    assert "StructuredOutputsParams" in tool_parser
    assert "_parse_tool_calls_from_content" in engine_serving


def test_gb10_lora_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()
    punica_gpu = (
        REPO_ROOT / "vllm" / "lora" / "punica_wrapper" / "punica_gpu.py"
    ).read_text()
    punica_selector = (
        REPO_ROOT / "vllm" / "lora" / "punica_wrapper" / "punica_selector.py"
    ).read_text()

    assert "_GB10_LORA_RUNTIME_MESSAGE" in vllm_config
    assert "LoRA runtime is not supported on GB10/SM12x" in vllm_config
    assert "CUDA Punica" in vllm_config
    assert "Triton LoRA adapter runtime paths" in vllm_config
    assert "current_platform.get_punica_wrapper()" in punica_selector
    assert "lora_shrink" in punica_gpu
    assert "lora_expand" in punica_gpu


def test_gb10_nvfp4_kv_cache_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_NVFP4_KV_CACHE_MESSAGE" in vllm_config
    assert "NVFP4 KV cache is not supported on GB10/SM12x" in vllm_config
    assert "FP8 E4M3 KV cache" in vllm_config
    assert "runtime NVFP4 KV-cache allocation" in vllm_config
    assert "native SM12x NVFP4 KV-cache evidence" in vllm_config


def test_gb10_unvalidated_kv_cache_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_UNVALIDATED_KV_CACHE_DTYPES" in vllm_config
    assert "_GB10_UNVALIDATED_KV_CACHE_MESSAGE" in vllm_config
    assert "unvalidated KV cache runtime dtype" in vllm_config
    assert "GB10/SM12x" in vllm_config
    assert "fp8_e5m2" in vllm_config
    assert "fp8_inc" in vllm_config
    assert "int8_per_token_head" in vllm_config
    assert "fp8_per_token_head" in vllm_config
    assert "fp8_ds_mla" in vllm_config


def test_gb10_kv_transfer_and_offload_runtime_are_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_KV_TRANSFER_RUNTIME_MESSAGE" in vllm_config
    assert "_GB10_KV_OFFLOAD_RUNTIME_MESSAGE" in vllm_config
    assert "KV transfer runtime is not supported on GB10/SM12x" in vllm_config
    assert "KV offload runtime is not supported on GB10/SM12x" in vllm_config
    assert "--kv-transfer-config" in vllm_config
    assert "--kv-offloading-size" in vllm_config
    assert "disaggregated prefill/decode" in vllm_config
    assert "slot-mapping" in vllm_config


def test_gb10_ubatching_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_UBATCHING_RUNTIME_MESSAGE" in vllm_config
    assert "ubatching runtime is not supported on GB10/SM12x" in vllm_config
    assert "--enable-dbo" in vllm_config
    assert "--ubatch-size" in vllm_config
    assert "scheduler microbatching" in vllm_config
    assert "DeepEP all-to-all" in vllm_config


def test_gb10_distributed_parallel_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_DISTRIBUTED_PARALLEL_RUNTIME_MESSAGE" in vllm_config
    assert "distributed parallel runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "data parallel, tensor parallel, pipeline parallel" in vllm_config
    assert "context parallel" in vllm_config
    assert "external launcher" in vllm_config
    assert "world_size_across_dp > 1" in vllm_config
    assert "distributed_executor_backend == \"external_launcher\"" in vllm_config


def test_gb10_kv_sharing_fast_prefill_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_KV_SHARING_FAST_PREFILL_RUNTIME_MESSAGE" in vllm_config
    assert "KV sharing fast prefill runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "--kv-sharing-fast-prefill" in vllm_config
    assert "overrides attention metadata and logits indexing" in vllm_config
    assert "cache_config.kv_sharing_fast_prefill" in vllm_config


def test_gb10_ec_transfer_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_EC_TRANSFER_RUNTIME_MESSAGE" in vllm_config
    assert "EC transfer runtime is not supported on GB10/SM12x" in vllm_config
    assert "--ec-transfer-config" in vllm_config
    assert "distributed EC cache transfer connectors" in vllm_config
    assert "ec_transfer_config.is_ec_transfer_instance" in vllm_config


def test_gb10_weight_transfer_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_WEIGHT_TRANSFER_RUNTIME_MESSAGE" in vllm_config
    assert "weight transfer runtime is not supported on GB10/SM12x" in vllm_config
    assert "--weight-transfer-config" in vllm_config
    assert "RL training weight update paths use NCCL or IPC" in vllm_config
    assert "weight_transfer_config is not None" in vllm_config


def test_gb10_return_routed_experts_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_RETURN_ROUTED_EXPERTS_RUNTIME_MESSAGE" in vllm_config
    assert "return routed experts runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "--enable-return-routed-experts" in vllm_config
    assert "routed experts capture changes MoE scheduler" in vllm_config
    assert "model_config.enable_return_routed_experts" in vllm_config


def test_gb10_logprobs_logits_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_LOGPROBS_LOGITS_RUNTIME_MESSAGE" in vllm_config
    assert "logprobs logits runtime is not supported on GB10/SM12x" in vllm_config
    assert "logprobs_mode={logprobs_mode!r}" in vllm_config
    assert "raw_logits" in vllm_config
    assert "processed_logits" in vllm_config
    assert "native SM12x logits-return correctness" in vllm_config


def test_gb10_custom_logits_processors_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_CUSTOM_LOGITS_PROCESSORS_RUNTIME_MESSAGE" in vllm_config
    assert "custom logits processors runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "--logits-processors" in vllm_config
    assert "model_config.logits_processors" in vllm_config
    assert "native SM12x custom logits processor correctness" in vllm_config


def test_gb10_io_processor_plugin_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()
    model_config = (REPO_ROOT / "vllm" / "config" / "model.py").read_text()
    arg_utils = (REPO_ROOT / "vllm" / "engine" / "arg_utils.py").read_text()

    assert "_GB10_IO_PROCESSOR_PLUGIN_RUNTIME_MESSAGE" in vllm_config
    assert "IO processor plugin runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "--io-processor-plugin" in vllm_config
    assert "model_config.io_processor_plugin" in vllm_config
    assert "io_processor_plugin" in model_config
    assert "--io-processor-plugin" in arg_utils


def test_gb10_hf_overrides_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()
    model_config = (REPO_ROOT / "vllm" / "config" / "model.py").read_text()
    arg_utils = (REPO_ROOT / "vllm" / "engine" / "arg_utils.py").read_text()

    assert "_GB10_HF_OVERRIDES_RUNTIME_MESSAGE" in vllm_config
    assert "HF overrides runtime is not supported on GB10/SM12x" in vllm_config
    assert "--hf-overrides" in vllm_config
    assert "model_config.hf_overrides" in vllm_config
    assert "hf_overrides" in model_config
    assert "--hf-overrides" in arg_utils


def test_gb10_transformers_model_impl_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()
    model_config = (REPO_ROOT / "vllm" / "config" / "model.py").read_text()
    arg_utils = (REPO_ROOT / "vllm" / "engine" / "arg_utils.py").read_text()

    assert "_GB10_TRANSFORMERS_MODEL_IMPL_RUNTIME_MESSAGE" in vllm_config
    assert (
        "Transformers model implementation runtime is not supported on GB10/SM12x"
        in vllm_config
    )
    assert "--model-impl transformers" in vllm_config
    assert 'model_config.model_impl == "transformers"' in vllm_config
    assert "using_transformers_backend" in model_config
    assert "--model-impl" in arg_utils


def test_gb10_trust_remote_code_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()
    model_config = (REPO_ROOT / "vllm" / "config" / "model.py").read_text()
    arg_utils = (REPO_ROOT / "vllm" / "engine" / "arg_utils.py").read_text()

    assert "_GB10_TRUST_REMOTE_CODE_RUNTIME_MESSAGE" in vllm_config
    assert "trust remote code runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "--trust-remote-code" in vllm_config
    assert "model_config.trust_remote_code" in vllm_config
    assert "trust_remote_code" in model_config
    assert "--trust-remote-code" in arg_utils


def test_gb10_custom_scheduler_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()
    scheduler_config = (
        REPO_ROOT / "vllm" / "config" / "scheduler.py"
    ).read_text()
    arg_utils = (REPO_ROOT / "vllm" / "engine" / "arg_utils.py").read_text()

    assert "_GB10_CUSTOM_SCHEDULER_RUNTIME_MESSAGE" in vllm_config
    assert "custom scheduler runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "--scheduler-cls" in vllm_config
    assert "scheduler_config.scheduler_cls" in vllm_config
    assert "scheduler_cls" in scheduler_config
    assert "--scheduler-cls" in arg_utils


def test_gb10_custom_worker_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()
    parallel_config = (REPO_ROOT / "vllm" / "config" / "parallel.py").read_text()
    arg_utils = (REPO_ROOT / "vllm" / "engine" / "arg_utils.py").read_text()

    assert "_GB10_CUSTOM_WORKER_RUNTIME_MESSAGE" in vllm_config
    assert "custom worker runtime is not supported on GB10/SM12x" in vllm_config
    assert "--worker-cls" in vllm_config
    assert "--worker-extension-cls" in vllm_config
    assert "parallel_config.worker_cls" in vllm_config
    assert "parallel_config.worker_extension_cls" in vllm_config
    assert "worker_cls" in parallel_config
    assert "worker_extension_cls" in parallel_config
    assert "--worker-cls" in arg_utils
    assert "--worker-extension-cls" in arg_utils


def test_gb10_kv_events_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()
    kv_events_config = (
        REPO_ROOT / "vllm" / "config" / "kv_events.py"
    ).read_text()
    arg_utils = (REPO_ROOT / "vllm" / "engine" / "arg_utils.py").read_text()

    assert "_GB10_KV_EVENTS_RUNTIME_MESSAGE" in vllm_config
    assert "KV events runtime is not supported on GB10/SM12x" in vllm_config
    assert "--kv-events-config" in vllm_config
    assert "kv_events_config.enable_kv_cache_events" in vllm_config
    assert "kv_events_config.publisher" in vllm_config
    assert "enable_kv_cache_events" in kv_events_config
    assert "publisher" in kv_events_config
    assert "--kv-events-config" in arg_utils


def test_gb10_prompt_embeds_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_PROMPT_EMBEDS_RUNTIME_MESSAGE" in vllm_config
    assert "prompt embeds runtime is not supported on GB10/SM12x" in vllm_config
    assert "--enable-prompt-embeds" in vllm_config
    assert "model_config.enable_prompt_embeds" in vllm_config
    assert "native SM12x prompt-embeds correctness" in vllm_config


def test_gb10_stock_torch_compile_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_STOCK_TORCH_COMPILE_RUNTIME_MESSAGE" in vllm_config
    assert "stock torch.compile runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "CompilationMode.STOCK_TORCH_COMPILE" in vllm_config
    assert "native SM12x stock torch.compile correctness" in vllm_config


def test_gb10_mamba_align_cache_runtime_is_reported():
    vllm_config = (REPO_ROOT / "vllm" / "config" / "vllm.py").read_text()

    assert "_GB10_MAMBA_ALIGN_CACHE_RUNTIME_MESSAGE" in vllm_config
    assert "Mamba align cache runtime is not supported on GB10/SM12x" in (
        vllm_config
    )
    assert "mamba_cache_mode='align'" in vllm_config
    assert "native SM12x Mamba align-cache correctness" in vllm_config


def test_gb10_mm_encoder_fp8_attention_is_reported():
    mm_encoder_attention = (
        REPO_ROOT
        / "vllm"
        / "model_executor"
        / "layers"
        / "attention"
        / "mm_encoder_attention.py"
    ).read_text()

    assert "_gb10_mm_encoder_fp8_attention_unsupported_reason" in (
        mm_encoder_attention
    )
    assert "MM encoder FP8 attention is not supported on GB10/SM12x" in (
        mm_encoder_attention
    )
    assert "FlashInfer cuDNN FP8 ViT attention path" in mm_encoder_attention
    assert "not native GB10 MM encoder attention correctness" in mm_encoder_attention
    assert "mm_encoder_attn_dtype unset" in mm_encoder_attention


def test_gb10_mm_encoder_attention_fallbacks_are_reported():
    cuda_platform = (REPO_ROOT / "vllm" / "platforms" / "cuda.py").read_text()

    assert "_gb10_vit_attn_backend_unsupported_reason" in cuda_platform
    assert "MM encoder attention backend" in cuda_platform
    assert "is not supported" in cuda_platform
    assert "on GB10/SM12x" in cuda_platform
    assert "public FlashAttention" in cuda_platform
    assert "Triton" in cuda_platform
    assert "Torch SDPA" in cuda_platform
    assert "requires FlashInfer on GB10/SM12x" in cuda_platform
    assert "not native GB10 MM encoder attention evidence" in cuda_platform


def test_gb10_compressed_tensors_qutlass_nvfp4_transform_rejects_sm12x(
    monkeypatch,
):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsLinearTransformMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW4A4Fp4,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.transform.schemes import (  # noqa: E501
        linear_qutlass_nvfp4,
    )

    quant_scheme = object.__new__(CompressedTensorsW4A4Fp4)
    quant_scheme.group_size = 16
    input_tfms = {0: SimpleNamespace(scheme=SimpleNamespace(head_dim=16))}

    monkeypatch.setattr(
        linear_qutlass_nvfp4,
        "_is_sm12x_device",
        lambda: True,
        raising=False,
    )

    with pytest.raises(ValueError, match="not supported on GB10/SM12x") as exc_info:
        CompressedTensorsLinearTransformMethod.from_schemes(
            SimpleNamespace(),
            quant_scheme,
            input_tfms,
            {},
        )

    reason = str(exc_info.value)
    assert "CompressedTensors Qutlass NVFP4 transform loading" in reason
    assert "QutlassNvFP4LinearMethod.apply" in reason
    assert "native GB10 transformed NVFP4 correctness evidence" in reason

    monkeypatch.setattr(
        linear_qutlass_nvfp4,
        "_is_sm12x_device",
        lambda: False,
        raising=False,
    )
    assert (
        linear_qutlass_nvfp4._gb10_qutlass_nvfp4_transform_unsupported_reason()
        is None
    )
    assert isinstance(
        CompressedTensorsLinearTransformMethod.from_schemes(
            SimpleNamespace(),
            quant_scheme,
            input_tfms,
            {},
        ),
        linear_qutlass_nvfp4.QutlassNvFP4LinearMethod,
    )


def test_gb10_nvfp4_moe_fallbacks_are_reported():
    nvfp4_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "fused_moe" / "oracle" / "nvfp4.py"
    ).read_text()

    assert "_NVFP4_MOE_FALLBACK_BACKENDS" in nvfp4_oracle
    assert "NvFp4MoeBackend.MARLIN" in nvfp4_oracle
    assert "NvFp4MoeBackend.EMULATION" in nvfp4_oracle
    assert "NvFp4MoeBackend.FLASHINFER_CUTEDSL" in nvfp4_oracle
    assert "NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED" in nvfp4_oracle
    assert "_gb10_unsupported_backend_reason" in nvfp4_oracle
    assert "_gb10_flashinfer_cutedsl_moe_unsupported_reason" in nvfp4_oracle
    assert "_gb10_nvfp4_moe_fallback_unsupported_reason" in nvfp4_oracle
    assert "not supported on GB10/SM12x" in nvfp4_oracle
    assert "FlashInfer CuteDSL NVFP4 MoE" in nvfp4_oracle
    assert "cannot satisfy native GB10 NVFP4 Tensor Core evidence" in nvfp4_oracle
    assert "if b not in gb10_unsupported_reasons_by_backend" in nvfp4_oracle
    assert "unavailable_native_backend_reasons" in nvfp4_oracle
    assert "record_nvfp4_backend_selection" in nvfp4_oracle
    assert "record_nvfp4_fallback" in nvfp4_oracle
    assert "NVFP4 MoE selected fallback backend '" in nvfp4_oracle
    assert "not the native " in nvfp4_oracle
    assert "GB10 W4A4 FP4 fused MoE path" in nvfp4_oracle
    assert "verify " in nvfp4_oracle
    assert "this fallback is intentional " in nvfp4_oracle
    assert "before publishing GB10 artifacts" in nvfp4_oracle
    assert "Unavailable native backend reasons" in nvfp4_oracle
    assert "VLLM_USE_FLASHINFER_MOE_FP4=0" in nvfp4_oracle


def test_gb10_attention_selector_logs_backend_and_kv_cache_dtype():
    selector = (REPO_ROOT / "vllm" / "v1" / "attention" / "selector.py").read_text()

    assert "Using %s attention backend with requested_backend=%s" in selector
    assert "num_heads=%s" in selector
    assert "selector_config=%s" in selector
    assert "AttentionSelectorConfig(head_size=" in selector
    assert "kv_cache_dtype={self.kv_cache_dtype}" in selector
    assert "dtype={self.dtype}" in selector


def test_gb10_nvfp4_backend_recorder_can_fail_fast(monkeypatch):
    import vllm.envs as envs
    from vllm.model_executor.layers.quantization.utils.nvfp4_fallback import (
        clear_nvfp4_backend_events,
        get_nvfp4_backend_selection_events,
        get_nvfp4_fallback_events,
        record_nvfp4_backend_selection,
        record_nvfp4_fallback,
    )

    clear_nvfp4_backend_events()
    monkeypatch.setattr(envs, "VLLM_FAIL_ON_NVFP4_FALLBACK", False)

    record_nvfp4_backend_selection(
        "linear",
        "FlashInferB12xNvFp4LinearKernel",
        is_fallback=False,
    )
    record_nvfp4_backend_selection(
        "moe",
        "MARLIN",
        is_fallback=True,
    )
    record_nvfp4_fallback("linear", "MarlinNvFp4LinearKernel", "fallback selected")
    selections = get_nvfp4_backend_selection_events()
    events = get_nvfp4_fallback_events()

    assert len(selections) == 2
    assert selections[0].path == "linear"
    assert selections[0].backend == "FlashInferB12xNvFp4LinearKernel"
    assert not selections[0].is_fallback
    assert selections[1].path == "moe"
    assert selections[1].backend == "MARLIN"
    assert selections[1].is_fallback

    assert len(events) == 1
    assert events[0].path == "linear"
    assert events[0].backend == "MarlinNvFp4LinearKernel"
    assert events[0].message == "fallback selected"

    monkeypatch.setattr(envs, "VLLM_FAIL_ON_NVFP4_FALLBACK", True)
    with pytest.raises(RuntimeError, match="fallback selected"):
        record_nvfp4_fallback("moe", "MARLIN", "fallback selected")


def test_gb10_nvfp4_model_smoke_asserts_native_backend_selection():
    script = (REPO_ROOT / "scripts" / "gb10-smoke-nvfp4.py").read_text()
    smoke = _load_gb10_smoke_module()

    assert "VLLM_FAIL_ON_NVFP4_FALLBACK" in script
    assert 'os.environ["VLLM_FAIL_ON_NVFP4_FALLBACK"] = "1"' in script
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" in script
    assert 'os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")' in script
    assert "GB10_NVFP4_MODEL" in script
    assert 'parser.error("--model or GB10_NVFP4_MODEL is required")' in script

    assert "engine_args_cls.add_cli_args" in script
    assert "EngineArgs.from_cli_args" in script
    assert "LLM.from_engine_args" in script
    assert "SamplingParams" in script
    assert 'quantization="modelopt_fp4"' in script
    assert 'kv_cache_dtype="fp8_e4m3"' in script
    assert "enable_prefix_caching=False" in script

    assert "clear_nvfp4_backend_events" in script
    assert "get_nvfp4_backend_selection_events" in script
    assert "get_nvfp4_fallback_events" in script
    assert "NVFP4 fallback events were recorded during GB10 smoke" in script
    assert "NVFP4 backend selections included fallback paths" in script
    assert "No NVFP4 backend-selection event was recorded" in script

    assert "--gb10-require-path" in script
    assert "--gb10-expect-backend" in script
    assert "--gb10-report-json" in script
    assert "_write_report" in script
    assert "_build_report" in script
    assert "_build_backend_summary" in script
    assert "_collect_runtime_metadata" in script
    assert '"schema_version": 1' in script
    assert "report[\"backend_summary\"] = backend_summary" in script
    assert '"gb10_release_summary"' in script
    assert "_build_gb10_release_summary(" in script
    assert '"first_path_smoke_passed"' in script
    assert '"remaining_release_evidence"' in script
    assert '"vllm_config": vllm_config_summary' in script
    assert "_collect_vllm_config_summary(llm)" in script
    assert '"native_nvfp4_gemm"' in script
    assert '"native_nvfp4_moe_non_ep"' in script
    assert '"native_nvfp4_moe_ep"' in script
    assert '"cuda_graph"' in script
    assert '"model_shape"' in script
    assert '"unsupported_paths"' in script
    assert '"deferred_paths"' in script
    assert "num_cudagraph_captured" in script
    assert "num_cudagraph_replayed" in script
    assert "attention_backend" in script
    assert "attention_backend_mismatch" in script
    assert "quantization_mode_mismatch" in script
    assert '"not_validated_by_smoke"' in script
    assert '"configured_cudagraph_mode"' in script
    assert '"configured_cudagraph_enabled"' in script
    assert "Expert-parallel/all2all/EPLB NVFP4 MoE is blocked" in script
    assert '"backend_selections": _events_to_dicts(selections)' in script
    assert '"fallback_events": _events_to_dicts(fallbacks)' in script
    assert '"device_capability"' in script
    assert '"flashinfer_version"' in script
    assert '"flashinfer_distributions"' in script
    assert smoke.FLASHINFER_RUNTIME_DISTRIBUTIONS == (
        GB10_FLASHINFER_RUNTIME_DISTRIBUTIONS
    )
    assert 'status="passed"' in script
    assert 'status="failed"' in script
    assert "except Exception as exc:" in script
    assert "linear=FlashInferB12x" in script
    assert "moe=FLASHINFER_B12X" in script
    assert 'choices=("linear", "linear_w4a16", "moe")' in script


def test_gb10_nvfp4_model_smoke_collects_flashinfer_distribution_versions(
    monkeypatch,
):
    smoke = _load_gb10_smoke_module()

    versions = {
        "flashinfer-python": "0.6.12+cu130gb10",
        "flashinfer-cubin": "0.6.12+cu130gb10",
        "flashinfer-jit-cache": "0.6.12+cu130gb10",
    }

    def fake_version(distribution_name):
        if distribution_name not in versions:
            raise smoke.importlib_metadata.PackageNotFoundError(distribution_name)
        return versions[distribution_name]

    monkeypatch.setattr(smoke.importlib_metadata, "version", fake_version)

    assert smoke._collect_distribution_versions(
        ("flashinfer-python", "flashinfer-cubin", "missing")
    ) == {
        "flashinfer-python": "0.6.12+cu130gb10",
        "flashinfer-cubin": "0.6.12+cu130gb10",
        "missing": None,
    }


def test_gb10_nvfp4_model_smoke_summarizes_vllm_config():
    smoke = _load_gb10_smoke_module()
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        world_size=1,
        distributed_executor_backend="uni",
    )
    model_config = SimpleNamespace(
        dtype="bfloat16",
        quantization="modelopt_fp4",
        max_model_len=4096,
        enforce_eager=False,
        use_mla=False,
        is_attention_free=False,
        get_head_size=lambda: 128,
        get_num_attention_heads=lambda _parallel_config: 32,
        get_num_kv_heads=lambda _parallel_config: 8,
    )
    attention_config = SimpleNamespace(
        backend=SimpleNamespace(name="FLASHINFER"),
        mla_prefill_backend=None,
        use_trtllm_attention=None,
        use_prefill_query_quantization=False,
        use_non_causal=False,
    )
    cache_config = SimpleNamespace(
        cache_dtype="fp8_e4m3",
        block_size=16,
        enable_prefix_caching=False,
        kv_cache_dtype_skip_layers=[],
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
    )
    compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(name="PIECEWISE"),
        max_cudagraph_capture_size=256,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        cudagraph_num_of_warmups=1,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        attention_config=attention_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=32,
            max_num_batched_tokens=4096,
            enable_chunked_prefill=True,
        ),
        compilation_config=compilation_config,
        observability_config=SimpleNamespace(cudagraph_metrics=True),
        kv_transfer_config=None,
    )
    llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=vllm_config))

    summary = smoke._collect_vllm_config_summary(llm)

    assert summary["model"]["quantization"] == "modelopt_fp4"
    assert summary["model"]["head_size"] == 128
    assert summary["model"]["num_attention_heads"] == 32
    assert summary["model"]["num_kv_heads"] == 8
    assert summary["attention"]["requested_backend"] == "FLASHINFER"
    assert summary["cache"]["cache_dtype"] == "fp8_e4m3"
    assert summary["cache"]["enable_prefix_caching"] is False
    assert summary["parallel"]["tensor_parallel_size"] == 1
    assert summary["scheduler"]["enable_chunked_prefill"] is True
    assert summary["compilation"]["cudagraph_mode"] == "PIECEWISE"
    assert summary["compilation"]["cudagraph_enabled"] is True
    assert summary["compilation"]["cudagraph_capture_sizes"] == {
        "count": 9,
        "first": [1, 2, 4, 8, 16, 32, 64, 128],
        "last": [2, 4, 8, 16, 32, 64, 128, 256],
        "max": 256,
    }

    backend_summary = smoke._build_backend_summary(
        status="passed",
        required_paths=("linear",),
        selections=(),
        fallbacks=(),
        vllm_config_summary=summary,
    )
    assert (
        backend_summary["capabilities"]["cuda_graph"]["configured_cudagraph_mode"]
        == "PIECEWISE"
    )
    assert (
        backend_summary["capabilities"]["cuda_graph"]["configured_cudagraph_enabled"]
        is True
    )


def test_gb10_nvfp4_model_smoke_builds_release_summary(monkeypatch):
    smoke = _load_gb10_smoke_module()
    monkeypatch.setattr(
        smoke,
        "_collect_runtime_metadata",
        lambda: {
            "compilation_counter": {
                "num_cudagraph_captured": 2,
                "num_cudagraph_replayed": 1,
            },
        },
    )
    args = SimpleNamespace(
        model="gb10-model",
        quantization="modelopt_fp4",
        kv_cache_dtype="fp8_e4m3",
        gb10_max_tokens=8,
        gb10_temperature=0.0,
        gb10_sampling_seed=0,
        gb10_skip_generate=False,
        gb10_allow_fallback=False,
        gb10_expect_backend=[],
        attention_backend="flashinfer",
    )
    selections = (
        SimpleNamespace(
            path="linear",
            backend="FlashInferB12xNvFp4LinearKernel",
            is_fallback=False,
        ),
        SimpleNamespace(path="moe", backend="FLASHINFER_B12X", is_fallback=False),
    )
    vllm_config_summary = {
        "model": {
            "max_model_len": 4096,
            "head_size": 128,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "quantization": "modelopt_fp4",
        },
        "attention": {
            "requested_backend": "FLASHINFER",
            "mla_prefill_backend": None,
        },
        "cache": {"cache_dtype": "fp8_e4m3"},
        "compilation": {
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_enabled": True,
        },
    }

    report = smoke._build_report(
        args=args,
        required_paths=("linear", "moe"),
        selections=selections,
        fallbacks=(),
        outputs=(),
        status="passed",
        vllm_config_summary=vllm_config_summary,
    )
    release_summary = report["gb10_release_summary"]

    assert release_summary["first_path_smoke_passed"] is True
    assert release_summary["release_ready"] is False
    assert release_summary["smoke_blockers"] == []
    assert release_summary["checks"]["native_nvfp4_gemm"]["status"] == "observed"
    assert release_summary["checks"]["native_nvfp4_moe_non_ep"]["status"] == "observed"
    assert release_summary["checks"]["fallback_free"]["status"] == "passed"
    assert release_summary["checks"]["kv_cache_dtype"] == {
        "status": "passed",
        "expected": "fp8_e4m3",
        "configured": "fp8_e4m3",
    }
    assert release_summary["checks"]["model_shape"] == {
        "status": "observed",
        "model": "gb10-model",
        "max_model_len": 4096,
        "head_size": 128,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
    }
    assert release_summary["checks"]["attention_backend"] == {
        "status": "passed",
        "expected": "FLASHINFER",
        "requested_backend": "FLASHINFER",
        "mla_prefill_backend": None,
    }
    assert release_summary["checks"]["quantization"] == {
        "status": "passed",
        "expected": "modelopt_fp4",
        "configured": "modelopt_fp4",
    }
    assert release_summary["checks"]["cuda_graph"] == {
        "status": "passed",
        "configured_mode": "PIECEWISE",
        "configured_enabled": True,
        "num_cudagraph_captured": 2,
        "num_cudagraph_replayed": 1,
    }
    assert release_summary["unsupported_paths"] == {
        name: {
            "status": "not_supported",
            "expected_handling": "route_or_reject_before_release_evidence",
            "reason": reason,
        }
        for name, reason in smoke.GB10_NOT_SUPPORTED_PATH_REASONS.items()
    }
    assert release_summary["deferred_paths"] == {
        name: {
            "status": "deferred",
            "expected_handling": "block_until_hardware_validated",
            "reason": reason,
        }
        for name, reason in smoke.GB10_DEFERRED_PATH_REASONS.items()
    }
    assert "OpenAI-compatible server smoke" in release_summary[
        "remaining_release_evidence"
    ]

    missing_shape_summary = copy.deepcopy(vllm_config_summary)
    missing_shape_summary["model"].pop("head_size")
    missing_shape_report = smoke._build_report(
        args=args,
        required_paths=("linear", "moe"),
        selections=selections,
        fallbacks=(),
        outputs=(),
        status="passed",
        vllm_config_summary=missing_shape_summary,
    )
    assert (
        missing_shape_report["gb10_release_summary"]["first_path_smoke_passed"]
        is False
    )
    assert (
        "model shape metadata was not observed"
        in missing_shape_report["gb10_release_summary"]["smoke_blockers"]
    )

    monkeypatch.setattr(
        smoke,
        "_collect_runtime_metadata",
        lambda: {
            "compilation_counter": {
                "num_cudagraph_captured": 2,
                "num_cudagraph_replayed": 0,
            },
        },
    )
    missing_replay_report = smoke._build_report(
        args=args,
        required_paths=("linear", "moe"),
        selections=selections,
        fallbacks=(),
        outputs=(),
        status="passed",
        vllm_config_summary=vllm_config_summary,
    )
    assert (
        missing_replay_report["gb10_release_summary"]["checks"]["cuda_graph"][
            "status"
        ]
        == "not_observed"
    )
    assert (
        missing_replay_report["gb10_release_summary"]["first_path_smoke_passed"]
        is False
    )
    assert (
        "CUDA graph capture/replay was not observed"
        in missing_replay_report["gb10_release_summary"]["smoke_blockers"]
    )


def test_gb10_image_smoke_wraps_nvfp4_model_harness():
    script = (REPO_ROOT / "scripts" / "gb10-smoke-image.sh").read_text()

    assert "GB10_NVFP4_MODEL is required unless --model is passed after --." in script
    assert "scripts/gb10-smoke-nvfp4.py" in script
    assert "/tmp/gb10-smoke-nvfp4.py:ro" in script
    assert "--gpus all" in script
    assert "--ipc" in script
    assert 'GB10_SMOKE_IPC:-host' in script
    assert "--shm-size" in script
    assert 'GB10_SMOKE_SHM_SIZE:-16g' in script
    assert "GB10_SMOKE_CACHE_DIR" in script
    assert "GB10_SMOKE_REPORT_DIR" in script
    assert "-v \"$cache_dir:/root/.cache\"" in script
    assert "-v \"$report_dir:/gb10-smoke-reports\"" in script
    assert "--gb10-report-json" in script
    assert "/gb10-smoke-reports/gb10-nvfp4-smoke.json" in script
    assert "VLLM_FAIL_ON_NVFP4_FALLBACK=1" in script
    assert "VLLM_ENABLE_V1_MULTIPROCESSING=0" in script
    assert "VLLM_NO_USAGE_STATS=1" in script
    assert "GB10_SMOKE_ENV_FILE" in script
    assert "GB10_SMOKE_EXTRA_DOCKER_ARGS" in script
    assert "HF_TOKEN" in script
    assert "HUGGING_FACE_HUB_TOKEN" in script
    assert 'python3 /tmp/gb10-smoke-nvfp4.py "$@"' in script


def test_gb10_openai_image_smoke_wraps_server_harness():
    script = (REPO_ROOT / "scripts" / "gb10-smoke-openai-image.sh").read_text()

    assert "scripts/gb10-smoke-openai-server.py" in script
    assert "vLLM runtime image to test" in script
    assert "--serve SERVE_ARGS" in script
    assert "--smoke SMOKE_ARGS" in script
    assert "GB10_OPENAI_IMAGE_HOST_PORT" in script
    assert 'host_port="${GB10_OPENAI_IMAGE_HOST_PORT:-18000}"' in script
    assert "GB10_OPENAI_IMAGE_CONTAINER_PORT" in script
    assert 'container_port="${GB10_OPENAI_IMAGE_CONTAINER_PORT:-8000}"' in script
    assert "--publish \"127.0.0.1:${host_port}:${container_port}\"" in script
    assert "GB10_OPENAI_IMAGE_KEEP_CONTAINER" in script
    assert "docker rm -f" in script
    assert "trap cleanup EXIT" in script
    assert "/v1/models" in script
    assert "curl -fsS --max-time 5" in script
    assert "docker logs" in script
    assert "VLLM_FAIL_ON_NVFP4_FALLBACK=1" in script
    assert "VLLM_NO_USAGE_STATS=1" in script
    assert "GB10_OPENAI_IMAGE_CACHE_DIR" in script
    assert "GB10_OPENAI_IMAGE_REPORT_DIR" in script
    assert "GB10_OPENAI_IMAGE_ENV_FILE" in script
    assert "GB10_OPENAI_IMAGE_EXTRA_DOCKER_ARGS" in script
    assert "HF_TOKEN" in script
    assert "HUGGING_FACE_HUB_TOKEN" in script
    assert "gb10-openai-server-smoke-image.json" in script
    assert "--gb10-base-url" in script
    assert "--gb10-report-json" in script
    assert "--gb10-repeat-count" in script
    assert "--gb10-require-deterministic" in script
    assert "GB10_OPENAI_IMAGE_REQUIRE_DETERMINISTIC" in script
    assert 'python3 "$smoke_script" "${smoke_args[@]}"' in script


def test_gb10_release_image_smoke_orchestrates_final_reports():
    script = (REPO_ROOT / "scripts" / "gb10-smoke-release-image.sh").read_text()
    bundler = _load_gb10_release_bundle_module()
    contract = _load_gb10_release_contract_module()
    report_lister = _load_gb10_release_evidence_report_file_lister_module()

    assert "scripts/gb10-smoke-image.sh" in script
    assert "scripts/gb10-smoke-openai-image.sh" in script
    assert "scripts/gb10-verify-release-evidence.py" in script
    assert "scripts/gb10-bundle-release-evidence.py" in script
    assert "scripts/gb10-list-release-evidence-report-files.py" in script
    assert "--offline OFFLINE_ARGS" in script
    assert "--serve SERVE_ARGS" in script
    assert "--openai OPENAI_ARGS" in script
    assert "--verify VERIFY_ARGS" in script
    assert "GB10_RELEASE_SMOKE_REPORT_DIR" in script
    assert "GB10_RELEASE_REQUIRE_MOE" in script
    assert "GB10_RELEASE_REQUIRE_OPENAI_DETERMINISTIC" in script
    assert "GB10_RELEASE_EVIDENCE_OUTPUT_DIR" in script
    assert "GB10_RELEASE_BUNDLE_ALLOW_PARTIAL" in script
    assert "GB10_RELEASE_MANIFEST_JSON" in script
    assert "GB10_RUNTIME_IMAGE_METADATA_JSON" in script
    assert "GB10_IMAGE_DIGEST" in script
    assert "GB10_RELEASE_TAG" in script
    assert "GB10_RELEASE_ALLOW_EXISTING_VLLM_CONTAINERS" in script
    assert "docker ps --format" in script
    assert "pre_smoke_resource_guard" in script
    assert "GB10_RELEASE_EVIDENCE_BUNDLE_NAME" in script
    assert "scripts/gb10-list-evidence-release-assets.py" in script
    assert "Evidence report files listed by" in script
    assert "GB10 release evidence assets:" in script
    assert "--gb10-report-json" in script
    assert 'basename "$nvfp4_report"' in script
    assert "--gb10-nvfp4-report-json" in script
    assert "--gb10-openai-report-json" in script
    assert "--gb10-release-manifest-json" in script
    assert "--gb10-runtime-image-metadata-json" in script
    assert "--gb10-image-ref" in script
    assert "--gb10-image-digest" in script
    assert "--gb10-release-tag" in script
    assert "--gb10-output-json" in script
    assert "--gb10-require-moe" in script
    assert "--gb10-require-openai-deterministic" in script
    assert "docker image inspect \"$image\"" in script
    assert "GB10_SMOKE_REPORT_DIR=\"$report_dir\"" in script
    assert "GB10_OPENAI_IMAGE_REPORT_DIR=\"$report_dir\"" in script
    assert 'verify_args+=(--gb10-image-ref "$image")' in script
    metadata_arg = (
        'verify_args+=(--gb10-runtime-image-metadata-json '
        '"$GB10_RUNTIME_IMAGE_METADATA_JSON")'
    )
    assert metadata_arg in script
    assert 'verify_args+=(--gb10-image-digest "$GB10_IMAGE_DIGEST")' in script
    assert 'verify_args+=(--gb10-release-tag "$GB10_RELEASE_TAG")' in script
    assert 'verify_args+=(--gb10-release-manifest-json' in script
    assert '"$offline_wrapper" "$image" -- "${offline_args[@]}"' in script
    assert '"$openai_wrapper" "$image" --serve "${serve_args[@]}" --smoke' in script
    assert '"$verifier" "${verify_args[@]}" || verify_status=$?' in script
    assert '"$bundler" "${bundle_args[@]}"' in script
    assert '"$asset_lister" >&2' in script
    assert 'bundle_args+=(--gb10-release-tag "$GB10_RELEASE_TAG")' in script
    assert 'bundle_args+=(--gb10-release-manifest-json' in script
    assert 'bundle_args+=(--gb10-runtime-image-metadata-json' in script
    assert 'bundle_args+=(--gb10-allow-partial)' in script
    assert 'exit "$verify_status"' in script

    assert 'report_files_file="$(mktemp)"' in script
    assert '"$report_file_lister" --gb10-report-dir "$report_dir"' in script
    assert "mapfile -t release_evidence_files" in script
    assert '"${#release_evidence_files[@]}" -ne 4' in script
    assert 'nvfp4_report="${release_evidence_files[0]}"' in script
    assert 'openai_report="${release_evidence_files[1]}"' in script
    assert 'evidence_report="${release_evidence_files[2]}"' in script
    assert 'smoked_image_digest_report="${release_evidence_files[3]}"' in script
    assert (
        'nvfp4_report="$report_dir/gb10-nvfp4-smoke.json"'
        not in script
    )
    assert (
        'openai_report="$report_dir/gb10-openai-server-smoke-image.json"'
        not in script
    )
    assert (
        'evidence_report="$report_dir/gb10-release-evidence-image.json"'
        not in script
    )
    assert (
        'smoked_image_digest_report="$report_dir/gb10-smoked-image-digest.txt"'
        not in script
    )
    orchestrator_evidence_files = tuple(
        path.name
        for path in report_lister.list_release_evidence_report_files(
            Path("reports")
        )
    )

    assert bundler.EXPECTED_REPORTS == GB10_EXPECTED_RELEASE_REPORTS
    assert bundler.EXPECTED_EVIDENCE_FILES == GB10_EXPECTED_RELEASE_EVIDENCE_FILES
    assert contract.EXPECTED_RELEASE_REPORTS == GB10_EXPECTED_RELEASE_REPORTS
    assert contract.EXPECTED_RELEASE_EVIDENCE_FILES == (
        GB10_EXPECTED_RELEASE_EVIDENCE_FILES
    )
    assert report_lister.release_evidence_file_paths(Path("reports")) == [
        Path("reports") / filename
        for filename in contract.EXPECTED_RELEASE_EVIDENCE_FILES
    ]
    assert orchestrator_evidence_files == bundler.EXPECTED_EVIDENCE_FILES


def test_gb10_release_image_smoke_guard_bundles_partial_evidence(tmp_path):
    script = REPO_ROOT / "scripts" / "gb10-smoke-release-image.sh"
    bundle_name = "guard-evidence"
    image_ref = "ghcr.io/gardner/vllm-gb10:gb10-test"
    fake_bin = tmp_path / "bin"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "evidence"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [ \"${1:-}\" = ps ]; then\n"
        "  printf '%s\\n' "
        "'abc123 running-vllm vllm/vllm-openai:nightly \"vllm serve\"'\n"
        "  exit 0\n"
        "fi\n"
        "echo \"unexpected docker invocation: $*\" >&2\n"
        "exit 64\n"
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "GB10_RELEASE_SMOKE_REPORT_DIR": str(report_dir),
            "GB10_RELEASE_EVIDENCE_OUTPUT_DIR": str(output_dir),
            "GB10_RELEASE_EVIDENCE_BUNDLE_NAME": bundle_name,
            "GB10_NVFP4_MODEL": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        }
    )

    proc = subprocess.run(
        [
            str(script),
            image_ref,
            "--serve",
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "GB10 final-image smoke refused to start" in proc.stderr
    assert "unexpected docker invocation" not in proc.stderr

    evidence_report = report_dir / "gb10-release-evidence-image.json"
    metadata_path = output_dir / "release-evidence-metadata.json"
    checksum_path = output_dir / "SHA256SUMS"
    archive_path = output_dir / f"{bundle_name}.tar.gz"
    archive_checksum_path = output_dir / f"{bundle_name}.tar.gz.sha256"
    for path in (
        evidence_report,
        metadata_path,
        checksum_path,
        archive_path,
        archive_checksum_path,
    ):
        assert path.is_file(), path

    evidence = json.loads(evidence_report.read_text())
    assert evidence["status"] == "failed"
    assert evidence["phase"] == "pre_smoke_resource_guard"
    assert evidence["image_ref"] == image_ref
    assert evidence["release_tag"] is None
    assert evidence["allow_override_env"] == (
        "GB10_RELEASE_ALLOW_EXISTING_VLLM_CONTAINERS=1"
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "partial"
    assert metadata["source"]["image_ref"] == image_ref
    assert metadata["source"]["release_tag"] is None
    assert metadata["release_gate_summary"] == {
        "present": True,
        "status": "failed",
        "release_gate_passed": None,
        "failure_count": None,
    }
    assert sorted(metadata["missing_reports"]) == [
        "gb10-nvfp4-smoke.json",
        "gb10-openai-server-smoke-image.json",
    ]
    assert sorted(metadata["missing_evidence_files"]) == [
        "gb10-nvfp4-smoke.json",
        "gb10-openai-server-smoke-image.json",
        "gb10-smoked-image-digest.txt",
    ]


def test_gb10_release_image_smoke_rejects_release_tag_without_local_provenance(
    tmp_path,
):
    script = REPO_ROOT / "scripts" / "gb10-smoke-release-image.sh"
    bundle_name = "tag-provenance-guard-evidence"
    image_ref = "ghcr.io/gardner/vllm-gb10:gb10-test"
    release_tag = "gb10-test-release"
    fake_bin = tmp_path / "bin"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "evidence"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"unexpected docker invocation: $*\" >&2\n"
        "exit 64\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "GB10_RELEASE_SMOKE_REPORT_DIR": str(report_dir),
            "GB10_RELEASE_EVIDENCE_OUTPUT_DIR": str(output_dir),
            "GB10_RELEASE_EVIDENCE_BUNDLE_NAME": bundle_name,
            "GB10_RELEASE_TAG": release_tag,
            "GB10_NVFP4_MODEL": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        }
    )

    proc = subprocess.run(
        [
            str(script),
            image_ref,
            "--serve",
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "GB10 release-tag smoke requires release provenance" in proc.stderr
    assert "unexpected docker invocation" not in proc.stderr

    evidence_report = report_dir / "gb10-release-evidence-image.json"
    metadata_path = output_dir / "release-evidence-metadata.json"
    archive_path = output_dir / f"{bundle_name}.tar.gz"
    for path in (evidence_report, metadata_path, archive_path):
        assert path.is_file(), path

    evidence = json.loads(evidence_report.read_text())
    assert evidence["status"] == "failed"
    assert evidence["phase"] == "pre_smoke_provenance_guard"
    assert evidence["image_ref"] == image_ref
    assert evidence["release_tag"] == release_tag
    assert evidence["release_manifest_json"] is None
    assert evidence["runtime_image_metadata_json"] is None

    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "partial"
    assert metadata["source"]["image_ref"] == image_ref
    assert metadata["source"]["release_tag"] == release_tag
    assert metadata["included_provenance"] == []


def test_gb10_release_image_smoke_rejects_incomplete_local_provenance(tmp_path):
    script = REPO_ROOT / "scripts" / "gb10-smoke-release-image.sh"
    bundle_name = "provenance-guard-evidence"
    image_ref = "ghcr.io/gardner/vllm-gb10:gb10-test"
    release_tag = "gb10-test-release"
    fake_bin = tmp_path / "bin"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "evidence"
    manifest_path = tmp_path / "gb10-release-manifest.json"
    fake_bin.mkdir()
    manifest_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"unexpected docker invocation: $*\" >&2\n"
        "exit 64\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "GB10_RELEASE_SMOKE_REPORT_DIR": str(report_dir),
            "GB10_RELEASE_EVIDENCE_OUTPUT_DIR": str(output_dir),
            "GB10_RELEASE_EVIDENCE_BUNDLE_NAME": bundle_name,
            "GB10_RELEASE_TAG": release_tag,
            "GB10_RELEASE_MANIFEST_JSON": str(manifest_path),
            "GB10_NVFP4_MODEL": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        }
    )

    proc = subprocess.run(
        [
            str(script),
            image_ref,
            "--serve",
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "GB10 release provenance requires both" in proc.stderr
    assert "unexpected docker invocation" not in proc.stderr

    evidence_report = report_dir / "gb10-release-evidence-image.json"
    metadata_path = output_dir / "release-evidence-metadata.json"
    archive_path = output_dir / f"{bundle_name}.tar.gz"
    for path in (evidence_report, metadata_path, archive_path):
        assert path.is_file(), path

    evidence = json.loads(evidence_report.read_text())
    assert evidence["status"] == "failed"
    assert evidence["phase"] == "pre_smoke_provenance_guard"
    assert evidence["image_ref"] == image_ref
    assert evidence["release_tag"] == release_tag
    assert evidence["release_manifest_json"] == str(manifest_path)
    assert evidence["runtime_image_metadata_json"] is None

    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "partial"
    assert metadata["source"]["image_ref"] == image_ref
    assert metadata["source"]["release_tag"] == release_tag
    assert {item["kind"] for item in metadata["included_provenance"]} == {
        "release_manifest"
    }


def test_gb10_release_evidence_bundle_preserves_smoke_artifacts():
    script = (REPO_ROOT / "scripts" / "gb10-bundle-release-evidence.py").read_text()
    bundler = _load_gb10_release_bundle_module()
    verifier = _load_gb10_release_evidence_module()
    contract = _load_gb10_release_contract_module()

    assert "Bundle GB10 release smoke reports" in script
    assert "EXPECTED_REPORTS" in script
    assert "EXPECTED_EVIDENCE_FILES" in script
    assert "gb10_release_contract" in script
    assert bundler.EXPECTED_REPORTS == GB10_EXPECTED_RELEASE_REPORTS
    assert bundler.EXPECTED_EVIDENCE_FILES == GB10_EXPECTED_RELEASE_EVIDENCE_FILES
    assert GB10_EXPECTED_RELEASE_REPORTS[0] == (
        contract.RELEASE_NVFP4_SMOKE_REPORT_FILE
    )
    assert GB10_EXPECTED_RELEASE_REPORTS[1] == (
        contract.RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE
    )
    assert GB10_EXPECTED_RELEASE_REPORTS[2] == (
        contract.RELEASE_EVIDENCE_SUMMARY_REPORT_FILE
    )
    assert GB10_RELEASE_SMOKED_IMAGE_DIGEST_FILE == (
        contract.RELEASE_SMOKED_IMAGE_DIGEST_FILE
    )
    assert bundler.RELEASE_EVIDENCE_SUMMARY_REPORT_FILE == (
        contract.RELEASE_EVIDENCE_SUMMARY_REPORT_FILE
    )
    assert bundler.RELEASE_SMOKED_IMAGE_DIGEST_FILE == (
        contract.RELEASE_SMOKED_IMAGE_DIGEST_FILE
    )
    assert verifier.RELEASE_NVFP4_SMOKE_REPORT_FILE == (
        contract.RELEASE_NVFP4_SMOKE_REPORT_FILE
    )
    assert verifier.RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE == (
        contract.RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE
    )
    assert "GB10_RELEASE_EVIDENCE_REPORT_DIR" in script
    assert "GB10_RELEASE_EVIDENCE_OUTPUT_DIR" in script
    assert "GB10_RELEASE_EVIDENCE_IMAGE_REF" in script
    assert "GB10_RELEASE_EVIDENCE_RELEASE_TAG" in script
    assert "GB10_RELEASE_EVIDENCE_COMMIT" in script
    assert "GB10_RELEASE_EVIDENCE_MANIFEST_JSON" in script
    assert "GB10_RELEASE_EVIDENCE_RUNTIME_IMAGE_METADATA_JSON" in script
    assert "--gb10-release-manifest-json" in script
    assert "--gb10-runtime-image-metadata-json" in script
    assert "included_provenance" in script
    assert bundler.PROVENANCE_RELATIVE_PATHS == {
        "release_manifest": "provenance/gb10-release-manifest.json",
        "runtime_image_metadata": "provenance/buildx-runtime-image-metadata.json",
    }
    assert "--gb10-allow-partial" in script
    assert "--gb10-include-glob" in script
    assert "gb10-*.txt" in script
    assert "RELEASE_EVIDENCE_METADATA_FILE" in script
    assert "RELEASE_EVIDENCE_CHECKSUM_FILE" in script
    assert "RELEASE_EVIDENCE_SUMMARY_REPORT_FILE" in script
    assert "RELEASE_SMOKED_IMAGE_DIGEST_FILE" in script
    assert '"gb10-release-evidence-image.json"' not in script
    assert '"gb10-smoked-image-digest.txt"' not in script
    assert "default_release_evidence_bundle_name" in script
    assert "release_evidence_bundle_archive_name" in script
    assert "release_evidence_bundle_archive_checksum_name" in script
    assert "release_evidence_asset_paths" in script
    assert bundler.RELEASE_EVIDENCE_METADATA_FILE == "release-evidence-metadata.json"
    assert bundler.RELEASE_EVIDENCE_CHECKSUM_FILE == "SHA256SUMS"
    assert bundler.release_evidence_bundle_archive_name() == (
        "gb10-release-evidence.tar.gz"
    )
    assert bundler.release_evidence_bundle_archive_checksum_name() == (
        "gb10-release-evidence.tar.gz.sha256"
    )
    assert '"report_summaries"' in script
    assert '"release_gate_summary"' in script
    assert '"release_gate_passed"' in script
    assert '"smoked_image_digest"' in script
    assert '"support_matrix_summary"' in script
    assert '"support_matrix_complete"' in script
    assert "REQUIRED_GB10_SUPPORT_MATRIX" in script
    assert '"required_entries"' in script
    assert '"missing_required_entries"' in script
    assert '"mismatched_required_entries"' in script
    assert "status_counts" in script
    assert '"failure_count"' in script
    assert bundler.SHA256_DIGEST_RE.pattern == "sha256:[0-9a-f]{64}"
    assert "tarfile.open" in script
    assert "hashlib.sha256" in script
    assert "_metadata_status(" in script


def test_gb10_release_evidence_bundle_builds_metadata_and_tarball(tmp_path):
    bundler = _load_gb10_release_bundle_module()
    validator = _load_gb10_release_asset_validator_module()
    lister = _load_gb10_evidence_release_asset_lister_module()
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "bundle"
    report_dir.mkdir()

    report_payloads = {
        "gb10-nvfp4-smoke.json": {"status": "passed", "kind": "offline"},
        "gb10-openai-server-smoke-image.json": {
            "status": "passed",
            "kind": "openai",
        },
        "gb10-release-evidence-image.json": {
            "status": "passed",
            "release_gate_passed": True,
            "failure_count": 0,
        },
        "gb10-extra-local-note.json": {"status": "partial"},
    }
    for name, payload in report_payloads.items():
        (report_dir / name).write_text(json.dumps(payload) + "\n")
    (report_dir / "gb10-smoked-image-digest.txt").write_text(
        "ghcr.io/gardner/vllm-gb10@sha256:" + "a" * 64 + "\n"
    )

    manifest_path = tmp_path / "gb10-release-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dependencies": {
                    "flashinfer": {
                        "all_required_components_present": True,
                    }
                },
                "gb10_support_matrix": {
                    "architecture": "sm_121a",
                    "first_release_scope": "single_spark_first_path",
                    "status_definitions": {
                        "supported_native": "Runs native SM121A code.",
                        "not_supported": "Rejected for GB10 release evidence.",
                        "deferred": "Not required for the first release.",
                    },
                    "entries": {
                        name: {"status": status}
                        for name, status in GB10_REQUIRED_SUPPORT_MATRIX.items()
                    },
                },
            }
        )
        + "\n"
    )
    runtime_metadata_path = tmp_path / "buildx-runtime-image-metadata.json"
    runtime_metadata_path.write_text(json.dumps({"containerimage.digest": "sha256:x"}))

    exit_code = bundler.main(
        [
            "--gb10-report-dir",
            str(report_dir),
            "--gb10-output-dir",
            str(output_dir),
            "--gb10-bundle-name",
            "evidence",
            "--gb10-image-ref",
            "ghcr.io/gardner/vllm-gb10:test",
            "--gb10-release-tag",
            "gb10-vllm-test",
            "--gb10-commit",
            "abc123",
            "--gb10-release-manifest-json",
            str(manifest_path),
            "--gb10-runtime-image-metadata-json",
            str(runtime_metadata_path),
        ]
    )

    assert exit_code == 0
    metadata_path = output_dir / bundler.RELEASE_EVIDENCE_METADATA_FILE
    checksum_path = output_dir / bundler.RELEASE_EVIDENCE_CHECKSUM_FILE
    archive_path = output_dir / bundler.release_evidence_bundle_archive_name(
        "evidence"
    )
    archive_checksum_path = (
        output_dir / bundler.release_evidence_bundle_archive_checksum_name("evidence")
    )
    assert metadata_path.exists()
    assert checksum_path.exists()
    assert archive_path.exists()
    assert archive_checksum_path.exists()
    listed_assets, list_errors = lister.list_release_evidence_assets(
        output_dir=output_dir,
        bundle_name="evidence",
    )
    assert list_errors == []
    assert listed_assets == [
        archive_path,
        archive_checksum_path,
        metadata_path,
        checksum_path,
    ]
    assert validator.validate_release_assets(
        output_dir=output_dir,
        bundle_name="evidence",
    ) == []

    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "complete"
    assert metadata["release_gate_passed"] is True
    assert metadata["release_gate_summary"] == {
        "present": True,
        "status": "passed",
        "release_gate_passed": True,
        "failure_count": 0,
    }
    assert metadata["smoked_image_digest"] == {
        "present": True,
        "raw": "ghcr.io/gardner/vllm-gb10@sha256:" + "a" * 64,
        "digest": "sha256:" + "a" * 64,
    }
    expected_status_counts = {
        status: list(GB10_REQUIRED_SUPPORT_MATRIX.values()).count(status)
        for status in sorted(set(GB10_REQUIRED_SUPPORT_MATRIX.values()))
    }
    assert metadata["support_matrix_summary"] == {
        "present": True,
        "release_manifest_present": True,
        "architecture": "sm_121a",
        "first_release_scope": "single_spark_first_path",
        "status_definitions": {
            "supported_native": "Runs native SM121A code.",
            "not_supported": "Rejected for GB10 release evidence.",
            "deferred": "Not required for the first release.",
        },
        "entry_count": len(GB10_REQUIRED_SUPPORT_MATRIX),
        "status_counts": expected_status_counts,
        "required_entries": GB10_REQUIRED_SUPPORT_MATRIX,
        "missing_required_entries": [],
        "mismatched_required_entries": {},
        "entries": GB10_REQUIRED_SUPPORT_MATRIX,
        "invalid_entries": [],
    }
    assert metadata["support_matrix_complete"] is True
    assert metadata["missing_reports"] == []
    assert metadata["missing_evidence_files"] == []
    assert metadata["report_summaries"] == {
        "gb10-nvfp4-smoke.json": {
            "present": True,
            "status": "passed",
        },
        "gb10-openai-server-smoke-image.json": {
            "present": True,
            "status": "passed",
        },
        "gb10-release-evidence-image.json": {
            "present": True,
            "status": "passed",
            "release_gate_passed": True,
            "failure_count": 0,
        },
    }
    assert metadata["source"] == {
        "report_dir": str(report_dir.resolve()),
        "commit": "abc123",
        "release_tag": "gb10-vllm-test",
        "image_ref": "ghcr.io/gardner/vllm-gb10:test",
        "image_digest": "sha256:" + "a" * 64,
    }
    assert {
        item["relative_path"]: item["source_path"]
        for item in metadata["included_provenance"]
    } == {
        "provenance/gb10-release-manifest.json": str(manifest_path.resolve()),
        "provenance/buildx-runtime-image-metadata.json": str(
            runtime_metadata_path.resolve()
        ),
    }
    assert {
        report["name"]: report["present"] for report in metadata["expected_reports"]
    } == {
        "gb10-nvfp4-smoke.json": True,
        "gb10-openai-server-smoke-image.json": True,
        "gb10-release-evidence-image.json": True,
    }
    assert {
        evidence_file["name"]: evidence_file["present"]
        for evidence_file in metadata["expected_evidence_files"]
    } == {
        "gb10-nvfp4-smoke.json": True,
        "gb10-openai-server-smoke-image.json": True,
        "gb10-release-evidence-image.json": True,
        "gb10-smoked-image-digest.txt": True,
    }
    included_paths = {
        item["relative_path"] for item in metadata["included_files"]
    }
    assert included_paths == {
        "reports/gb10-nvfp4-smoke.json",
        "reports/gb10-openai-server-smoke-image.json",
        "reports/gb10-release-evidence-image.json",
        "reports/gb10-extra-local-note.json",
        "reports/gb10-smoked-image-digest.txt",
    }

    checksum_text = checksum_path.read_text()
    assert "reports/gb10-nvfp4-smoke.json" in checksum_text
    assert "reports/gb10-smoked-image-digest.txt" in checksum_text
    assert "provenance/gb10-release-manifest.json" in checksum_text
    assert "provenance/buildx-runtime-image-metadata.json" in checksum_text
    assert "release-evidence-metadata.json" in checksum_text

    with tarfile.open(archive_path, "r:gz") as tar:
        tar_names = set(tar.getnames())
    assert "evidence/SHA256SUMS" in tar_names
    assert "evidence/release-evidence-metadata.json" in tar_names
    assert "evidence/reports/gb10-nvfp4-smoke.json" in tar_names
    assert "evidence/reports/gb10-openai-server-smoke-image.json" in tar_names
    assert "evidence/reports/gb10-release-evidence-image.json" in tar_names
    assert "evidence/reports/gb10-smoked-image-digest.txt" in tar_names
    assert "evidence/provenance/gb10-release-manifest.json" in tar_names
    assert "evidence/provenance/buildx-runtime-image-metadata.json" in tar_names


def test_gb10_release_evidence_bundle_failure_leaves_no_partial_outputs(
    tmp_path,
):
    bundler = _load_gb10_release_bundle_module()
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "bundle"
    report_dir.mkdir()

    (report_dir / "gb10-release-evidence-image.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "release_gate_passed": False,
                "failure_count": 3,
            }
        )
        + "\n"
    )

    exit_code = bundler.main(
        [
            "--gb10-report-dir",
            str(report_dir),
            "--gb10-output-dir",
            str(output_dir),
            "--gb10-bundle-name",
            "evidence",
        ]
    )

    assert exit_code == 1
    for path in (
        output_dir / "reports",
        output_dir / "provenance",
        output_dir / bundler.RELEASE_EVIDENCE_METADATA_FILE,
        output_dir / bundler.RELEASE_EVIDENCE_CHECKSUM_FILE,
        output_dir / bundler.release_evidence_bundle_archive_name("evidence"),
        output_dir / bundler.release_evidence_bundle_archive_checksum_name("evidence"),
    ):
        assert not path.exists(), path


def test_gb10_release_evidence_bundle_missing_provenance_leaves_no_partial_outputs(
    tmp_path,
):
    bundler = _load_gb10_release_bundle_module()
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "bundle"
    report_dir.mkdir()

    report_payloads = {
        "gb10-nvfp4-smoke.json": {"status": "passed"},
        "gb10-openai-server-smoke-image.json": {"status": "passed"},
        "gb10-release-evidence-image.json": {
            "status": "passed",
            "release_gate_passed": True,
            "failure_count": 0,
        },
    }
    for name, payload in report_payloads.items():
        (report_dir / name).write_text(json.dumps(payload) + "\n")
    (report_dir / "gb10-smoked-image-digest.txt").write_text(
        "ghcr.io/gardner/vllm-gb10@sha256:" + "d" * 64 + "\n"
    )

    exit_code = bundler.main(
        [
            "--gb10-report-dir",
            str(report_dir),
            "--gb10-output-dir",
            str(output_dir),
            "--gb10-bundle-name",
            "evidence",
            "--gb10-release-manifest-json",
            str(tmp_path / "missing-manifest.json"),
        ]
    )

    assert exit_code == 1
    for path in (
        output_dir / "reports",
        output_dir / "provenance",
        output_dir / bundler.RELEASE_EVIDENCE_METADATA_FILE,
        output_dir / bundler.RELEASE_EVIDENCE_CHECKSUM_FILE,
        output_dir / bundler.release_evidence_bundle_archive_name("evidence"),
        output_dir / bundler.release_evidence_bundle_archive_checksum_name("evidence"),
    ):
        assert not path.exists(), path


def test_gb10_release_evidence_bundle_marks_failed_gate(tmp_path):
    bundler = _load_gb10_release_bundle_module()
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "bundle"
    report_dir.mkdir()

    report_payloads = {
        "gb10-nvfp4-smoke.json": {"status": "passed"},
        "gb10-openai-server-smoke-image.json": {"status": "passed"},
        "gb10-release-evidence-image.json": {
            "status": "failed",
            "release_gate_passed": False,
            "failure_count": 2,
        },
    }
    for name, payload in report_payloads.items():
        (report_dir / name).write_text(json.dumps(payload) + "\n")
    (report_dir / "gb10-smoked-image-digest.txt").write_text(
        "ghcr.io/gardner/vllm-gb10@sha256:" + "b" * 64 + "\n"
    )

    exit_code = bundler.main(
        [
            "--gb10-report-dir",
            str(report_dir),
            "--gb10-output-dir",
            str(output_dir),
            "--gb10-bundle-name",
            "evidence",
        ]
    )

    assert exit_code == 0
    metadata = json.loads(
        (output_dir / bundler.RELEASE_EVIDENCE_METADATA_FILE).read_text()
    )
    assert metadata["status"] == "failed"
    assert metadata["release_gate_passed"] is False
    assert metadata["release_gate_summary"] == {
        "present": True,
        "status": "failed",
        "release_gate_passed": False,
        "failure_count": 2,
    }
    assert metadata["missing_reports"] == []
    assert metadata["missing_evidence_files"] == []


def test_gb10_release_evidence_bundle_marks_missing_support_matrix_partial(
    tmp_path,
):
    bundler = _load_gb10_release_bundle_module()
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "bundle"
    report_dir.mkdir()

    report_payloads = {
        "gb10-nvfp4-smoke.json": {"status": "passed"},
        "gb10-openai-server-smoke-image.json": {"status": "passed"},
        "gb10-release-evidence-image.json": {
            "status": "passed",
            "release_gate_passed": True,
            "failure_count": 0,
        },
    }
    for name, payload in report_payloads.items():
        (report_dir / name).write_text(json.dumps(payload) + "\n")
    (report_dir / "gb10-smoked-image-digest.txt").write_text(
        "ghcr.io/gardner/vllm-gb10@sha256:" + "c" * 64 + "\n"
    )

    exit_code = bundler.main(
        [
            "--gb10-report-dir",
            str(report_dir),
            "--gb10-output-dir",
            str(output_dir),
            "--gb10-bundle-name",
            "evidence",
        ]
    )

    assert exit_code == 0
    metadata = json.loads(
        (output_dir / bundler.RELEASE_EVIDENCE_METADATA_FILE).read_text()
    )
    assert metadata["status"] == "partial"
    assert metadata["release_gate_passed"] is True
    assert metadata["support_matrix_complete"] is False
    assert metadata["support_matrix_summary"] == {
        "present": False,
        "release_manifest_present": False,
        "architecture": None,
        "first_release_scope": None,
        "status_definitions": {},
        "entry_count": 0,
        "status_counts": {},
        "required_entries": GB10_REQUIRED_SUPPORT_MATRIX,
        "missing_required_entries": sorted(GB10_REQUIRED_SUPPORT_MATRIX),
        "mismatched_required_entries": {},
        "entries": {},
        "invalid_entries": [],
        "reason": "release manifest was not included",
    }


def test_gb10_openai_server_smoke_reports_api_evidence():
    script = (REPO_ROOT / "scripts" / "gb10-smoke-openai-server.py").read_text()

    assert "OpenAI-compatible" in script
    assert "already running server" in script
    assert "GB10_OPENAI_BASE_URL" in script
    assert "GB10_OPENAI_MODEL" in script
    assert "GB10_OPENAI_API_KEY" in script
    assert "OPENAI_API_KEY" in script
    assert "/v1/models" in script
    assert "/v1/completions" in script
    assert "/v1/chat/completions" in script
    assert "reasoning_content" in script
    assert '"reasoning")' in script
    assert 'f"message.{field_name}"' in script
    assert "--gb10-endpoint" in script
    assert "--gb10-report-json" in script
    assert '"schema_version": 1' in script
    assert '"served_models": served_models' in script
    assert '"selected": selected' in script
    assert '"system_fingerprint": completion_body.get("system_fingerprint")' in script
    assert '"finish_reason": first_choice.get("finish_reason")' in script
    assert '"stop_reason": first_choice.get("stop_reason")' in script
    assert "--gb10-repeat-count" in script
    assert "--gb10-require-deterministic" in script
    assert '"deterministic_generation"' in script
    assert '"responses": response_summaries' in script
    assert '"gb10_release_evidence"' in script
    assert '"openai_compatible_server_smoke"' in script
    assert '"offline NVFP4 backend-selection smoke report"' in script
    assert "native backend evidence comes from the" in script
    assert "_request_with_retries" in script
    assert "_extract_generated_text" in script
    assert "_extract_generated_text_with_source" in script
    assert "_build_report" in script
    assert "raise SystemExit(main())" in script


def test_gb10_openai_server_smoke_extracts_text_and_builds_report():
    smoke = _load_gb10_openai_smoke_module()

    assert (
        smoke._extract_generated_text(
            "completions",
            {"choices": [{"text": " native GB10 output"}]},
        )
        == " native GB10 output"
    )
    assert (
        smoke._extract_generated_text(
            "chat",
            {"choices": [{"message": {"content": "native chat output"}}]},
        )
        == "native chat output"
    )
    assert (
        smoke._extract_generated_text(
            "chat",
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "native "},
                                {"type": "text", "text": "list output"},
                            ]
                        }
                    }
                ]
            },
        )
        == "native list output"
    )
    generated_text, generated_text_source = smoke._extract_generated_text_with_source(
        "chat",
        {"choices": [{"message": {"content": None, "reasoning": "native reasoning"}}]},
    )
    assert generated_text == "native reasoning"
    assert generated_text_source == "message.reasoning"

    args = SimpleNamespace(
        gb10_endpoint="chat",
        gb10_prompt="hello",
        gb10_max_tokens=8,
        gb10_temperature=0.0,
        gb10_seed=0,
        gb10_timeout=30.0,
        gb10_retries=1,
        gb10_repeat_count=2,
        gb10_require_deterministic=True,
    )
    response_summaries = [
        {
            "status": 200,
            "id": "chatcmpl-gb10-a",
            "object": "chat.completion",
            "created": 456,
            "model": "gb10-model",
            "system_fingerprint": "vllm-0.22.1rc1.dev-gb10",
            "choice": {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
            },
            "generated_text": "native chat output",
            "generated_text_source": "message.content",
            "usage": {"completion_tokens": 3},
        },
        {
            "status": 200,
            "id": "chatcmpl-gb10-b",
            "object": "chat.completion",
            "created": 457,
            "model": "gb10-model",
            "system_fingerprint": "vllm-0.22.1rc1.dev-gb10",
            "choice": {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
            },
            "generated_text": "native chat output",
            "generated_text_source": "message.content",
            "usage": {"completion_tokens": 3},
        },
    ]
    report = smoke._build_report(
        args=args,
        base_url="http://127.0.0.1:8000",
        model="gb10-model",
        models_status=200,
        models_body={
            "data": [
                {
                    "id": "gb10-model",
                    "object": "model",
                    "created": 123,
                    "owned_by": "vllm",
                    "root": "org/gb10-model",
                    "parent": None,
                    "max_model_len": 4096,
                    "permission": [{"allow_sampling": True}],
                }
            ]
        },
        completion_status=200,
        completion_body={
            "id": "chatcmpl-gb10",
            "object": "chat.completion",
            "created": 456,
            "model": "gb10-model",
            "system_fingerprint": "vllm-0.22.1rc1.dev-gb10",
            "choices": [
                {
                    "index": 0,
                    "message": {"content": "native chat output"},
                    "finish_reason": "stop",
                    "stop_reason": None,
                }
            ],
            "usage": {"completion_tokens": 3},
        },
        generated_text="native chat output",
        generated_text_source="message.content",
        response_summaries=response_summaries,
        status="passed",
    )

    assert report["status"] == "passed"
    assert report["model"] == "gb10-model"
    assert report["models"]["ids"] == ["gb10-model"]
    assert report["models"]["raw_count"] == 1
    assert report["models"]["served_models"] == [
        {
            "id": "gb10-model",
            "object": "model",
            "created": 123,
            "owned_by": "vllm",
            "root": "org/gb10-model",
            "parent": None,
            "max_model_len": 4096,
        }
    ]
    assert report["models"]["selected"] == {
        "id": "gb10-model",
        "object": "model",
        "created": 123,
        "owned_by": "vllm",
        "root": "org/gb10-model",
        "parent": None,
        "max_model_len": 4096,
    }
    assert report["endpoint"] == {
        "name": "chat",
        "path": "/v1/chat/completions",
    }
    assert report["response"]["id"] == "chatcmpl-gb10-a"
    assert report["response"]["object"] == "chat.completion"
    assert report["response"]["created"] == 456
    assert report["response"]["model"] == "gb10-model"
    assert report["response"]["system_fingerprint"] == "vllm-0.22.1rc1.dev-gb10"
    assert report["response"]["choice"] == {
        "index": 0,
        "finish_reason": "stop",
        "stop_reason": None,
    }
    assert report["response"]["usage"] == {"completion_tokens": 3}
    assert report["response"]["generated_text_source"] == "message.content"
    assert report["responses"] == response_summaries
    assert report["deterministic_generation"] == {
        "status": "passed",
        "repeat_count": 2,
        "unique_generated_text_count": 1,
        "generated_texts_match": True,
    }
    assert report["gb10_release_evidence"]["openai_compatible_server_smoke"] == {
        "status": "passed",
        "endpoint": "/v1/chat/completions",
        "generated_text_observed": True,
    }
    assert report["gb10_release_evidence"]["deterministic_generation"] == {
        "status": "passed",
        "repeat_count": 2,
        "unique_generated_text_count": 1,
        "generated_texts_match": True,
    }
    assert report["gb10_release_evidence"]["release_ready"] is False
    assert "CUDA graph capture/replay validation" in report[
        "gb10_release_evidence"
    ]["remaining_release_evidence"]


def test_gb10_release_evidence_verifier_checks_required_smoke_reports():
    script = (REPO_ROOT / "scripts" / "gb10-verify-release-evidence.py").read_text()

    assert "scripts/gb10-smoke-nvfp4.py" in script
    assert "scripts/gb10-smoke-openai-server.py" in script
    assert "--gb10-nvfp4-report-json" in script
    assert "--gb10-openai-report-json" in script
    assert "--gb10-release-manifest-json" in script
    assert "--gb10-runtime-image-metadata-json" in script
    assert "--gb10-image-ref" in script
    assert "--gb10-image-digest" in script
    assert "--gb10-release-tag" in script
    assert "--gb10-output-json" in script
    assert "--gb10-require-moe" in script
    assert "--gb10-require-openai-deterministic" in script
    assert "--gb10-allow-partial" in script
    assert '"release_gate_passed"' in script
    assert '"native_nvfp4_gemm_observed"' in script
    assert '"native_nvfp4_moe_non_ep_observed"' in script
    assert '"openai_deterministic_generation"' in script
    assert '"gb10_device_sm121"' in script
    assert '"flashinfer_gb10_runtime_version"' in script
    assert '"flashinfer_gb10_distribution_versions"' in script
    assert '"kv_cache_fp8_e4m3"' in script
    assert '"attention_backend_flashinfer"' in script
    assert '"attention_backend_allowed_by_support_matrix"' in script
    assert '"cuda_graph_capture_replay"' in script
    assert '"model_shape_reported"' in script
    assert '"quantization_modelopt_fp4"' in script
    assert '"quantization_allowed_by_support_matrix"' in script
    assert '"nvfp4_backend_selections_allowed_by_support_matrix"' in script
    assert '"release_manifest_flashinfer_components"' in script
    assert '"release_manifest_durable_inputs"' in script
    assert '"release_manifest_source_dependencies_present"' in script
    assert '"release_manifest_gb10_support_matrix"' in script
    assert '"release_manifest_source_refs_pinned"' in script
    assert '"release_manifest_no_local_deps"' in script
    assert '"release_manifest_image_pushed_for_tagged_release"' in script
    assert '"release_manifest_image_ref_matches_smoke"' in script
    assert '"release_manifest_tag_matches_expected"' in script
    assert '"runtime_image_metadata_has_digest"' in script
    assert '"runtime_image_digest_matches_smoke"' in script
    assert "GB10 release evidence gate failed" in script


def test_gb10_release_evidence_verifier_builds_gate_summary():
    verifier = _load_gb10_release_evidence_module()
    nvfp4_report = {
        "status": "passed",
        "backend_selections": [
            {
                "path": "linear",
                "backend": "FlashInferB12xNvFp4LinearKernel",
                "is_fallback": False,
            },
            {
                "path": "moe",
                "backend": "FLASHINFER_B12X",
                "is_fallback": False,
            },
        ],
        "fallback_events": [],
        "runtime": {
            "cuda_available": True,
            "device_name": "NVIDIA GB10",
            "device_capability": {
                "major": 12,
                "minor": 1,
                "arch": "sm_121",
            },
            "flashinfer_version": "0.6.12+cu130gb10",
            "flashinfer_distributions": {
                "flashinfer-python": "0.6.12+cu130gb10",
                "flashinfer-cubin": "0.6.12+cu130gb10",
                "flashinfer-jit-cache": "0.6.12+cu130gb10",
            },
        },
        "backend_summary": {
            "capabilities": {
                "native_nvfp4_gemm": {"status": "observed"},
                "native_nvfp4_moe_non_ep": {"status": "observed"},
            }
        },
        "gb10_release_summary": {
            "first_path_smoke_passed": True,
            "smoke_blockers": [],
            "unsupported_paths": {
                "public_flashattention_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "public FlashAttention runtime is not validated",
                },
                "flashinfer_trtllm_nvfp4_dense": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "FlashInfer TRTLLM dense is not validated on SM12x",
                },
                "flashinfer_trtllm_mxfp4_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "TRTLLM MXFP4 MoE is not validated on SM121",
                },
                "flashinfer_cutedsl_nvfp4_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "generic FlashInfer CuteDSL NVFP4 MoE is not validated "
                        "on SM12x"
                    ),
                },
                "trtllm_gen_attention": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "TRTLLM Gen attention rejects SM121",
                },
                "triton_attention_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Triton attention lacks native GB10 evidence",
                },
                "flex_attention_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "FlexAttention lacks native GB10 evidence",
                },
                "turboquant_attention": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "TurboQuant KV-cache compression lacks native GB10 "
                        "evidence"
                    ),
                },
                "triton_mla_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Triton MLA lacks native GB10 evidence",
                },
                "flashinfer_trtllm_mla_attention": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "FlashInfer TRT-LLM MLA lacks native GB10 evidence",
                },
                "flashinfer_trtllm_sparse_mla_attention": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "FlashInfer TRT-LLM Sparse MLA lacks native GB10 "
                        "evidence"
                    ),
                },
                "public_flashattention_mla_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "public FlashAttention MLA lacks native GB10 evidence",
                },
                "cutlass_mla_sm100_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "SM100 CUTLASS MLA lacks native GB10 evidence",
                },
                "tokenspeed_mla_cutedsl_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "TokenSpeed CuTe DSL MLA lacks native GB10 evidence",
                },
                "triton_mamba_ssu_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Triton Mamba SSU lacks native GB10 evidence",
                },
                "mamba1_triton_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Mamba1 runtime lacks native GB10 evidence",
                },
                "mamba2_triton_ssd_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Mamba2 Triton SSD runtime lacks native GB10 evidence",
                },
                "short_conv_triton_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "ShortConv runtime lacks native GB10 evidence",
                },
                "linear_attention_triton_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Linear attention runtime lacks native GB10 evidence",
                },
                "speculative_decoding_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Speculative decoding runtime lacks native GB10 evidence"
                    ),
                },
                "pooling_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Pooling runtime lacks native GB10 evidence",
                },
                "reasoning_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Reasoning runtime lacks native GB10 evidence",
                },
                "structured_outputs_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Structured outputs runtime lacks native GB10 evidence"
                    ),
                },
                "openai_tool_calling_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "OpenAI tool-calling runtime lacks native GB10 evidence"
                    ),
                },
                "lora_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "LoRA runtime lacks native GB10 evidence",
                },
                "gdn_prefill_triton_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "GDN prefill Triton/FLA lacks native GB10 evidence",
                },
                "gdn_prefill_cutedsl_backend": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "GDN prefill CuteDSL lacks native GB10 evidence",
                },
                "mm_encoder_fp8_attention": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "MM encoder FP8 attention lacks native GB10 evidence"
                    ),
                },
                "mm_encoder_public_flashattention_backend": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "public FlashAttention MM encoder attention lacks "
                        "native GB10 evidence"
                    ),
                },
                "mm_encoder_triton_attention_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Triton MM encoder attention lacks native GB10 evidence"
                    ),
                },
                "mm_encoder_torch_sdpa_attention_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Torch SDPA MM encoder attention lacks native GB10 "
                        "evidence"
                    ),
                },
                "trtllm_gen_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "TRTLLM Gen MoE rejects SM121",
                },
                "marlin_nvfp4_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Marlin is a fallback path",
                },
                "fbgemm_nvfp4_dense": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "FBGEMM NVFP4 dense is not native GB10 evidence",
                },
                "modelopt_w4a16_nvfp4_checkpoint_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "ModelOpt W4A16 NVFP4 checkpoint loading is not validated"
                    ),
                },
                "modelopt_nvfp4_kv_cache_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "ModelOpt NVFP4 KV-cache loading can auto-select "
                        "vLLM kv_cache_dtype='nvfp4' without native SM12x "
                        "NVFP4 KV-cache correctness evidence"
                    ),
                },
                "nvfp4_kv_cache_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "User-selected vLLM kv_cache_dtype='nvfp4' lacks "
                        "native SM12x NVFP4 KV-cache correctness evidence"
                    ),
                },
                "unvalidated_kv_cache_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "E5M2, Gaudi FP8, and per-token-head KV-cache "
                        "runtime dtypes lack native SM12x correctness evidence"
                    ),
                },
                "kv_events_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "KV cache event publishing lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "kv_offload_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "KV offload changes KV allocation and slot-mapping "
                        "outside native SM12x correctness evidence"
                    ),
                },
                "kv_transfer_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Distributed KV transfer and disaggregated serving "
                        "lack native SM12x correctness evidence"
                    ),
                },
                "ubatching_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Dual batch overlap and ubatching lack native SM12x "
                        "scheduler and DeepEP all-to-all correctness evidence"
                    ),
                },
                "distributed_parallel_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Distributed parallel process topologies lack native "
                        "SM12x distributed correctness evidence"
                    ),
                },
                "kv_sharing_fast_prefill_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "KV sharing fast prefill lacks native SM12x attention "
                        "metadata and logits-indexing correctness evidence"
                    ),
                },
                "ec_transfer_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Distributed EC cache transfer lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "weight_transfer_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "RL training weight transfer lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "return_routed_experts_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Routed experts capture lacks native SM12x correctness "
                        "evidence"
                    ),
                },
                "logprobs_logits_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "raw_logits and processed_logits logprobs modes lack "
                        "native SM12x logits-return correctness evidence"
                    ),
                },
                "custom_logits_processors_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "custom logits processor hooks lack native SM12x "
                        "correctness evidence"
                    ),
                },
                "io_processor_plugin_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "IO processor plugin code lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "hf_overrides_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "HF config overrides lack native SM12x correctness "
                        "evidence"
                    ),
                },
                "transformers_model_impl_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Transformers model implementation runtime lacks "
                        "native SM12x correctness evidence"
                    ),
                },
                "trust_remote_code_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "trusted remote model code lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "custom_scheduler_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "custom scheduler class runtime lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "custom_worker_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "custom worker class runtime lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "prompt_embeds_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "prompt embeds input handling lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "stock_torch_compile_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "stock torch.compile lacks native SM12x correctness "
                        "evidence"
                    ),
                },
                "mamba_align_cache_runtime": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Mamba align-cache state handling lacks native SM12x "
                        "correctness evidence"
                    ),
                },
                "compressed_tensors_fp4_kv_cache_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors FP4 KV-cache loading is not supported "
                        "in vLLM and can masquerade as the FP8 KV-cache path"
                    ),
                },
                "marlin_mxfp4_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "MXFP4 Marlin is a fallback path",
                },
                "mxfp4_moe_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "MXFP4 MoE fallback paths are not native GB10 evidence",
                },
                "gpt_oss_triton_mxfp4_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "GPT-OSS Triton MXFP4 MoE is not native GB10 evidence"
                    ),
                },
                "public_mxfp4_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Public MXFP4 quantization can select unquantized "
                        "linear/attention handling and MXFP4 MoE backend "
                        "selection without native GB10 evidence"
                    ),
                },
                "public_fp8_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Public FP8 quantization can select FP8 dense kernel "
                        "selection and FP8 MoE backend selection without "
                        "native GB10 evidence"
                    ),
                },
                "deepseek_v4_fp8_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "DeepSeek V4 FP8 quantization can select FP8 "
                        "block-quantized linear/attention layers and FP8, "
                        "MXFP4, or ModelOpt NVFP4 MoE dispatch without "
                        "native GB10 DeepSeek V4 evidence"
                    ),
                },
                "torchao_fp8_activation_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "TorchAO FP8 activation quantization can call "
                        "torchao.quantization.quantize_ and hardware packing "
                        "without native GB10 TorchAO FP8 evidence"
                    ),
                },
                "torchao_weight_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "TorchAO weight quantization can call "
                        "torchao.quantization.quantize_ and hardware packing "
                        "without native GB10 TorchAO weight evidence"
                    ),
                },
                "bitsandbytes_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "BitsAndBytes quantization can select bitsandbytes "
                        "4-bit linear kernels, 8-bit matmul kernels, or MoE "
                        "handling without native GB10 BitsAndBytes evidence"
                    ),
                },
                "awq_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "AWQ quantization can select AWQ dense kernels, "
                        "AWQ-Marlin dense kernels, AWQ-Marlin MoE, or Moe "
                        "WNA16 fallback handling without native GB10 AWQ "
                        "evidence"
                    ),
                },
                "gptq_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "GPTQ quantization can select GPTQ dense kernels, "
                        "AutoGPTQ-Marlin MoE, or Moe WNA16 fallback handling "
                        "without native GB10 GPTQ evidence"
                    ),
                },
                "inc_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "INC/AutoRound quantization can select AWQ or GPTQ "
                        "Marlin dense kernels, AWQ/GPTQ MoE, or Moe WNA16 "
                        "fallback handling without native GB10 INC/AutoRound "
                        "evidence"
                    ),
                },
                "gguf_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "GGUF quantization can select GGUF dense, embedding, "
                        "or MoE kernels and dequantization fallbacks without "
                        "native GB10 GGUF evidence"
                    ),
                },
                "humming_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Humming quantization can select Humming dense and "
                        "MoE kernels without native GB10 Humming evidence"
                    ),
                },
                "humming_mxfp4_moe_backend": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "The Humming MXFP4 MoE backend can select Humming "
                        "Mixed Precision kernels without native GB10 Humming "
                        "MXFP4 MoE evidence"
                    ),
                },
                "fp8_w8a16_marlin_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "FP8 W8A16 Marlin fallback is not native GB10 evidence"
                    ),
                },
                "fp8_w8a16_moe_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "FP8 MoE W8A16 fallback paths are not native GB10 evidence"
                    ),
                },
                "int8_moe_triton_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Int8 MoE Triton fallback is not native GB10 evidence"
                    ),
                },
                "wna16_moe_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "WNA16 MoE fallback paths are not native GB10 evidence",
                },
                "compressed_tensors_wna16_dense_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors WNA16 dense loading can select "
                        "generic mixed-precision fallbacks"
                    ),
                },
                "compressed_tensors_wna16_moe_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors WNA16 MoE legacy fused-experts "
                        "fallback is not native GB10 evidence"
                    ),
                },
                "moe_wna16_legacy_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "MoeWNA16 legacy fused-experts fallback is not native "
                        "GB10 evidence"
                    ),
                },
                "rocm_aiter_fp8_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "AITER FP8 MoE is not a native GB10 CUDA path",
                },
                "deep_gemm_fp8_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "DeepGEMM FP8 MoE lacks GB10 evidence",
                },
                "triton_fp8_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Triton FP8 MoE lacks GB10 evidence",
                },
                "vllm_cutlass_fp8_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "vLLM CUTLASS FP8 MoE lacks GB10 evidence",
                },
                "rocm_aiter_mxfp4_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "AITER MXFP4 MoE is not a native GB10 CUDA path",
                },
                "rocm_aiter_unquantized_moe": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "AITER unquantized MoE is not a native GB10 CUDA path"
                    ),
                },
                "unquantized_moe_triton_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Generic Triton unquantized MoE fallback is not "
                        "native GB10 unquantized MoE evidence"
                    ),
                },
                "mxfp8_dense_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "MXFP8 dense fallback paths are not native GB10 evidence",
                },
                "mxfp8_moe_fallback": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "MXFP8 MoE fallback paths are not native GB10 evidence",
                },
                "modelopt_fp8_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "ModelOpt FP8 quantization can select FP8 dense "
                        "kernel selection and FP8 MoE backend selection "
                        "without native GB10 evidence"
                    ),
                },
                "modelopt_mxfp8_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "ModelOpt MXFP8 quantization can select MXFP8 dense "
                        "kernel selection and MXFP8 MoE backend selection "
                        "without native GB10 evidence"
                    ),
                },
                "modelopt_mixed_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "ModelOpt mixed precision quantization can select FP8, "
                        "NVFP4, and W4A16 NVFP4 fallback paths without native "
                        "GB10 mixed precision evidence"
                    ),
                },
                "fbgemm_fp8_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "FBGEMM FP8 quantization can select generic FP8 linear "
                        "kernels without native GB10 evidence"
                    ),
                },
                "experts_int8_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "ExpertsInt8 quantization can select online Int8 MoE "
                        "backend selection without native GB10 evidence"
                    ),
                },
                "fp_quant_fp4_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "FPQuant FP4 quantization can select MXFP4/NVFP4 "
                        "FPQuant linear kernels without native GB10 evidence"
                    ),
                },
                "online_fp8_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Online FP8 quantization can select FP8 scaled-mm "
                        "dense and generic FP8 MoE paths without native "
                        "GB10 evidence"
                    ),
                },
                "online_mxfp8_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Online MXFP8 quantization can select FlashInfer "
                        "CUTLASS MXFP8 dense and generic MXFP8 MoE paths "
                        "without native GB10 evidence"
                    ),
                },
                "online_mxfp4_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Online MXFP4 quantization can accept weight='mxfp4' "
                        "for dense or MoE online quantization without a wired "
                        "online MXFP4 method or native GB10 evidence"
                    ),
                },
                "online_int8_moe_quantization": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Online Int8 MoE quantization can select online Int8 "
                        "MoE backend selection through "
                        "int8_per_channel_weight_only without native GB10 "
                        "evidence"
                    ),
                },
                "compressed_tensors_w8a8_mxfp8_dense_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W8A8 MXFP8 dense loading can select "
                        "MXFP8 dense kernel selection without native GB10 evidence"
                    ),
                },
                "compressed_tensors_w8a8_mxfp8_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W8A8 MXFP8 MoE loading can select "
                        "generic MXFP8 MoE backend selection without native "
                        "GB10 evidence"
                    ),
                },
                "quark_nvfp4_checkpoint_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Quark NVFP4 checkpoint loading is not validated",
                },
                "quark_ocp_mx_checkpoint_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Quark OCP-MX checkpoint loading is not validated",
                },
                "quark_w4a8_mxfp4_fp8_checkpoint_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Quark W4A8 MXFP4+FP8 checkpoint loading is not validated"
                    ),
                },
                "quark_w4a8_fp8_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Quark W4A8 FP8 MoE checkpoint loading can select "
                        "ROCm AITER fused MoE support without native GB10 "
                        "evidence"
                    ),
                },
                "quark_w8a8_fp8_checkpoint_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Quark W8A8 FP8 checkpoint loading can select FP8 "
                        "scaled-mm dense kernel selection without native GB10 "
                        "evidence"
                    ),
                },
                "quark_w8a8_int8_checkpoint_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Quark W8A8 Int8 checkpoint loading can select Int8 "
                        "scaled-mm dense kernel selection without native GB10 "
                        "evidence"
                    ),
                },
                "quark_w8a8_fp8_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Quark W8A8 FP8 MoE checkpoint loading can select "
                        "generic FP8 W8A8 MoE backend selection without native "
                        "GB10 evidence"
                    ),
                },
                "quark_w8a8_int8_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "Quark W8A8 Int8 MoE checkpoint loading can select "
                        "generic Int8 W8A8 MoE backend selection without native "
                        "GB10 evidence"
                    ),
                },
                "compressed_tensors_w4a8_fp8_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W4A8 FP8 loading is exact-SM90 "
                        "CUTLASS evidence, not native GB10 evidence"
                    ),
                },
                "compressed_tensors_w4a8_int_dense_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W4A8 Int dense loading can select "
                        "generic mixed-precision fallbacks"
                    ),
                },
                "compressed_tensors_w4a8_int_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W4A8 Int8 MoE loading can select "
                        "CPU-only W4A8 Int8 MoE backend selection"
                    ),
                },
                "compressed_tensors_w8a16_fp8_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W8A16 FP8 loading selects the FP8 "
                        "W8A16 Marlin fallback"
                    ),
                },
                "compressed_tensors_w8a8_fp8_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W8A8 FP8 loading can select "
                        "scaled-mm kernels without native GB10 dense evidence"
                    ),
                },
                "compressed_tensors_w8a8_fp8_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W8A8 FP8 MoE loading can select "
                        "generic FP8 MoE backends without native GB10 evidence"
                    ),
                },
                "compressed_tensors_w8a8_int_dense_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W8A8 Int dense loading can select "
                        "generic scaled-mm kernels without native GB10 evidence"
                    ),
                },
                "compressed_tensors_w8a8_int_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W8A8 Int8 MoE loading can select "
                        "generic Int8 MoE backends without native GB10 evidence"
                    ),
                },
                "compressed_tensors_w4a4_mxfp4_dense_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W4A4 MXFP4 dense loading is not validated"
                    ),
                },
                "compressed_tensors_w4a4_mxfp4_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W4A4 MXFP4 MoE loading can fall back "
                        "to Marlin without native SM12x evidence"
                    ),
                },
                "compressed_tensors_w4a16_nvfp4_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "CompressedTensors W4A16 NVFP4 selects Marlin",
                },
                "compressed_tensors_w4a16_nvfp4_moe_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors W4A16 NVFP4 MoE loading can reach "
                        "weight-only NVFP4 MoE handling"
                    ),
                },
                "compressed_tensors_qutlass_nvfp4_transform_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": (
                        "CompressedTensors Qutlass NVFP4 transform loading "
                        "has no implemented apply path"
                    ),
                },
            },
            "deferred_paths": {
                "deepseek_v4_deep_gemm_mega_moe": {
                    "status": "deferred",
                    "expected_handling": "block_until_hardware_validated",
                    "reason": "DeepSeek V4 DeepGEMM MegaMoE lacks GB10 evidence",
                },
                "flashinfer_b12x_ep_all2all_eplb": {
                    "status": "deferred",
                    "expected_handling": "block_until_hardware_validated",
                    "reason": "multi-Spark EP/all-to-all/EPLB is not validated",
                },
                "flashinfer_cudnn_nvfp4_dense": {
                    "status": "deferred",
                    "expected_handling": "block_until_hardware_validated",
                    "reason": "FlashInfer cuDNN dense is not validated on SM12x",
                },
                "multi_spark_ep_all2all_eplb": {
                    "status": "deferred",
                    "expected_handling": "block_until_hardware_validated",
                    "reason": "multi-Spark communication hardware is not available",
                },
            },
            "routed_paths": {
                "gb10_attention_trtllm_gen_to_flashinfer_fa2": {
                    "status": "supported_routed",
                    "expected_handling": "route_to_validated_gb10_backend",
                    "target": "flashinfer_attention_fa2",
                    "reason": "TRTLLM Gen attention is unavailable on SM121",
                },
                "gb10_attention_public_flashattention_to_flashinfer_or_flashmla": {
                    "status": "supported_routed",
                    "expected_handling": "route_to_validated_gb10_backend",
                    "target": "flashinfer_attention_fa2 or flashmla_attention",
                    "reason": "public FlashAttention is not selected on SM12x",
                },
                "gb10_moe_trtllm_gen_to_flashinfer_non_ep": {
                    "status": "supported_routed",
                    "expected_handling": "route_to_validated_gb10_backend",
                    "target": (
                        "flashinfer_b12x_non_ep_moe or "
                        "flashinfer_cutlass_non_ep_moe"
                    ),
                    "reason": "TRTLLM Gen MoE is unavailable on SM121",
                },
            },
            "checks": {
                "model_shape": {
                    "status": "observed",
                    "model": "gb10-model",
                    "max_model_len": 4096,
                    "head_size": 128,
                    "num_attention_heads": 32,
                    "num_kv_heads": 8,
                },
                "kv_cache_dtype": {
                    "status": "passed",
                    "expected": "fp8_e4m3",
                    "configured": "fp8_e4m3",
                },
                "cuda_graph": {
                    "status": "passed",
                    "configured_mode": "PIECEWISE",
                    "configured_enabled": True,
                    "num_cudagraph_captured": 2,
                    "num_cudagraph_replayed": 1,
                },
                "attention_backend": {
                    "status": "passed",
                    "expected": "FLASHINFER",
                    "requested_backend": "FLASHINFER",
                    "mla_prefill_backend": None,
                },
                "quantization": {
                    "status": "passed",
                    "expected": "modelopt_fp4",
                    "configured": "modelopt_fp4",
                },
            },
        },
    }
    openai_report = {
        "status": "passed",
        "models": {
            "selected": {
                "id": "qwen3.6",
                "root": "nvidia/Qwen3.6-35B-A3B-NVFP4",
            }
        },
        "response": {
            "status": 200,
            "generated_text": "native output",
            "generated_text_source": "message.content",
            "system_fingerprint": "vllm-gb10",
        },
        "deterministic_generation": {
            "status": "passed",
            "repeat_count": 2,
            "unique_generated_text_count": 1,
            "generated_texts_match": True,
        },
        "gb10_release_evidence": {
            "openai_compatible_server_smoke": {
                "status": "passed",
                "endpoint": "/v1/chat/completions",
                "generated_text_observed": True,
            }
        },
    }
    release_manifest = {
        "schema_version": 1,
        "git": {"commit": "abcdef1234567890abcdef1234567890abcdef12"},
        "release": {"tag": "gb10-vllm-test", "preflight_only": False},
        "image": {
            "name": "ghcr.io/gardner/vllm-gb10",
            "tag": "gb10-vllm-test",
            "push": True,
        },
        "vllm": {"version": "0.22.1rc0+gb10.abcdef123456"},
        "dependencies": {
            "flashinfer": {
                "all_required_components_present": True,
                "missing_components": [],
                "wheels": [
                    {
                        "component": component,
                        "release_tag": FLASHINFER_RELEASE_TAG,
                        "url": (
                            "https://github.com/gardner/flashinfer/releases/"
                            f"download/{FLASHINFER_RELEASE_TAG}/{wheel}"
                        ),
                    }
                    for component, wheel in zip(
                        (
                            "flashinfer_python",
                            "flashinfer_cubin",
                            "flashinfer_jit_cache",
                        ),
                        FLASHINFER_RELEASE_WHEELS,
                    )
                ],
            },
            "vllm_flash_attn": {
                "repository": "https://github.com/gardner/vllm-flash-attention.git",
                "ref": VLLM_FLASH_ATTN_GIT_TAG,
                "ref_is_full_git_sha": True,
            },
            "source_dependencies": {
                "deepgemm": {
                    "name": "DeepGEMM",
                    "repository": "https://github.com/gardner/DeepGEMM.git",
                    "ref": DEEPGEMM_GIT_TAG,
                    "ref_is_full_git_sha": True,
                },
                "flashmla": {
                    "name": "FlashMLA",
                    "repository": "https://github.com/gardner/FlashMLA.git",
                    "ref": FLASHMLA_GIT_TAG,
                    "ref_is_full_git_sha": True,
                },
                "triton_kernels": {
                    "name": "triton_kernels",
                    "repository": "https://github.com/gardner/triton.git",
                    "ref": TRITON_KERNELS_GIT_TAG,
                    "ref_is_full_git_sha": True,
                },
            },
        },
        "gb10_support_matrix": {
            "architecture": "sm_121a",
            "first_release_scope": "single_spark_first_path",
            "entries": {
                "flashinfer_b12x_nvfp4_dense": {"status": "supported_native"},
                "flashinfer_cutlass_nvfp4_dense": {"status": "supported_native"},
                "flashinfer_nvfp4_quantization": {"status": "supported_native"},
                "modelopt_fp4_quantization": {"status": "supported_native"},
                "compressed_tensors_w4a4_nvfp4_dense_loading": {
                    "status": "supported_native"
                },
                "compressed_tensors_w4a4_nvfp4_moe_loading": {
                    "status": "supported_native"
                },
                "compressed_tensors_fp4_kv_cache_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_qutlass_nvfp4_transform_loading": {
                    "status": "not_supported"
                },
                "flashinfer_attention_fa2": {"status": "supported_native"},
                "flashinfer_b12x_non_ep_moe": {"status": "supported_native"},
                "flashinfer_cutlass_non_ep_moe": {"status": "supported_native"},
                "flashmla_attention": {"status": "supported_native"},
                "flashmla_sparse_attention": {"status": "supported_native"},
                "flashinfer_mamba_ssu": {"status": "supported_native"},
                "flashinfer_gdn_prefill": {"status": "supported_native"},
                "gb10_attention_trtllm_gen_to_flashinfer_fa2": {
                    "status": "supported_routed"
                },
                "gb10_attention_public_flashattention_to_flashinfer_or_flashmla": {
                    "status": "supported_routed"
                },
                "gb10_moe_trtllm_gen_to_flashinfer_non_ep": {
                    "status": "supported_routed"
                },
                "public_flashattention_runtime": {"status": "not_supported"},
                "flashinfer_trtllm_nvfp4_dense": {"status": "not_supported"},
                "flashinfer_trtllm_mxfp4_moe": {"status": "not_supported"},
                "flashinfer_cutedsl_nvfp4_moe": {"status": "not_supported"},
                "trtllm_gen_attention": {"status": "not_supported"},
                "triton_attention_fallback": {"status": "not_supported"},
                "flex_attention_fallback": {"status": "not_supported"},
                "turboquant_attention": {"status": "not_supported"},
                "triton_mla_fallback": {"status": "not_supported"},
                "flashinfer_trtllm_mla_attention": {"status": "not_supported"},
                "flashinfer_trtllm_sparse_mla_attention": {
                    "status": "not_supported"
                },
                "public_flashattention_mla_runtime": {"status": "not_supported"},
                "cutlass_mla_sm100_fallback": {"status": "not_supported"},
                "tokenspeed_mla_cutedsl_fallback": {"status": "not_supported"},
                "triton_mamba_ssu_fallback": {"status": "not_supported"},
                "mamba1_triton_runtime": {"status": "not_supported"},
                "mamba2_triton_ssd_runtime": {"status": "not_supported"},
                "short_conv_triton_runtime": {"status": "not_supported"},
                "linear_attention_triton_runtime": {"status": "not_supported"},
                "speculative_decoding_runtime": {"status": "not_supported"},
                "pooling_runtime": {"status": "not_supported"},
                "reasoning_runtime": {"status": "not_supported"},
                "structured_outputs_runtime": {"status": "not_supported"},
                "openai_tool_calling_runtime": {"status": "not_supported"},
                "lora_runtime": {"status": "not_supported"},
                "gdn_prefill_triton_fallback": {"status": "not_supported"},
                "gdn_prefill_cutedsl_backend": {"status": "not_supported"},
                "mm_encoder_fp8_attention": {"status": "not_supported"},
                "mm_encoder_public_flashattention_backend": {
                    "status": "not_supported"
                },
                "mm_encoder_triton_attention_fallback": {
                    "status": "not_supported"
                },
                "mm_encoder_torch_sdpa_attention_fallback": {
                    "status": "not_supported"
                },
                "trtllm_gen_moe": {"status": "not_supported"},
                "rocm_aiter_unquantized_moe": {"status": "not_supported"},
                "unquantized_moe_triton_fallback": {"status": "not_supported"},
                "rocm_aiter_fp8_moe": {"status": "not_supported"},
                "deep_gemm_fp8_moe": {"status": "not_supported"},
                "triton_fp8_moe": {"status": "not_supported"},
                "vllm_cutlass_fp8_moe": {"status": "not_supported"},
                "rocm_aiter_mxfp4_moe": {"status": "not_supported"},
                "gpt_oss_triton_mxfp4_moe": {"status": "not_supported"},
                "marlin_nvfp4_fallback": {"status": "not_supported"},
                "fbgemm_nvfp4_dense": {"status": "not_supported"},
                "modelopt_w4a16_nvfp4_checkpoint_loading": {
                    "status": "not_supported"
                },
                "modelopt_nvfp4_kv_cache_loading": {"status": "not_supported"},
                "nvfp4_kv_cache_runtime": {"status": "not_supported"},
                "unvalidated_kv_cache_runtime": {"status": "not_supported"},
                "kv_events_runtime": {"status": "not_supported"},
                "kv_offload_runtime": {"status": "not_supported"},
                "kv_transfer_runtime": {"status": "not_supported"},
                "ubatching_runtime": {"status": "not_supported"},
                "distributed_parallel_runtime": {"status": "not_supported"},
                "kv_sharing_fast_prefill_runtime": {"status": "not_supported"},
                "ec_transfer_runtime": {"status": "not_supported"},
                "weight_transfer_runtime": {"status": "not_supported"},
                "return_routed_experts_runtime": {"status": "not_supported"},
                "logprobs_logits_runtime": {"status": "not_supported"},
                "custom_logits_processors_runtime": {"status": "not_supported"},
                "io_processor_plugin_runtime": {"status": "not_supported"},
                "hf_overrides_runtime": {"status": "not_supported"},
                "transformers_model_impl_runtime": {"status": "not_supported"},
                "trust_remote_code_runtime": {"status": "not_supported"},
                "custom_scheduler_runtime": {"status": "not_supported"},
                "custom_worker_runtime": {"status": "not_supported"},
                "prompt_embeds_runtime": {"status": "not_supported"},
                "stock_torch_compile_runtime": {"status": "not_supported"},
                "mamba_align_cache_runtime": {"status": "not_supported"},
                "marlin_mxfp4_fallback": {"status": "not_supported"},
                "mxfp4_moe_fallback": {"status": "not_supported"},
                "public_mxfp4_quantization": {"status": "not_supported"},
                "public_fp8_quantization": {"status": "not_supported"},
                "deepseek_v4_fp8_quantization": {"status": "not_supported"},
                "torchao_fp8_activation_quantization": {"status": "not_supported"},
                "torchao_weight_quantization": {"status": "not_supported"},
                "bitsandbytes_quantization": {"status": "not_supported"},
                "awq_quantization": {"status": "not_supported"},
                "gptq_quantization": {"status": "not_supported"},
                "inc_quantization": {"status": "not_supported"},
                "gguf_quantization": {"status": "not_supported"},
                "humming_quantization": {"status": "not_supported"},
                "humming_mxfp4_moe_backend": {"status": "not_supported"},
                "fp8_w8a16_marlin_fallback": {"status": "not_supported"},
                "fp8_w8a16_moe_fallback": {"status": "not_supported"},
                "int8_moe_triton_fallback": {"status": "not_supported"},
                "wna16_moe_fallback": {"status": "not_supported"},
                "compressed_tensors_wna16_dense_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_wna16_moe_fallback": {
                    "status": "not_supported"
                },
                "moe_wna16_legacy_fallback": {"status": "not_supported"},
                "mxfp8_dense_fallback": {"status": "not_supported"},
                "mxfp8_moe_fallback": {"status": "not_supported"},
                "modelopt_fp8_quantization": {"status": "not_supported"},
                "modelopt_mxfp8_quantization": {"status": "not_supported"},
                "modelopt_mixed_quantization": {"status": "not_supported"},
                "fbgemm_fp8_quantization": {"status": "not_supported"},
                "experts_int8_quantization": {"status": "not_supported"},
                "fp_quant_fp4_quantization": {"status": "not_supported"},
                "online_fp8_quantization": {"status": "not_supported"},
                "online_mxfp8_quantization": {"status": "not_supported"},
                "online_mxfp4_quantization": {"status": "not_supported"},
                "online_int8_moe_quantization": {"status": "not_supported"},
                "compressed_tensors_w8a8_mxfp8_dense_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w8a8_mxfp8_moe_loading": {
                    "status": "not_supported"
                },
                "quark_nvfp4_checkpoint_loading": {"status": "not_supported"},
                "quark_ocp_mx_checkpoint_loading": {"status": "not_supported"},
                "quark_w4a8_mxfp4_fp8_checkpoint_loading": {
                    "status": "not_supported"
                },
                "quark_w4a8_fp8_moe_loading": {
                    "status": "not_supported"
                },
                "quark_w8a8_fp8_checkpoint_loading": {
                    "status": "not_supported"
                },
                "quark_w8a8_int8_checkpoint_loading": {
                    "status": "not_supported"
                },
                "quark_w8a8_fp8_moe_loading": {
                    "status": "not_supported"
                },
                "quark_w8a8_int8_moe_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w4a8_fp8_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w4a8_int_dense_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w4a8_int_moe_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w8a16_fp8_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w8a8_fp8_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w8a8_fp8_moe_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w8a8_int_dense_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w8a8_int_moe_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w4a4_mxfp4_dense_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w4a4_mxfp4_moe_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w4a16_nvfp4_loading": {
                    "status": "not_supported"
                },
                "compressed_tensors_w4a16_nvfp4_moe_loading": {
                    "status": "not_supported"
                },
                "deepseek_v4_deep_gemm_mega_moe": {"status": "deferred"},
                "flashinfer_b12x_ep_all2all_eplb": {"status": "deferred"},
                "flashinfer_cudnn_nvfp4_dense": {"status": "deferred"},
                "multi_spark_ep_all2all_eplb": {"status": "deferred"},
            },
        },
            "build": {
                "cache_refs": {
                    "preflight": GB10_RELEASE_CACHE_REF_ENV["GB10_PREFLIGHT_CACHE_REF"],
                    "wheel": GB10_RELEASE_CACHE_REF_ENV["GB10_WHEEL_CACHE_REF"],
                    "runtime": GB10_RELEASE_CACHE_REF_ENV["GB10_RUNTIME_CACHE_REF"],
                },
                "local_gb10_dependency_checkouts": False,
                "native_cuda_archs_only": True,
            },
        }
    runtime_image_digest = "sha256:" + "a" * 64
    runtime_image_metadata = {"containerimage.digest": runtime_image_digest}

    summary = verifier._build_summary(
        nvfp4_report=nvfp4_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        runtime_image_metadata=runtime_image_metadata,
        runtime_image_metadata_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        image_digest=f"ghcr.io/gardner/vllm-gb10@{runtime_image_digest}",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_runtime_image_metadata=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert summary["status"] == "passed"
    assert summary["release_gate_passed"] is True
    assert summary["failure_count"] == 0
    checks_by_name = {check["name"]: check for check in summary["checks"]}
    check_statuses = {name: check["status"] for name, check in checks_by_name.items()}
    assert check_statuses["native_nvfp4_gemm_observed"] == "passed"
    assert check_statuses["native_nvfp4_moe_non_ep_observed"] == "passed"
    assert check_statuses["nvfp4_fallback_free"] == "passed"
    assert check_statuses["gb10_device_sm121"] == "passed"
    assert check_statuses["flashinfer_gb10_runtime_version"] == "passed"
    assert check_statuses["flashinfer_gb10_distribution_versions"] == "passed"
    assert check_statuses["cuda_graph_capture_replay"] == "passed"
    assert check_statuses["model_shape_reported"] == "passed"
    assert checks_by_name["model_shape_reported"]["details"] == {
        "status": "observed",
        "model": "gb10-model",
        "max_model_len": 4096,
        "head_size": 128,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
    }
    assert check_statuses["attention_backend_flashinfer"] == "passed"
    assert check_statuses["attention_backend_allowed_by_support_matrix"] == "passed"
    assert check_statuses["quantization_modelopt_fp4"] == "passed"
    assert check_statuses["quantization_allowed_by_support_matrix"] == "passed"
    assert check_statuses["supported_routed_paths_reported"] == "passed"
    assert check_statuses["unsupported_paths_reported"] == "passed"
    assert check_statuses["deferred_paths_reported"] == "passed"
    assert checks_by_name["quantization_allowed_by_support_matrix"]["details"][
        "support_matrix_entry"
    ] == "modelopt_fp4_quantization"
    assert checks_by_name["unsupported_paths_reported"]["details"][
        "reported_not_supported_entries"
    ] == [
        "awq_quantization",
        "bitsandbytes_quantization",
        "compressed_tensors_fp4_kv_cache_loading",
        "compressed_tensors_qutlass_nvfp4_transform_loading",
        "compressed_tensors_w4a16_nvfp4_loading",
        "compressed_tensors_w4a16_nvfp4_moe_loading",
        "compressed_tensors_w4a4_mxfp4_dense_loading",
        "compressed_tensors_w4a4_mxfp4_moe_loading",
        "compressed_tensors_w4a8_fp8_loading",
        "compressed_tensors_w4a8_int_dense_loading",
        "compressed_tensors_w4a8_int_moe_loading",
        "compressed_tensors_w8a16_fp8_loading",
        "compressed_tensors_w8a8_fp8_loading",
        "compressed_tensors_w8a8_fp8_moe_loading",
        "compressed_tensors_w8a8_int_dense_loading",
        "compressed_tensors_w8a8_int_moe_loading",
        "compressed_tensors_w8a8_mxfp8_dense_loading",
        "compressed_tensors_w8a8_mxfp8_moe_loading",
        "compressed_tensors_wna16_dense_loading",
        "compressed_tensors_wna16_moe_fallback",
        "custom_logits_processors_runtime",
        "custom_scheduler_runtime",
        "custom_worker_runtime",
        "cutlass_mla_sm100_fallback",
        "deep_gemm_fp8_moe",
        "deepseek_v4_fp8_quantization",
        "distributed_parallel_runtime",
        "ec_transfer_runtime",
        "experts_int8_quantization",
        "fbgemm_fp8_quantization",
        "fbgemm_nvfp4_dense",
        "flashinfer_cutedsl_nvfp4_moe",
        "flashinfer_trtllm_mla_attention",
        "flashinfer_trtllm_mxfp4_moe",
        "flashinfer_trtllm_nvfp4_dense",
        "flashinfer_trtllm_sparse_mla_attention",
        "flex_attention_fallback",
        "fp8_w8a16_marlin_fallback",
        "fp8_w8a16_moe_fallback",
        "fp_quant_fp4_quantization",
        "gdn_prefill_cutedsl_backend",
        "gdn_prefill_triton_fallback",
        "gguf_quantization",
        "gpt_oss_triton_mxfp4_moe",
        "gptq_quantization",
        "hf_overrides_runtime",
        "humming_mxfp4_moe_backend",
        "humming_quantization",
        "inc_quantization",
        "int8_moe_triton_fallback",
        "io_processor_plugin_runtime",
        "kv_events_runtime",
        "kv_offload_runtime",
        "kv_sharing_fast_prefill_runtime",
        "kv_transfer_runtime",
        "linear_attention_triton_runtime",
        "logprobs_logits_runtime",
        "lora_runtime",
        "mamba1_triton_runtime",
        "mamba2_triton_ssd_runtime",
        "mamba_align_cache_runtime",
        "marlin_mxfp4_fallback",
        "marlin_nvfp4_fallback",
        "mm_encoder_fp8_attention",
        "mm_encoder_public_flashattention_backend",
        "mm_encoder_torch_sdpa_attention_fallback",
        "mm_encoder_triton_attention_fallback",
        "modelopt_fp8_quantization",
        "modelopt_mixed_quantization",
        "modelopt_mxfp8_quantization",
        "modelopt_nvfp4_kv_cache_loading",
        "modelopt_w4a16_nvfp4_checkpoint_loading",
        "moe_wna16_legacy_fallback",
        "mxfp4_moe_fallback",
        "mxfp8_dense_fallback",
        "mxfp8_moe_fallback",
        "nvfp4_kv_cache_runtime",
        "online_fp8_quantization",
        "online_int8_moe_quantization",
        "online_mxfp4_quantization",
        "online_mxfp8_quantization",
        "openai_tool_calling_runtime",
        "pooling_runtime",
        "prompt_embeds_runtime",
        "public_flashattention_mla_runtime",
        "public_flashattention_runtime",
        "public_fp8_quantization",
        "public_mxfp4_quantization",
        "quark_nvfp4_checkpoint_loading",
        "quark_ocp_mx_checkpoint_loading",
        "quark_w4a8_fp8_moe_loading",
        "quark_w4a8_mxfp4_fp8_checkpoint_loading",
        "quark_w8a8_fp8_checkpoint_loading",
        "quark_w8a8_fp8_moe_loading",
        "quark_w8a8_int8_checkpoint_loading",
        "quark_w8a8_int8_moe_loading",
        "reasoning_runtime",
        "return_routed_experts_runtime",
        "rocm_aiter_fp8_moe",
        "rocm_aiter_mxfp4_moe",
        "rocm_aiter_unquantized_moe",
        "short_conv_triton_runtime",
        "speculative_decoding_runtime",
        "stock_torch_compile_runtime",
        "structured_outputs_runtime",
        "tokenspeed_mla_cutedsl_fallback",
        "torchao_fp8_activation_quantization",
        "torchao_weight_quantization",
        "transformers_model_impl_runtime",
        "triton_attention_fallback",
        "triton_fp8_moe",
        "triton_mamba_ssu_fallback",
        "triton_mla_fallback",
        "trtllm_gen_attention",
        "trtllm_gen_moe",
        "trust_remote_code_runtime",
        "turboquant_attention",
        "ubatching_runtime",
        "unquantized_moe_triton_fallback",
        "unvalidated_kv_cache_runtime",
        "vllm_cutlass_fp8_moe",
        "weight_transfer_runtime",
        "wna16_moe_fallback",
    ]
    assert checks_by_name["supported_routed_paths_reported"]["details"][
        "reported_supported_routed_entries"
    ] == [
        "gb10_attention_public_flashattention_to_flashinfer_or_flashmla",
        "gb10_attention_trtllm_gen_to_flashinfer_fa2",
        "gb10_moe_trtllm_gen_to_flashinfer_non_ep",
    ]
    assert checks_by_name["deferred_paths_reported"]["details"][
        "reported_deferred_entries"
    ] == [
        "deepseek_v4_deep_gemm_mega_moe",
        "flashinfer_b12x_ep_all2all_eplb",
        "flashinfer_cudnn_nvfp4_dense",
        "multi_spark_ep_all2all_eplb",
    ]
    assert (
        check_statuses["nvfp4_backend_selections_allowed_by_support_matrix"]
        == "passed"
    )
    assert check_statuses["openai_deterministic_generation"] == "passed"
    assert check_statuses["release_manifest_flashinfer_components"] == "passed"
    assert check_statuses["release_manifest_durable_inputs"] == "passed"
    assert (
        check_statuses["release_manifest_source_dependencies_present"] == "passed"
    )
    assert check_statuses["release_manifest_gb10_support_matrix"] == "passed"
    assert check_statuses["release_manifest_source_refs_pinned"] == "passed"
    assert check_statuses["release_manifest_no_local_deps"] == "passed"
    assert (
        check_statuses["release_manifest_image_pushed_for_tagged_release"] == "passed"
    )
    assert check_statuses["release_manifest_image_ref_matches_smoke"] == "passed"
    assert check_statuses["release_manifest_tag_matches_expected"] == "passed"
    assert check_statuses["runtime_image_metadata_has_digest"] == "passed"
    assert check_statuses["runtime_image_digest_matches_smoke"] == "passed"

    unsupported_selection_report = copy.deepcopy(nvfp4_report)
    unsupported_selection_report["backend_selections"] = [
        {
            "path": "linear",
            "backend": "MARLIN",
            "is_fallback": False,
        }
    ]
    unsupported_selection_summary = verifier._build_summary(
        nvfp4_report=unsupported_selection_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert unsupported_selection_summary["status"] == "failed"
    unsupported_selection_failures = {
        failure["name"] for failure in unsupported_selection_summary["failures"]
    }
    assert "nvfp4_backend_selections_allowed_by_support_matrix" in (
        unsupported_selection_failures
    )

    missing_routed_target_report = copy.deepcopy(nvfp4_report)
    del missing_routed_target_report["gb10_release_summary"]["routed_paths"][
        "gb10_attention_trtllm_gen_to_flashinfer_fa2"
    ]["target"]
    missing_routed_target_summary = verifier._build_summary(
        nvfp4_report=missing_routed_target_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )
    assert missing_routed_target_summary["status"] == "failed"
    assert any(
        failure["name"] == "supported_routed_paths_reported"
        for failure in missing_routed_target_summary["failures"]
    )

    missing_unsupported_paths_report = copy.deepcopy(nvfp4_report)
    del missing_unsupported_paths_report["gb10_release_summary"]["unsupported_paths"][
        "trtllm_gen_attention"
    ]
    missing_unsupported_paths_summary = verifier._build_summary(
        nvfp4_report=missing_unsupported_paths_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )
    assert missing_unsupported_paths_summary["status"] == "failed"
    assert any(
        failure["name"] == "unsupported_paths_reported"
        for failure in missing_unsupported_paths_summary["failures"]
    )

    missing_deferred_paths_report = copy.deepcopy(nvfp4_report)
    del missing_deferred_paths_report["gb10_release_summary"]["deferred_paths"][
        "multi_spark_ep_all2all_eplb"
    ]
    missing_deferred_paths_summary = verifier._build_summary(
        nvfp4_report=missing_deferred_paths_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )
    assert missing_deferred_paths_summary["status"] == "failed"
    assert any(
        failure["name"] == "deferred_paths_reported"
        for failure in missing_deferred_paths_summary["failures"]
    )

    missing_model_shape_report = copy.deepcopy(nvfp4_report)
    del missing_model_shape_report["gb10_release_summary"]["checks"]["model_shape"]
    missing_model_shape_summary = verifier._build_summary(
        nvfp4_report=missing_model_shape_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )
    assert missing_model_shape_summary["status"] == "failed"
    assert any(
        failure["name"] == "model_shape_reported"
        for failure in missing_model_shape_summary["failures"]
    )

    missing_cuda_graph_replay_report = copy.deepcopy(nvfp4_report)
    missing_cuda_graph_replay_report["gb10_release_summary"]["checks"]["cuda_graph"][
        "num_cudagraph_replayed"
    ] = 0
    missing_cuda_graph_replay_report["gb10_release_summary"]["checks"]["cuda_graph"][
        "status"
    ] = "not_observed"
    missing_cuda_graph_replay_summary = verifier._build_summary(
        nvfp4_report=missing_cuda_graph_replay_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )
    assert missing_cuda_graph_replay_summary["status"] == "failed"
    assert any(
        failure["name"] == "cuda_graph_capture_replay"
        for failure in missing_cuda_graph_replay_summary["failures"]
    )

    nvfp4_report["fallback_events"] = [
        {
            "path": "linear",
            "backend": "MarlinNvFp4LinearKernel",
            "message": "fallback selected",
        }
    ]
    failed_summary = verifier._build_summary(
        nvfp4_report=nvfp4_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert failed_summary["status"] == "failed"
    assert failed_summary["release_gate_passed"] is False
    assert any(
        failure["name"] == "nvfp4_fallback_free"
        for failure in failed_summary["failures"]
    )

    attention_failed_report = {
        **nvfp4_report,
        "fallback_events": [],
        "gb10_release_summary": {
            **nvfp4_report["gb10_release_summary"],
            "checks": {
                **nvfp4_report["gb10_release_summary"]["checks"],
                "attention_backend": {
                    "status": "mismatched",
                    "expected": "FLASHINFER",
                    "requested_backend": "FLASH_ATTN",
                },
            },
        },
    }
    attention_failed_summary = verifier._build_summary(
        nvfp4_report=attention_failed_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert attention_failed_summary["status"] == "failed"
    assert any(
        failure["name"] == "attention_backend_flashinfer"
        for failure in attention_failed_summary["failures"]
    )
    assert any(
        failure["name"] == "attention_backend_allowed_by_support_matrix"
        for failure in attention_failed_summary["failures"]
    )

    quantization_failed_report = {
        **nvfp4_report,
        "fallback_events": [],
        "gb10_release_summary": {
            **nvfp4_report["gb10_release_summary"],
            "checks": {
                **nvfp4_report["gb10_release_summary"]["checks"],
                "quantization": {
                    "status": "mismatched",
                    "expected": "modelopt_fp4",
                    "configured": "none",
                },
            },
        },
    }
    quantization_failed_summary = verifier._build_summary(
        nvfp4_report=quantization_failed_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert quantization_failed_summary["status"] == "failed"
    assert any(
        failure["name"] == "quantization_modelopt_fp4"
        for failure in quantization_failed_summary["failures"]
    )
    assert any(
        failure["name"] == "quantization_allowed_by_support_matrix"
        for failure in quantization_failed_summary["failures"]
    )

    wrong_device_report = {
        **nvfp4_report,
        "fallback_events": [],
        "runtime": {
            "cuda_available": True,
            "device_name": "NVIDIA H100",
            "device_capability": {
                "major": 9,
                "minor": 0,
                "arch": "sm_90",
            },
        },
    }
    wrong_device_summary = verifier._build_summary(
        nvfp4_report=wrong_device_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert wrong_device_summary["status"] == "failed"
    assert any(
        failure["name"] == "gb10_device_sm121"
        for failure in wrong_device_summary["failures"]
    )

    generic_flashinfer_report = {
        **nvfp4_report,
        "fallback_events": [],
        "runtime": {
            **nvfp4_report["runtime"],
            "flashinfer_version": "0.6.12+cu130",
            "flashinfer_distributions": {
                "flashinfer-python": "0.6.12+cu130",
                "flashinfer-cubin": "0.6.12+cu130gb10",
                "flashinfer-jit-cache": "0.6.12+cu130gb10",
            },
        },
    }
    generic_flashinfer_summary = verifier._build_summary(
        nvfp4_report=generic_flashinfer_report,
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert generic_flashinfer_summary["status"] == "failed"
    assert any(
        failure["name"] == "flashinfer_gb10_runtime_version"
        for failure in generic_flashinfer_summary["failures"]
    )
    assert any(
        failure["name"] == "flashinfer_gb10_distribution_versions"
        for failure in generic_flashinfer_summary["failures"]
    )

    release_manifest["dependencies"]["flashinfer"][
        "all_required_components_present"
    ] = False
    release_manifest["dependencies"]["flashinfer"]["missing_components"] = [
        "flashinfer_jit_cache"
    ]
    manifest_failed_summary = verifier._build_summary(
        nvfp4_report={**nvfp4_report, "fallback_events": []},
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert manifest_failed_summary["status"] == "failed"
    assert any(
        failure["name"] == "release_manifest_flashinfer_components"
        for failure in manifest_failed_summary["failures"]
    )

    release_manifest["dependencies"]["flashinfer"][
        "all_required_components_present"
    ] = True
    release_manifest["dependencies"]["flashinfer"]["missing_components"] = []
    original_flashinfer_python_url = release_manifest["dependencies"]["flashinfer"][
        "wheels"
    ][0]["url"]
    release_manifest["dependencies"]["flashinfer"]["wheels"][0]["url"] = (
        "https://github.com/gardner/flashinfer/raw/main/"
        "flashinfer_python-0.6.12+cu130gb10-py3-none-any.whl"
    )
    durable_manifest_failed_summary = verifier._build_summary(
        nvfp4_report={**nvfp4_report, "fallback_events": []},
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert durable_manifest_failed_summary["status"] == "failed"
    assert any(
        failure["name"] == "release_manifest_durable_inputs"
        for failure in durable_manifest_failed_summary["failures"]
    )

    release_manifest["dependencies"]["flashinfer"]["wheels"][0][
        "url"
    ] = original_flashinfer_python_url
    missing_source_dependency_manifest = copy.deepcopy(release_manifest)
    del missing_source_dependency_manifest["dependencies"]["source_dependencies"][
        "flashmla"
    ]
    missing_source_dependency_summary = verifier._build_summary(
        nvfp4_report={**nvfp4_report, "fallback_events": []},
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=missing_source_dependency_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert missing_source_dependency_summary["status"] == "failed"
    assert any(
        failure["name"] == "release_manifest_source_dependencies_present"
        for failure in missing_source_dependency_summary["failures"]
    )

    missing_support_matrix_manifest = copy.deepcopy(release_manifest)
    del missing_support_matrix_manifest["gb10_support_matrix"]["entries"][
        "public_flashattention_runtime"
    ]
    missing_support_matrix_summary = verifier._build_summary(
        nvfp4_report={**nvfp4_report, "fallback_events": []},
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=missing_support_matrix_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert missing_support_matrix_summary["status"] == "failed"
    assert any(
        failure["name"] == "release_manifest_gb10_support_matrix"
        for failure in missing_support_matrix_summary["failures"]
    )

    image_mismatch_summary = verifier._build_summary(
        nvfp4_report={**nvfp4_report, "fallback_events": []},
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:other",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert image_mismatch_summary["status"] == "failed"
    assert any(
        failure["name"] == "release_manifest_image_ref_matches_smoke"
        for failure in image_mismatch_summary["failures"]
    )

    tag_mismatch_summary = verifier._build_summary(
        nvfp4_report={**nvfp4_report, "fallback_events": []},
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        release_tag="gb10-other-release",
        require_release_manifest=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert tag_mismatch_summary["status"] == "failed"
    assert any(
        failure["name"] == "release_manifest_tag_matches_expected"
        for failure in tag_mismatch_summary["failures"]
    )

    digest_mismatch_summary = verifier._build_summary(
        nvfp4_report={**nvfp4_report, "fallback_events": []},
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        runtime_image_metadata=runtime_image_metadata,
        runtime_image_metadata_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        image_digest="ghcr.io/gardner/vllm-gb10@sha256:" + "b" * 64,
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_runtime_image_metadata=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert digest_mismatch_summary["status"] == "failed"
    assert any(
        failure["name"] == "runtime_image_digest_matches_smoke"
        for failure in digest_mismatch_summary["failures"]
    )

    malformed_digest_summary = verifier._build_summary(
        nvfp4_report={**nvfp4_report, "fallback_events": []},
        nvfp4_error=None,
        openai_report=openai_report,
        openai_error=None,
        release_manifest=release_manifest,
        release_manifest_error=None,
        runtime_image_metadata={"containerimage.digest": "sha256:not-a-digest"},
        runtime_image_metadata_error=None,
        image_ref="ghcr.io/gardner/vllm-gb10:gb10-vllm-test",
        image_digest="ghcr.io/gardner/vllm-gb10@sha256:not-a-digest",
        release_tag="gb10-vllm-test",
        require_release_manifest=True,
        require_runtime_image_metadata=True,
        require_moe=True,
        require_openai_deterministic=True,
        allow_partial=False,
    )

    assert malformed_digest_summary["status"] == "failed"
    malformed_digest_failures = {
        failure["name"] for failure in malformed_digest_summary["failures"]
    }
    assert "runtime_image_metadata_has_digest" in malformed_digest_failures
    assert "runtime_image_digest_matches_smoke" in malformed_digest_failures
