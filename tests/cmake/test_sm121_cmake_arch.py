import re
import subprocess
from pathlib import Path

import pytest

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

    assert "VLLM_NATIVE_CUDA_ARCHS_ONLY" in cmake_lists
    assert "Skipping cross-major PTX fallback CUDA archs" in cmake_lists


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
    versions_json = (REPO_ROOT / "docker" / "versions.json").read_text()

    for text in (dockerfile, docker_bake, gb10_workflow, versions_json):
        assert "gb10_require_flashinfer_wheels" in text

    for text in (dockerfile, gb10_workflow):
        assert "flashinfer_python" in text
        assert "flashinfer_cubin" in text
        assert "flashinfer_jit_cache" in text

    assert "GB10 prebuilt FlashInfer wheel URLs are required" in gb10_workflow
    assert "GB10 FlashInfer wheels are required" in dockerfile


def test_gb10_release_workflow_defaults_to_published_flashinfer_wheels():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert "GB10_DEFAULT_PREBUILT_WHEEL_URLS" in gb10_workflow
    assert FLASHINFER_RELEASE_TAG in gb10_workflow
    for wheel in FLASHINFER_RELEASE_WHEELS:
        assert wheel in gb10_workflow
        assert (
            f"https://github.com/gardner/flashinfer/releases/download/"
            f"{FLASHINFER_RELEASE_TAG}/{wheel}"
        ) in gb10_workflow
    assert 'prebuilt_wheel_urls="$GB10_DEFAULT_PREBUILT_WHEEL_URLS"' in gb10_workflow


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


def test_gb10_release_workflow_supports_manual_preflight_only():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert "preflight-only:" in gb10_workflow
    assert "GB10_PREFLIGHT_ONLY=${preflight_only}" in gb10_workflow
    assert "preflight_only=\"${{ inputs['preflight-only'] }}\"" in gb10_workflow

    for step_name in (
        "Build wheel stage",
        "Extract vLLM wheel",
        "Verify wheel contains only SM121A CUDA images",
        "Upload wheel artifact",
        "Publish wheel to GitHub Release",
        "Build runtime image",
    ):
        step_block = gb10_workflow.split(f"- name: {step_name}", 1)[1].split(
            "\n      - name:",
            1,
        )[0]
        assert "env.GB10_PREFLIGHT_ONLY != 'true'" in step_block


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


def test_gb10_release_workflow_uses_modest_remote_parallelism():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert 'GB10_MAX_JOBS: "24"' in gb10_workflow
    assert 'GB10_NVCC_THREADS: "8"' in gb10_workflow
    assert "24 / 8 = 3 jobs" in gb10_workflow
    assert gb10_workflow.count('--build-arg max_jobs="$GB10_MAX_JOBS"') == 2
    assert gb10_workflow.count('--build-arg nvcc_threads="$GB10_NVCC_THREADS"') == 2


def test_gb10_local_cached_build_script_defaults_to_serial_builds():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert "preflight|wheel|runtime" in script
    assert 'GB10_MAX_JOBS="${GB10_MAX_JOBS:-1}"' in script
    assert 'GB10_NVCC_THREADS="${GB10_NVCC_THREADS:-1}"' in script
    assert ".buildx-cache/gb10" in script
    assert 'GB10_USE_REGISTRY_CACHE="${GB10_USE_REGISTRY_CACHE:-0}"' in script
    assert 'GB10_BUILDX_BUILDER="${GB10_BUILDX_BUILDER:-gb10-builder}"' in script
    assert "--driver docker-container" in script
    assert '--cache-to "type=local,dest=$cache_next,mode=max"' in script
    assert "GB10_USE_REGISTRY_CACHE=1" in script
    assert "--builder \"$GB10_BUILDX_BUILDER\"" in script
    assert "--target \"$docker_target\"" in script


def test_gb10_release_workflow_overrides_pep440_wheel_version():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()

    assert 'vllm_version_base="0.22.1rc0"' in gb10_workflow
    assert 'vllm_version="${vllm_version_base}+gb10.${GITHUB_SHA::12}"' in gb10_workflow
    assert "GB10_VLLM_VERSION=${vllm_version}" in gb10_workflow
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
    modelopt_quant = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "modelopt.py"
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

    assert "W4A16_NVFP4 linear selected MarlinNvFp4LinearKernel" in modelopt_quant
    assert "record_nvfp4_backend_selection" in modelopt_quant
    assert "record_nvfp4_fallback" in modelopt_quant
    assert '"linear_w4a16"' in modelopt_quant
    assert "weight-only fallback path" in modelopt_quant
    assert "not the native GB10 W4A4 FP4 Tensor " in modelopt_quant
    assert "Core path; verify this fallback is intentional " in modelopt_quant
    assert "before publishing " in modelopt_quant
    assert "GB10 artifacts" in modelopt_quant


def test_gb10_nvfp4_moe_fallbacks_are_reported():
    nvfp4_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "fused_moe" / "oracle" / "nvfp4.py"
    ).read_text()

    assert "_NVFP4_MOE_FALLBACK_BACKENDS" in nvfp4_oracle
    assert "NvFp4MoeBackend.MARLIN" in nvfp4_oracle
    assert "NvFp4MoeBackend.EMULATION" in nvfp4_oracle
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
    assert '"backend_summary": _build_backend_summary(' in script
    assert '"native_nvfp4_gemm"' in script
    assert '"native_nvfp4_moe_non_ep"' in script
    assert '"native_nvfp4_moe_ep"' in script
    assert '"cuda_graph"' in script
    assert '"model_shape"' in script
    assert '"not_validated_by_smoke"' in script
    assert "Expert-parallel/all2all/EPLB NVFP4 MoE is blocked" in script
    assert '"backend_selections": _events_to_dicts(selections)' in script
    assert '"fallback_events": _events_to_dicts(fallbacks)' in script
    assert '"device_capability"' in script
    assert '"flashinfer_version"' in script
    assert 'status="passed"' in script
    assert 'status="failed"' in script
    assert "except Exception as exc:" in script
    assert "linear=FlashInferB12x" in script
    assert "moe=FLASHINFER_B12X" in script
    assert 'choices=("linear", "linear_w4a16", "moe")' in script


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
