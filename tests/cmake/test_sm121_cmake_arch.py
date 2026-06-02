import re
import subprocess
from pathlib import Path

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

    assert gb10_workflow.count("--file docker/Dockerfile") == 2


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
