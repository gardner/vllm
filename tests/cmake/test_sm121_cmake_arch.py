import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_gb10_build_prefers_sibling_dependency_forks():
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
    gb10_workflow = (REPO_ROOT / ".github" / "workflows" / "gb10-release.yml").read_text()

    assert "VLLM_BUILD_FLASH_ATTN" in cmake_lists
    assert "include(cmake/external_projects/vllm_flash_attn.cmake)" in cmake_lists
    assert "VERSION_GREATER_EQUAL 13.0)\n        set(VLLM_BUILD_FLASH_ATTN OFF)" not in cmake_lists
    assert "VLLM_FLASH_ATTN_GIT_REPOSITORY" in vllm_flash_attn_cmake
    assert "https://github.com/gardner/vllm-flash-attention.git" in vllm_flash_attn_cmake
    assert "VLLM_FLASH_ATTN_GIT_TAG" in vllm_flash_attn_cmake
    assert "6407c49b28a365f5a5f7722116c8e44aaea45692" in vllm_flash_attn_cmake
    assert "def _build_vllm_flash_attn" in setup_py
    assert 'torch.version.cuda.split(".")[0] in ("12", "13")' in setup_py
    assert "if _build_vllm_flash_attn():" in setup_py
    assert "VLLM_BUILD_FLASH_ATTN=1" in dockerfile
    assert "VLLM_BUILD_FLASH_ATTN_FA3=0" in dockerfile
    assert "gb10_prebuilt_wheel_urls" in dockerfile
    assert "gb10_prebuilt_wheel_urls" in gb10_workflow
    assert "gh release upload" in gb10_workflow
    assert "ghcr.io/gardner/vllm-gb10" in gb10_workflow
