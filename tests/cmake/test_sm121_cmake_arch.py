import copy
import importlib.util
import json
import re
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

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
GB10_RELEASE_CACHE_REF_ENV = {
    "GB10_PREFLIGHT_CACHE_REF": "ghcr.io/gardner/vllm-gb10-buildcache:preflight",
    "GB10_WHEEL_CACHE_REF": "ghcr.io/gardner/vllm-gb10-buildcache:wheel",
    "GB10_RUNTIME_CACHE_REF": "ghcr.io/gardner/vllm-gb10-buildcache:runtime",
}
GB10_REQUIRED_SUPPORT_MATRIX = {
    "flashinfer_nvfp4_dense": "supported_native",
    "flashinfer_nvfp4_quantization": "supported_native",
    "flashinfer_attention_fa2": "supported_native",
    "flashinfer_b12x_non_ep_moe": "supported_native",
    "flashmla_attention": "supported_native",
    "public_flashattention_runtime": "not_supported",
    "trtllm_gen_attention": "not_supported",
    "trtllm_gen_moe": "not_supported",
    "marlin_nvfp4_fallback": "not_supported",
    "flashinfer_b12x_ep_all2all_eplb": "deferred",
    "multi_spark_ep_all2all_eplb": "deferred",
}


def _load_gb10_smoke_module():
    script_path = REPO_ROOT / "scripts" / "gb10-smoke-nvfp4.py"
    spec = importlib.util.spec_from_file_location("gb10_smoke_nvfp4", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gb10_openai_smoke_module():
    script_path = REPO_ROOT / "scripts" / "gb10-smoke-openai-server.py"
    spec = importlib.util.spec_from_file_location(
        "gb10_smoke_openai_server",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gb10_release_evidence_module():
    script_path = REPO_ROOT / "scripts" / "gb10-verify-release-evidence.py"
    spec = importlib.util.spec_from_file_location(
        "gb10_verify_release_evidence",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gb10_release_bundle_module():
    script_path = REPO_ROOT / "scripts" / "gb10-bundle-release-evidence.py"
    spec = importlib.util.spec_from_file_location(
        "gb10_bundle_release_evidence",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gb10_release_manifest_module():
    script_path = REPO_ROOT / "scripts" / "gb10-write-release-manifest.py"
    spec = importlib.util.spec_from_file_location(
        "gb10_write_release_manifest",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_gb10_release_workflow_requires_pushed_image_for_tagged_release():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    resolve_step = gb10_workflow.split(
        "- name: Resolve release settings",
        1,
    )[1].split("- name: Write GB10 release manifest", 1)[0]

    assert '[ "$preflight_only" != "true" ]' in resolve_step
    assert '[ -n "$release_tag" ]' in resolve_step
    assert '[ "$push_image" != "true" ]' in resolve_step
    assert "GB10 full release publication requires push-image=true" in resolve_step


def test_gb10_release_workflow_requires_durable_image_ref_for_tagged_release():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    resolve_step = gb10_workflow.split(
        "- name: Resolve release settings",
        1,
    )[1].split("- name: Write GB10 release manifest", 1)[0]

    assert '[[ "$image_name" != ghcr.io/* ]]' in resolve_step
    assert "requires a GHCR image-name" in resolve_step
    assert '[[ "$image_name" == *:* || "$image_name" == *@* ]]' in resolve_step
    assert "must not include a tag or digest" in resolve_step
    assert "image_repository=\"${image_name#ghcr.io/}\"" in resolve_step
    assert "IFS=/ read -r -a image_repository_parts" in resolve_step
    assert "must be a lowercase Docker repository name" in resolve_step
    assert (
        '[[ ! "$image_tag" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]'
        in resolve_step
    )
    assert "runtime image tag must be a Docker-compatible tag" in resolve_step
    assert '[ "$image_tag" != "$release_tag" ]' in resolve_step
    assert "runtime image tag must match the release tag" in resolve_step


def test_gb10_release_workflow_rejects_multiline_env_values():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    resolve_step = gb10_workflow.split(
        "- name: Resolve release settings",
        1,
    )[1].split("- name: Write GB10 release manifest", 1)[0]

    assert "reject_multiline_env_value()" in resolve_step
    assert '$\'\\n\'' in resolve_step
    assert '$\'\\r\'' in resolve_step
    assert "GB10 release setting must be single-line" in resolve_step
    for variable in (
        "release_tag",
        "image_name",
        "image_tag",
        "vllm_version",
        "prebuilt_wheel_urls",
        "flash_attn_repo",
        "flash_attn_ref",
        "push_image",
        "preflight_only",
    ):
        assert f'reject_multiline_env_value "{variable}" "${variable}"' in resolve_step


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
    assert (
        "group: ${{ github.workflow }}-${{ github.ref }}-${{ inputs['image-ref'] }}"
        in smoke_workflow
    )
    assert "cancel-in-progress: true" in smoke_workflow


def test_gb10_release_workflow_uses_modest_remote_parallelism():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert 'GB10_MAX_JOBS: "24"' in gb10_workflow
    assert 'GB10_NVCC_THREADS: "8"' in gb10_workflow
    assert "24 / 8 = 3 jobs" in gb10_workflow
    assert gb10_workflow.count('--build-arg max_jobs="$GB10_MAX_JOBS"') == 2
    assert gb10_workflow.count('--build-arg nvcc_threads="$GB10_NVCC_THREADS"') == 2


def test_gb10_release_workflow_publishes_release_manifest():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
    ).read_text()

    assert "GB10_RELEASE_MANIFEST_DIR" in gb10_workflow
    assert "GB10_RUNTIME_IMAGE_METADATA_JSON" in gb10_workflow
    assert "GB10_RELEASE_MANIFEST_DIR: gb10-release-manifest" in gb10_workflow
    assert (
        "GB10_RUNTIME_IMAGE_METADATA_JSON: "
        "gb10-release-manifest/buildx-runtime-image-metadata.json"
    ) in gb10_workflow
    assert "${{ github.workspace }}/gb10-release-manifest" not in gb10_workflow
    assert "Write GB10 release manifest" in gb10_workflow
    assert "scripts/gb10-write-release-manifest.py" in gb10_workflow
    assert "--gb10-validate-release-inputs" in gb10_workflow
    assert "gb10-release-manifest.json" in gb10_workflow
    assert "buildx-runtime-image-metadata.json" in gb10_workflow
    assert "gb10-runtime-image-ref.txt" in gb10_workflow
    assert "gb10-runtime-image-digest.txt" in gb10_workflow
    assert "gb10-vllm-release-SHA256SUMS" in gb10_workflow
    assert "Write GB10 runtime image refs" in gb10_workflow
    assert "Write GB10 release checksums" in gb10_workflow
    assert "Validate GB10 release assets" in gb10_workflow
    assert "Upload GB10 release manifest" in gb10_workflow
    assert "name: gb10-release-manifest" in gb10_workflow
    assert "if: always()" in gb10_workflow
    assert "Publish GB10 release assets" in gb10_workflow
    assert "Publish wheel to GitHub Release" not in gb10_workflow
    assert '"$GB10_RELEASE_MANIFEST_DIR/gb10-release-manifest.json"' in (
        gb10_workflow
    )
    assert '--metadata-file "$GB10_RUNTIME_IMAGE_METADATA_JSON"' in gb10_workflow

    assert gb10_workflow.index("Write GB10 release manifest") < (
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

    validation_step = gb10_workflow.split(
        "- name: Validate GB10 release assets",
        1,
    )[1].split("- name: Upload GB10 release manifest", 1)[0]
    assert "env.GB10_PREFLIGHT_ONLY != 'true'" in validation_step
    assert "env.GB10_RELEASE_TAG != ''" in validation_step
    assert "dist/vllm-*.whl" in validation_step
    assert "GB10 release publication expects exactly one vLLM wheel" in (
        validation_step
    )
    assert "GB10 release asset is missing or empty" in validation_step
    assert "sha256sum --check" in validation_step
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-release-manifest.json" in validation_step
    assert "$GB10_RUNTIME_IMAGE_METADATA_JSON" in validation_step
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-runtime-image-ref.txt" in validation_step
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-runtime-image-digest.txt" in (
        validation_step
    )
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-vllm-release-SHA256SUMS" in (
        validation_step
    )

    release_step = gb10_workflow.split(
        "- name: Publish GB10 release assets",
        1,
    )[1]
    assert "env.GB10_PREFLIGHT_ONLY != 'true'" in release_step
    assert "env.GB10_RELEASE_TAG != ''" in release_step
    assert "dist/vllm-*.whl" in release_step
    assert "GB10 release publication expects exactly one vLLM wheel" in release_step
    assert "release_assets=(" in release_step
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-release-manifest.json" in release_step
    assert "$GB10_RUNTIME_IMAGE_METADATA_JSON" in release_step
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-runtime-image-ref.txt" in release_step
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-runtime-image-digest.txt" in release_step
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-vllm-release-SHA256SUMS" in release_step
    assert 'if [ -f "$GB10_RUNTIME_IMAGE_METADATA_JSON" ]' not in release_step
    assert (
        'if [ -f "$GB10_RELEASE_MANIFEST_DIR/gb10-runtime-image-digest.txt" ]'
        not in release_step
    )

    refs_step = gb10_workflow.split(
        "- name: Write GB10 runtime image refs",
        1,
    )[1].split("- name: Write GB10 release checksums", 1)[0]
    assert "containerimage.digest" in refs_step
    assert "GB10 pushed runtime image metadata did not include a digest" in refs_step
    assert '[[ ! "$image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]' in refs_step
    assert "GB10 runtime image metadata digest is not a valid sha256 digest" in (
        refs_step
    )
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-runtime-image-ref.txt" in refs_step
    assert "$GB10_RELEASE_MANIFEST_DIR/gb10-runtime-image-digest.txt" in refs_step


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
        "GB10_MAX_JOBS": "24",
        "GB10_NVCC_THREADS": "8",
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
    assert data["build"]["parallelism"] == {"max_jobs": "24", "nvcc_threads": "8"}
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
    assert support_matrix["entries"]["flashinfer_nvfp4_dense"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["flashmla_attention"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["public_flashattention_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["trtllm_gen_attention"]["status"] == (
        "not_supported"
    )
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
    assert "model:" in smoke_workflow
    assert "served-model-name:" in smoke_workflow
    assert "require-moe:" in smoke_workflow
    assert "require-openai-deterministic:" in smoke_workflow
    assert "runs-on: [self-hosted, linux, aarch64, cuda13, dgx-spark, sm121]" in (
        smoke_workflow
    )
    assert "permissions:" in smoke_workflow
    assert "contents: write" in smoke_workflow
    assert "packages: read" in smoke_workflow
    assert "actions: read" in smoke_workflow
    assert "docker/login-action@v3" in smoke_workflow
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
    assert "gb10-smoked-image-digest.txt" in smoke_workflow
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
    assert "--name gb10-release-manifest" in smoke_workflow
    assert "--dir \"$GB10_RELEASE_PROVENANCE_DIR\"" in smoke_workflow
    assert "GB10_RELEASE_MANIFEST_JSON=$manifest" in smoke_workflow
    assert "GB10_RUNTIME_IMAGE_METADATA_JSON=$runtime_metadata" in smoke_workflow
    assert "gb10-runtime-image-ref.txt" in smoke_workflow
    assert "gb10-runtime-image-digest.txt" in smoke_workflow
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
    missing_run_id_guard = (
        "inputs['publish-to-release'] && "
        "inputs['release-workflow-run-id'] == ''"
    )
    assert missing_run_id_guard in smoke_workflow
    assert "inputs['publish-to-release'] && inputs['release-tag'] == ''" in (
        smoke_workflow
    )
    assert "inputs['release-workflow-run-id'] != ''" in smoke_workflow
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
    assert "actions/upload-artifact@v4" in smoke_workflow
    assert "gb10-smoke-reports/**" in smoke_workflow
    assert "gb10-release-provenance/**" in smoke_workflow
    assert "dist/gb10-release-evidence/**" in smoke_workflow
    assert "Validate GB10 evidence release assets" in smoke_workflow
    assert "id: validate_evidence_release_assets" in smoke_workflow
    validation_block_match = re.search(
        r'python3 - "\$GB10_RELEASE_EVIDENCE_OUTPUT_DIR/'
        r"release-evidence-metadata\.json\" <<'PY'\n"
        r"(?P<block>.*?)\n          PY",
        smoke_workflow,
        re.DOTALL,
    )
    assert validation_block_match is not None
    validation_block = validation_block_match.group("block")
    assert "GB10 evidence release asset is missing or empty" in smoke_workflow
    assert "sha256sum --check \"$GB10_RELEASE_EVIDENCE_OUTPUT_DIR/SHA256SUMS\"" in (
        smoke_workflow
    )
    assert (
        "sha256sum --check "
        '"$GB10_RELEASE_EVIDENCE_OUTPUT_DIR/gb10-release-evidence.tar.gz.sha256"'
    ) in smoke_workflow
    assert (
        "GB10 evidence release metadata does not prove a passed release gate"
        in smoke_workflow
    )
    assert 'metadata.get("status") != "complete"' in smoke_workflow
    assert 'metadata.get("release_gate_passed") is not True' in smoke_workflow
    image_digest_mismatch = (
        "GB10 evidence release metadata image digest does not match pulled digest"
    )
    assert image_digest_mismatch in smoke_workflow
    image_ref_mismatch = (
        "GB10 evidence release metadata image ref does not match input image-ref"
    )
    assert image_ref_mismatch in smoke_workflow
    assert "GB10 evidence release metadata tag does not match release-tag" in (
        smoke_workflow
    )
    assert (
        "GB10 evidence release metadata is missing a GB10 support matrix summary"
        in smoke_workflow
    )
    assert (
        "GB10 evidence release metadata does not prove a complete GB10 support matrix"
        in smoke_workflow
    )
    assert 'metadata.get("support_matrix_complete") is not True' in (
        validation_block
    )
    assert 'support_matrix = metadata.get("support_matrix_summary")' in (
        validation_block
    )
    assert 'support_matrix.get("present") is not True' in validation_block
    assert (
        'support_matrix.get("release_manifest_present") is not True'
        in validation_block
    )
    assert 'support_matrix.get("architecture") != "sm_121a"' in validation_block
    assert (
        'support_matrix.get("first_release_scope") != "single_spark_first_path"'
        in validation_block
    )
    assert 'support_matrix.get("entry_count", 0) <= 0' in validation_block
    assert 'support_matrix.get("invalid_entries")' in validation_block
    assert "GB10 evidence release metadata support matrix does not match" in (
        smoke_workflow
    )
    assert "required_support_matrix = {" in validation_block
    assert '"flashinfer_nvfp4_dense": "supported_native"' in validation_block
    assert '"flashinfer_nvfp4_quantization": "supported_native"' in validation_block
    assert '"flashinfer_attention_fa2": "supported_native"' in validation_block
    assert '"flashinfer_b12x_non_ep_moe": "supported_native"' in validation_block
    assert '"flashmla_attention": "supported_native"' in validation_block
    assert '"public_flashattention_runtime": "not_supported"' in validation_block
    assert '"trtllm_gen_attention": "not_supported"' in validation_block
    assert '"trtllm_gen_moe": "not_supported"' in validation_block
    assert '"marlin_nvfp4_fallback": "not_supported"' in validation_block
    assert '"flashinfer_b12x_ep_all2all_eplb": "deferred"' in validation_block
    assert '"multi_spark_ep_all2all_eplb": "deferred"' in validation_block
    assert 'entries = support_matrix.get("entries")' in validation_block
    assert "mismatched_support = {" in validation_block
    assert validation_block.index(
        'support_matrix = metadata.get("support_matrix_summary")'
    ) < validation_block.index(
        'metadata.get("status") != "complete"'
    )
    assert "GB10 evidence release metadata is missing release provenance" in (
        smoke_workflow
    )
    assert 'source = metadata.get("source")' in smoke_workflow
    assert 'source.get("image_ref") != os.environ["GB10_IMAGE_REF"]' in smoke_workflow
    assert (
        'normalize_image_digest(source.get("image_digest"))'
        in validation_block
    )
    assert (
        'normalize_image_digest(os.environ.get("GB10_IMAGE_DIGEST"))'
        in validation_block
    )
    assert "          import re" in validation_block
    assert "re.fullmatch" in validation_block
    assert (
        'source.get("release_tag") != os.environ["GB10_RELEASE_TAG"]'
        in smoke_workflow
    )
    assert 'required_provenance = {"release_manifest", "runtime_image_metadata"}' in (
        smoke_workflow
    )
    assert "$GB10_RELEASE_EVIDENCE_OUTPUT_DIR/gb10-release-evidence.tar.gz" in (
        smoke_workflow
    )
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
    assert "gb10-release-evidence.tar.gz" in smoke_workflow
    assert "release-evidence-metadata.json" in smoke_workflow
    assert "SHA256SUMS" in smoke_workflow


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


def test_gb10_local_cached_build_script_validates_manifest_before_buildx():
    script = (REPO_ROOT / "scripts" / "gb10-build-cached.sh").read_text()

    assert 'GITHUB_SHA="${GITHUB_SHA:-$(git rev-parse HEAD)}"' in script
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
    assert "flashinfer-python" in script
    assert "flashinfer-cubin" in script
    assert "flashinfer-jit-cache" in script
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


def test_gb10_nvfp4_model_smoke_builds_release_summary():
    smoke = _load_gb10_smoke_module()
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
        "status": "not_validated_by_smoke",
        "configured_mode": "PIECEWISE",
        "configured_enabled": True,
    }
    assert "OpenAI-compatible server smoke" in release_summary[
        "remaining_release_evidence"
    ]


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

    assert "scripts/gb10-smoke-image.sh" in script
    assert "scripts/gb10-smoke-openai-image.sh" in script
    assert "scripts/gb10-verify-release-evidence.py" in script
    assert "scripts/gb10-bundle-release-evidence.py" in script
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
    assert "gb10-nvfp4-smoke.json" in script
    assert "gb10-openai-server-smoke-image.json" in script
    assert "gb10-release-evidence-image.json" in script
    assert "gb10-smoked-image-digest.txt" in script
    assert "gb10-release-evidence.tar.gz" in script
    assert "--gb10-report-json /gb10-smoke-reports/gb10-nvfp4-smoke.json" in script
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
    assert 'bundle_args+=(--gb10-release-tag "$GB10_RELEASE_TAG")' in script
    assert 'bundle_args+=(--gb10-release-manifest-json' in script
    assert 'bundle_args+=(--gb10-runtime-image-metadata-json' in script
    assert 'bundle_args+=(--gb10-allow-partial)' in script
    assert 'exit "$verify_status"' in script


def test_gb10_release_evidence_bundle_preserves_smoke_artifacts():
    script = (REPO_ROOT / "scripts" / "gb10-bundle-release-evidence.py").read_text()

    assert "Bundle GB10 release smoke reports" in script
    assert "EXPECTED_REPORTS" in script
    assert "EXPECTED_EVIDENCE_FILES" in script
    assert "gb10-nvfp4-smoke.json" in script
    assert "gb10-openai-server-smoke-image.json" in script
    assert "gb10-release-evidence-image.json" in script
    assert "gb10-smoked-image-digest.txt" in script
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
    assert "provenance/gb10-release-manifest.json" in script
    assert "provenance/buildx-runtime-image-metadata.json" in script
    assert "--gb10-allow-partial" in script
    assert "--gb10-include-glob" in script
    assert "gb10-*.txt" in script
    assert "release-evidence-metadata.json" in script
    assert "SHA256SUMS" in script
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
    assert "sha256:[0-9a-f]{64}" in script
    assert "tarfile.open" in script
    assert "hashlib.sha256" in script
    assert "_metadata_status(" in script


def test_gb10_release_evidence_bundle_builds_metadata_and_tarball(tmp_path):
    bundler = _load_gb10_release_bundle_module()
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
    metadata_path = output_dir / "release-evidence-metadata.json"
    checksum_path = output_dir / "SHA256SUMS"
    archive_path = output_dir / "evidence.tar.gz"
    archive_checksum_path = output_dir / "evidence.tar.gz.sha256"
    assert metadata_path.exists()
    assert checksum_path.exists()
    assert archive_path.exists()
    assert archive_checksum_path.exists()

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
        "status_counts": {
            "deferred": 2,
            "not_supported": 4,
            "supported_native": 5,
        },
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
    metadata = json.loads((output_dir / "release-evidence-metadata.json").read_text())
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
    metadata = json.loads((output_dir / "release-evidence-metadata.json").read_text())
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
            "checks": {
                "kv_cache_dtype": {
                    "status": "passed",
                    "expected": "fp8_e4m3",
                    "configured": "fp8_e4m3",
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
                "flashinfer_nvfp4_dense": {"status": "supported_native"},
                "flashinfer_nvfp4_quantization": {"status": "supported_native"},
                "flashinfer_attention_fa2": {"status": "supported_native"},
                "flashinfer_b12x_non_ep_moe": {"status": "supported_native"},
                "flashmla_attention": {"status": "supported_native"},
                "public_flashattention_runtime": {"status": "not_supported"},
                "trtllm_gen_attention": {"status": "not_supported"},
                "trtllm_gen_moe": {"status": "not_supported"},
                "marlin_nvfp4_fallback": {"status": "not_supported"},
                "flashinfer_b12x_ep_all2all_eplb": {"status": "deferred"},
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
    check_statuses = {check["name"]: check["status"] for check in summary["checks"]}
    assert check_statuses["native_nvfp4_gemm_observed"] == "passed"
    assert check_statuses["native_nvfp4_moe_non_ep_observed"] == "passed"
    assert check_statuses["nvfp4_fallback_free"] == "passed"
    assert check_statuses["gb10_device_sm121"] == "passed"
    assert check_statuses["flashinfer_gb10_runtime_version"] == "passed"
    assert check_statuses["flashinfer_gb10_distribution_versions"] == "passed"
    assert check_statuses["attention_backend_flashinfer"] == "passed"
    assert check_statuses["attention_backend_allowed_by_support_matrix"] == "passed"
    assert check_statuses["quantization_modelopt_fp4"] == "passed"
    assert check_statuses["quantization_allowed_by_support_matrix"] == "passed"
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
