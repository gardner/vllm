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
    "gb10_attention_trtllm_gen_to_flashinfer_fa2": "supported_routed",
    "gb10_attention_public_flashattention_to_flashinfer_or_flashmla": (
        "supported_routed"
    ),
    "gb10_moe_trtllm_gen_to_flashinfer_non_ep": "supported_routed",
    "public_flashattention_runtime": "not_supported",
    "flashinfer_trtllm_nvfp4_dense": "not_supported",
    "trtllm_gen_attention": "not_supported",
    "trtllm_gen_moe": "not_supported",
    "marlin_nvfp4_fallback": "not_supported",
    "marlin_mxfp4_fallback": "not_supported",
    "mxfp4_moe_fallback": "not_supported",
    "quark_nvfp4_checkpoint_loading": "not_supported",
    "compressed_tensors_w4a16_nvfp4_loading": "not_supported",
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


def test_gb10_release_workflow_routes_full_builds_to_self_hosted_gb10():
    gb10_workflow = (
        REPO_ROOT / ".github" / "workflows" / "gb10-release.yml"
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
    assert "runner_labels='${{ inputs['runner-labels'] }}'" in resolve_step
    assert 'release_runner_labels="$GB10_SELF_HOSTED_RUNNER_LABELS"' in (
        resolve_step
    )
    assert "GB10_RUNNER_LABELS<<GB10_RUNNER_LABELS_EOF" in resolve_step
    assert 'echo "$release_runner_labels"' in resolve_step
    assert "reject_multiline_env_value \"release_runner_labels\"" in resolve_step
    assert '[ "$preflight_only" != "true" ]' in resolve_step
    assert '[[ "$release_runner_labels" != *"\\"self-hosted\\""* ]]' in (
        resolve_step
    )
    assert "GB10 full release builds require self-hosted runner labels" in (
        resolve_step
    )


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

    assert "max-jobs:" in gb10_workflow
    assert "nvcc-threads:" in gb10_workflow
    assert 'GB10_MAX_JOBS: "1"' in gb10_workflow
    assert 'GB10_NVCC_THREADS: "1"' in gb10_workflow
    assert 'GB10_NATIVE_CUDA_ARCHS_ONLY: "1"' in gb10_workflow
    assert "1 / 1 = 1 job" in gb10_workflow
    assert 'max_jobs="${{ inputs[\'max-jobs\'] }}"' in gb10_workflow
    assert 'nvcc_threads="${{ inputs[\'nvcc-threads\'] }}"' in gb10_workflow
    assert 'echo "GB10_MAX_JOBS=${max_jobs}"' in gb10_workflow
    assert 'echo "GB10_NVCC_THREADS=${nvcc_threads}"' in gb10_workflow
    assert "GB10 max-jobs must be a positive integer" in gb10_workflow
    assert "GB10 nvcc-threads must be a positive integer" in gb10_workflow
    assert gb10_workflow.count('--build-arg max_jobs="$GB10_MAX_JOBS"') == 2
    assert gb10_workflow.count('--build-arg nvcc_threads="$GB10_NVCC_THREADS"') == 2
    assert gb10_workflow.count(
        '--build-arg vllm_native_cuda_archs_only="$GB10_NATIVE_CUDA_ARCHS_ONLY"'
    ) == 2


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
    stale_digest = manifest_dir / "gb10-runtime-image-digest.txt"
    manifest_dir.mkdir()
    stale_digest.write_text("sha256:" + "b" * 64 + "\n", encoding="utf-8")

    errors = writer.write_runtime_image_provenance(
        runtime_image_metadata_json=metadata_json,
        release_manifest_dir=manifest_dir,
        image_name="ghcr.io/gardner/vllm-gb10",
        image_tag="gb10-test",
        push_image=True,
    )

    assert errors == ["GB10 pushed runtime image metadata did not include a digest."]
    assert (
        manifest_dir / "gb10-runtime-image-ref.txt"
    ).read_text() == "ghcr.io/gardner/vllm-gb10:gb10-test\n"
    assert not stale_digest.exists()


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
    assert (
        manifest_dir / "gb10-runtime-image-ref.txt"
    ).read_text() == "ghcr.io/gardner/vllm-gb10:gb10-test\n"
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

    for path, content in (
        (wheel, b"wheel"),
        (manifest, b'{"schema_version":1}\n'),
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
    assert support_matrix["entries"]["flashmla_attention"]["status"] == (
        "supported_native"
    )
    assert support_matrix["entries"]["public_flashattention_runtime"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["flashinfer_trtllm_nvfp4_dense"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["marlin_mxfp4_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["mxfp4_moe_fallback"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["quark_nvfp4_checkpoint_loading"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["compressed_tensors_w4a16_nvfp4_loading"][
        "status"
    ] == "not_supported"
    assert support_matrix["entries"]["trtllm_gen_attention"]["status"] == (
        "not_supported"
    )
    assert support_matrix["entries"]["flashinfer_cudnn_nvfp4_dense"]["status"] == (
        "deferred"
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
    compressed_tensors_mxfp4_moe = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" /
        "quantization" / "compressed_tensors" / "compressed_tensors_moe" /
        "compressed_tensors_moe_w4a4_mxfp4.py"
    ).read_text()
    mxfp4_moe_oracle = (
        REPO_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" /
        "oracle" / "mxfp4.py"
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
    assert "MarlinMxFp4LinearKernel" in linear_selector
    assert "_gb10_mxfp4_linear_fallback_unsupported_reason" in linear_selector
    assert "non-native MXFP4 dense fallback" in linear_selector
    assert "FlashInfer TRTLLM NVFP4 dense is not supported on GB10/SM12x" in (
        flashinfer_nvfp4_linear
    )
    assert "FlashInfer cuDNN NVFP4 dense is deferred on GB10/SM12x" in (
        flashinfer_nvfp4_linear
    )

    assert "W4A16_NVFP4 linear selected MarlinNvFp4LinearKernel" in modelopt_quant
    assert "_gb10_w4a16_nvfp4_marlin_unsupported_reason" in modelopt_quant
    assert "not supported on GB10/SM12x" in modelopt_quant
    assert "record_nvfp4_backend_selection" in modelopt_quant
    assert "record_nvfp4_fallback" in modelopt_quant
    assert '"linear_w4a16"' in modelopt_quant
    assert "weight-only fallback path" in modelopt_quant
    assert "not the native GB10 W4A4 FP4 Tensor " in modelopt_quant
    assert "Core path; verify this fallback is intentional " in modelopt_quant

    assert "_gb10_w4a16_nvfp4_marlin_unsupported_reason" in (
        compressed_tensors_w4a16
    )
    assert "CompressedTensors W4A16 NVFP4 loading would select" in (
        compressed_tensors_w4a16
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_w4a16
    assert "_gb10_mxfp4_moe_marlin_unsupported_reason" in (
        compressed_tensors_mxfp4_moe
    )
    assert "CompressedTensors W4A4 MXFP4 MoE would select" in (
        compressed_tensors_mxfp4_moe
    )
    assert "not supported on GB10/SM12x" in compressed_tensors_mxfp4_moe
    assert "_gb10_mxfp4_moe_fallback_unsupported_reason" in mxfp4_moe_oracle
    assert "_MXFP4_MOE_FALLBACK_BACKENDS" in mxfp4_moe_oracle
    assert "CPU fallback" in mxfp4_moe_oracle
    assert "not supported on GB10/SM12x" in mxfp4_moe_oracle
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
    assert "_gb10_unsupported_backend_reason" in nvfp4_oracle
    assert "_gb10_nvfp4_moe_fallback_unsupported_reason" in nvfp4_oracle
    assert "not supported on GB10/SM12x" in nvfp4_oracle
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
                "trtllm_gen_attention": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "TRTLLM Gen attention rejects SM121",
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
                "quark_nvfp4_checkpoint_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "Quark NVFP4 checkpoint loading is not validated",
                },
                "compressed_tensors_w4a16_nvfp4_loading": {
                    "status": "not_supported",
                    "expected_handling": "route_or_reject_before_release_evidence",
                    "reason": "CompressedTensors W4A16 NVFP4 selects Marlin",
                },
            },
            "deferred_paths": {
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
                "flashinfer_attention_fa2": {"status": "supported_native"},
                "flashinfer_b12x_non_ep_moe": {"status": "supported_native"},
                "flashinfer_cutlass_non_ep_moe": {"status": "supported_native"},
                "flashmla_attention": {"status": "supported_native"},
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
                "trtllm_gen_attention": {"status": "not_supported"},
                "trtllm_gen_moe": {"status": "not_supported"},
                "marlin_nvfp4_fallback": {"status": "not_supported"},
                "marlin_mxfp4_fallback": {"status": "not_supported"},
                "mxfp4_moe_fallback": {"status": "not_supported"},
                "quark_nvfp4_checkpoint_loading": {"status": "not_supported"},
                "compressed_tensors_w4a16_nvfp4_loading": {
                    "status": "not_supported"
                },
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
        "compressed_tensors_w4a16_nvfp4_loading",
        "flashinfer_trtllm_nvfp4_dense",
        "marlin_mxfp4_fallback",
        "marlin_nvfp4_fallback",
        "mxfp4_moe_fallback",
        "public_flashattention_runtime",
        "quark_nvfp4_checkpoint_loading",
        "trtllm_gen_attention",
        "trtllm_gen_moe",
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
