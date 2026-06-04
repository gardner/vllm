#!/usr/bin/env python3
"""Fail fast when the GB10 release path can emit non-native CUDA images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

NATIVE_ONLY_BUILD_ARG = (
    '--build-arg vllm_native_cuda_archs_only="$GB10_NATIVE_CUDA_ARCHS_ONLY"'
)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _step_block(workflow: str, step_name: str) -> str:
    marker = f"- name: {step_name}"
    if marker not in workflow:
        return ""
    return workflow.split(marker, 1)[1].split("\n      - name:", 1)[0]


def check_native_cuda_arch_contract(
    repo_root: Path,
    *,
    require_env: bool,
) -> list[str]:
    errors: list[str] = []

    workflow_path = repo_root / ".github" / "workflows" / "gb10-release.yml"
    dockerfile_path = repo_root / "docker" / "Dockerfile"
    cmake_lists_path = repo_root / "CMakeLists.txt"
    cmake_utils_path = repo_root / "cmake" / "utils.cmake"

    workflow = workflow_path.read_text()
    dockerfile = dockerfile_path.read_text()
    cmake_lists = cmake_lists_path.read_text()
    cmake_utils = cmake_utils_path.read_text()

    native_only_env = os.environ.get("GB10_NATIVE_CUDA_ARCHS_ONLY")
    if require_env and not _truthy(native_only_env):
        errors.append(
            "GB10_NATIVE_CUDA_ARCHS_ONLY must be enabled before a release build."
        )
    elif native_only_env is not None and not _truthy(native_only_env):
        errors.append(
            "GB10_NATIVE_CUDA_ARCHS_ONLY is set but disabled; this can emit "
            "cross-major PTX fallback images."
        )

    if 'GB10_NATIVE_CUDA_ARCHS_ONLY: "1"' not in workflow:
        errors.append("gb10-release.yml must default GB10_NATIVE_CUDA_ARCHS_ONLY to 1.")

    if "ARG vllm_native_cuda_archs_only=false" not in dockerfile:
        errors.append("docker/Dockerfile is missing the native CUDA arch build arg.")
    if (
        "ENV VLLM_NATIVE_CUDA_ARCHS_ONLY=${vllm_native_cuda_archs_only}"
        not in dockerfile
    ):
        errors.append(
            "docker/Dockerfile must export VLLM_NATIVE_CUDA_ARCHS_ONLY from "
            "the build arg."
        )

    if workflow.count(NATIVE_ONLY_BUILD_ARG) != 2:
        errors.append(
            "gb10-release.yml must pass vllm_native_cuda_archs_only to both "
            "wheel and runtime Docker builds."
        )
    for step_name in ("Build wheel stage", "Build runtime image"):
        step = _step_block(workflow, step_name)
        if NATIVE_ONLY_BUILD_ARG not in step:
            errors.append(
                f"gb10-release.yml step {step_name!r} is missing "
                "vllm_native_cuda_archs_only."
            )

    if "VLLM_NATIVE_CUDA_ARCHS_ONLY" not in cmake_lists:
        errors.append("CMakeLists.txt must read VLLM_NATIVE_CUDA_ARCHS_ONLY.")
    if "Skipping cross-major PTX fallback CUDA archs" not in cmake_lists:
        errors.append("CMakeLists.txt must log the native CUDA arch-only mode.")

    if "VLLM_NATIVE_CUDA_ARCHS_ONLY" not in cmake_utils:
        errors.append("cmake/utils.cmake must use VLLM_NATIVE_CUDA_ARCHS_ONLY.")
    if "set(_ALLOW_CROSS_MAJOR_PTX_FALLBACK FALSE)" not in cmake_utils:
        errors.append(
            "cmake/utils.cmake must disable cross-major PTX fallback when "
            "native-only mode is enabled."
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gb10-repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="vLLM repository root.",
    )
    parser.add_argument(
        "--gb10-require-env",
        action="store_true",
        help="Require GB10_NATIVE_CUDA_ARCHS_ONLY to be enabled in the environment.",
    )
    args = parser.parse_args()

    errors = check_native_cuda_arch_contract(
        args.gb10_repo_root.resolve(),
        require_env=args.gb10_require_env,
    )
    if errors:
        for error in errors:
            print(f"GB10 native CUDA arch contract error: {error}")
        return 1

    print("GB10 native CUDA arch contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
