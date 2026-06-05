#!/usr/bin/env python3
"""Resolve GB10 release workflow settings into GitHub Actions env values."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path

from gb10_release_contract import REQUIRED_FLASHINFER_COMPONENTS

DEFAULT_IMAGE_NAME = "ghcr.io/gardner/vllm-gb10"
DEFAULT_FLASH_ATTN_REPO = "https://github.com/gardner/vllm-flash-attention.git"
DEFAULT_FLASH_ATTN_REF = "de3849e75d07edd1c00aec02c92ec852ba757adc"
DEFAULT_VLLM_VERSION_BASE = "0.22.1rc0"
DEFAULT_FLASHINFER_RELEASE_TAG = "gb10-flashinfer-v0.6.12-1c80efb3"
DEFAULT_PREBUILT_WHEEL_URLS = " ".join(
    (
        "https://github.com/gardner/flashinfer/releases/download/"
        f"{DEFAULT_FLASHINFER_RELEASE_TAG}/"
        "flashinfer_python-0.6.12+cu130gb10-py3-none-any.whl",
        "https://github.com/gardner/flashinfer/releases/download/"
        f"{DEFAULT_FLASHINFER_RELEASE_TAG}/"
        "flashinfer_cubin-0.6.12+cu130gb10-py3-none-any.whl",
        "https://github.com/gardner/flashinfer/releases/download/"
        f"{DEFAULT_FLASHINFER_RELEASE_TAG}/"
        "flashinfer_jit_cache-0.6.12+cu130gb10-"
        "cp39-abi3-manylinux_2_28_aarch64.whl",
    )
)
DEFAULT_SELF_HOSTED_RUNNER_LABELS = (
    '["self-hosted","linux","aarch64","cuda13","dgx-spark","sm121"]'
)
DOCKER_REPOSITORY_COMPONENT_RE = re.compile(r"[a-z0-9]+(?:[._-]+[a-z0-9]+)*")
DOCKER_TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")

RESOLVED_ENV_KEYS = (
    "GB10_RELEASE_TAG",
    "GB10_IMAGE_NAME",
    "GB10_IMAGE_TAG",
    "GB10_VLLM_VERSION",
    "GB10_PREBUILT_WHEEL_URLS",
    "GB10_FLASH_ATTN_REPO",
    "GB10_FLASH_ATTN_REF",
    "GB10_PUSH_IMAGE",
    "GB10_PREFLIGHT_ONLY",
    "GB10_MAX_JOBS",
    "GB10_NVCC_THREADS",
    "GB10_RUNNER_LABELS",
)


def _env(env: Mapping[str, str], name: str, default: str = "") -> str:
    return env.get(name, default)


def _env_nonempty(env: Mapping[str, str], name: str, default: str = "") -> str:
    value = env.get(name)
    return default if value is None or value == "" else value


def _bool_string(name: str, value: str) -> str:
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return "true"
    if normalized_value in {"", "0", "false", "no", "off"}:
        return "false"
    raise ValueError(
        f"Unsupported GB10 boolean setting {name}={value}. "
        "Use 1, 0, true, false, yes, no, on, or off."
    )


def _is_positive_integer(value: str) -> bool:
    return re.fullmatch(r"[1-9][0-9]*", value) is not None


def _reject_multiline(name: str, value: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError(f"GB10 release setting must be single-line: {name}")


def _runner_labels(value: str) -> list[str]:
    try:
        labels = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("GB10 runner-labels must be a JSON string list.") from exc
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError("GB10 runner-labels must be a JSON string list.")
    return labels


def _validate_full_release_runner_labels(labels_json: str) -> None:
    if "self-hosted" not in _runner_labels(labels_json):
        raise ValueError(
            "GB10 full release builds require self-hosted runner labels; "
            "GitHub-hosted ARM runners repeatedly terminate the multi-hour "
            "wheel build without logs."
        )


def _validate_required_flashinfer_wheels(prebuilt_wheel_urls: str) -> None:
    if not prebuilt_wheel_urls:
        raise ValueError("GB10 prebuilt FlashInfer wheel URLs are required.")
    for required_wheel in REQUIRED_FLASHINFER_COMPONENTS:
        if required_wheel not in prebuilt_wheel_urls:
            raise ValueError(
                f"GB10 prebuilt wheel URLs must include {required_wheel}."
            )


def _validate_tagged_release_image(image_name: str, image_tag: str) -> None:
    if not image_name.startswith("ghcr.io/"):
        raise ValueError(
            f"GB10 full release publication requires a GHCR image-name, "
            f"got {image_name}."
        )
    if ":" in image_name or "@" in image_name:
        raise ValueError(
            f"GB10 full release image-name must not include a tag or digest, "
            f"got {image_name}."
        )

    image_repository = image_name.removeprefix("ghcr.io/")
    image_repository_parts = image_repository.split("/")
    if len(image_repository_parts) < 2:
        raise ValueError(
            "GB10 full release image-name must include owner and package "
            f"components, got {image_name}."
        )
    for image_repository_part in image_repository_parts:
        if DOCKER_REPOSITORY_COMPONENT_RE.fullmatch(image_repository_part) is None:
            raise ValueError(
                "GB10 full release image-name must be a lowercase Docker "
                f"repository name, got {image_name}."
            )

    if DOCKER_TAG_RE.fullmatch(image_tag) is None:
        raise ValueError(
            "GB10 full release runtime image tag must be a Docker-compatible "
            f"tag, got {image_tag}."
        )


def _release_tag_from_ref(github_ref: str) -> str:
    if github_ref.startswith("refs/tags/"):
        return github_ref.removeprefix("refs/tags/")
    return ""


def _vllm_version_base(release_tag: str) -> str:
    match = re.fullmatch(r"gb10-vllm-v([^-]+)-[0-9a-f]{7,40}", release_tag)
    if match is None:
        return DEFAULT_VLLM_VERSION_BASE
    return match.group(1)


def resolve_release_settings(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return resolved GB10 release env values from GitHub Actions inputs."""

    env = os.environ if env is None else env
    event_name = _env(env, "GITHUB_EVENT_NAME")
    github_sha = _env(env, "GITHUB_SHA")
    if len(github_sha) < 12:
        raise ValueError("GITHUB_SHA must contain at least 12 characters.")

    if event_name == "push":
        release_tag = _release_tag_from_ref(_env(env, "GITHUB_REF"))
        image_name = DEFAULT_IMAGE_NAME
        prebuilt_wheel_urls = _env_nonempty(
            env,
            "GB10_DEFAULT_PREBUILT_WHEEL_URLS",
            DEFAULT_PREBUILT_WHEEL_URLS,
        )
        flash_attn_repo = DEFAULT_FLASH_ATTN_REPO
        flash_attn_ref = DEFAULT_FLASH_ATTN_REF
        push_image = "true"
        preflight_only = "false"
        release_runner_labels = _env(
            env,
            "GB10_SELF_HOSTED_RUNNER_LABELS",
            DEFAULT_SELF_HOSTED_RUNNER_LABELS,
        )
        max_jobs = _env(env, "GB10_MAX_JOBS", "1")
        nvcc_threads = _env(env, "GB10_NVCC_THREADS", "1")
    else:
        release_tag = _env(env, "GB10_INPUT_RELEASE_TAG")
        image_name = _env_nonempty(env, "GB10_INPUT_IMAGE_NAME", DEFAULT_IMAGE_NAME)
        prebuilt_wheel_urls = _env_nonempty(env, "GB10_INPUT_PREBUILT_WHEEL_URLS")
        flash_attn_repo = _env_nonempty(
            env,
            "GB10_INPUT_FLASH_ATTN_REPO",
            DEFAULT_FLASH_ATTN_REPO,
        )
        flash_attn_ref = _env_nonempty(
            env,
            "GB10_INPUT_FLASH_ATTN_REF",
            DEFAULT_FLASH_ATTN_REF,
        )
        push_image = _bool_string(
            "GB10_INPUT_PUSH_IMAGE",
            _env(env, "GB10_INPUT_PUSH_IMAGE", "true"),
        )
        preflight_only = _bool_string(
            "GB10_INPUT_PREFLIGHT_ONLY",
            _env(env, "GB10_INPUT_PREFLIGHT_ONLY"),
        )
        release_runner_labels = _env(
            env,
            "GB10_INPUT_RUNNER_LABELS",
            '["ubuntu-22.04-arm"]',
        )
        max_jobs = _env(env, "GB10_INPUT_MAX_JOBS", _env(env, "GB10_MAX_JOBS", "1"))
        nvcc_threads = _env(
            env,
            "GB10_INPUT_NVCC_THREADS",
            _env(env, "GB10_NVCC_THREADS", "1"),
        )

    if not prebuilt_wheel_urls:
        prebuilt_wheel_urls = _env_nonempty(
            env,
            "GB10_DEFAULT_PREBUILT_WHEEL_URLS",
            DEFAULT_PREBUILT_WHEEL_URLS,
        )

    image_tag_override = ""
    if event_name != "push":
        image_tag_override = _env_nonempty(env, "GB10_INPUT_IMAGE_TAG")
    image_tag = image_tag_override or release_tag or f"gb10-{github_sha[:12]}"
    vllm_version = f"{_vllm_version_base(release_tag)}+gb10.{github_sha[:12]}"

    settings = {
        "GB10_RELEASE_TAG": release_tag,
        "GB10_IMAGE_NAME": image_name,
        "GB10_IMAGE_TAG": image_tag,
        "GB10_VLLM_VERSION": vllm_version,
        "GB10_PREBUILT_WHEEL_URLS": prebuilt_wheel_urls,
        "GB10_FLASH_ATTN_REPO": flash_attn_repo,
        "GB10_FLASH_ATTN_REF": flash_attn_ref,
        "GB10_PUSH_IMAGE": push_image,
        "GB10_PREFLIGHT_ONLY": preflight_only,
        "GB10_MAX_JOBS": max_jobs,
        "GB10_NVCC_THREADS": nvcc_threads,
        "GB10_RUNNER_LABELS": release_runner_labels,
    }

    for name, value in settings.items():
        _reject_multiline(name, value)

    if not _is_positive_integer(max_jobs):
        raise ValueError(f"GB10 max-jobs must be a positive integer, got {max_jobs}.")
    if not _is_positive_integer(nvcc_threads):
        raise ValueError(
            f"GB10 nvcc-threads must be a positive integer, got {nvcc_threads}."
        )

    if preflight_only != "true":
        _validate_full_release_runner_labels(release_runner_labels)

    if preflight_only != "true" and release_tag and push_image != "true":
        raise ValueError(
            "GB10 full release publication requires push-image=true so the "
            "runtime image is durable in GHCR."
        )

    _validate_required_flashinfer_wheels(prebuilt_wheel_urls)

    if preflight_only != "true" and release_tag:
        _validate_tagged_release_image(image_name, image_tag)
        if image_tag != release_tag:
            raise ValueError(
                "GB10 full release runtime image tag must match the release tag."
            )

    return settings


def write_github_env(path: Path, settings: Mapping[str, str]) -> None:
    """Append resolved settings to a GitHub Actions environment file."""

    lines = [f"{key}={settings[key]}" for key in RESOLVED_ENV_KEYS]
    with path.open("a", encoding="utf-8") as env_file:
        env_file.write("\n".join(lines) + "\n")


def shell_env_lines(settings: Mapping[str, str]) -> list[str]:
    """Return shell-safe exports for resolved settings."""

    return [f"export {key}={shlex.quote(settings[key])}" for key in RESOLVED_ENV_KEYS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gb10-output-env",
        type=Path,
        default=Path(os.environ["GITHUB_ENV"]) if "GITHUB_ENV" in os.environ else None,
        help="GitHub Actions env file to append resolved settings to.",
    )
    parser.add_argument(
        "--gb10-output-shell",
        action="store_true",
        help="Print shell-safe export statements for resolved settings.",
    )
    args = parser.parse_args()

    try:
        settings = resolve_release_settings()
        if args.gb10_output_env is not None and args.gb10_output_shell:
            raise ValueError("Use only one GB10 release settings output mode.")
        if args.gb10_output_env is not None:
            write_github_env(args.gb10_output_env, settings)
        elif args.gb10_output_shell:
            for line in shell_env_lines(settings):
                print(line)
        else:
            for key in RESOLVED_ENV_KEYS:
                print(f"{key}={settings[key]}")
    except ValueError as exc:
        print(f"GB10 release settings error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
