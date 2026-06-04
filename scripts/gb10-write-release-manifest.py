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
    return _env(env, name).lower() in {"1", "true", "yes", "on"}


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
            "flashinfer_nvfp4_dense": {
                "status": "supported_native",
                "release_contract": (
                    "First-path smoke must observe native FlashInfer dense "
                    "NVFP4 backend selection."
                ),
            },
            "flashinfer_nvfp4_quantization": {
                "status": "supported_native",
                "release_contract": (
                    "First-path smoke must observe ModelOpt FP4 quantization "
                    "and GB10 FlashInfer runtime packages."
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
            "flashmla_attention": {
                "status": "supported_native",
                "release_contract": (
                    "FlashMLA source provenance is required and vLLM may route "
                    "to native SM121A FlashMLA kernels where selected."
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
            "trtllm_gen_attention": {
                "status": "not_supported",
                "release_contract": (
                    "TRTLLM Gen attention rejects SM121 today; vLLM should "
                    "route GB10 attention through FlashInfer or FlashMLA."
                ),
            },
            "trtllm_gen_moe": {
                "status": "not_supported",
                "release_contract": (
                    "TRTLLM Gen MoE rejects SM121 today and must not satisfy "
                    "native NVFP4 MoE release evidence."
                ),
            },
            "marlin_nvfp4_fallback": {
                "status": "not_supported",
                "release_contract": (
                    "Marlin-backed API smoke can prove serving reachability, "
                    "but it cannot satisfy native NVFP4 release evidence."
                ),
            },
            "flashinfer_b12x_ep_all2all_eplb": {
                "status": "deferred",
                "release_contract": (
                    "Blocked until multi-Spark EP/all-to-all/EPLB contracts "
                    "are validated on hardware."
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
    manifest = build_manifest(env)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


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
            "Validate durable GB10 release inputs after writing the manifest. "
            "Fails on local dependency checkouts, non-SHA refs, missing "
            "FlashInfer release wheels, or tagged full releases without an "
            "image push."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_manifest(args.gb10_output_json)
    print(f"GB10 release manifest written to {args.gb10_output_json}")
    if args.gb10_validate_release_inputs:
        errors = validate_manifest(manifest)
        if errors:
            print("GB10 release manifest validation failed:")
            for error in errors:
                print(f"- {error}")
            raise SystemExit(1)
        print("GB10 release manifest validation passed.")


if __name__ == "__main__":
    main()
