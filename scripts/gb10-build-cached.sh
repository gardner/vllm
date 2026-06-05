#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/gb10-build-cached.sh [preflight|wheel|runtime]

Build GB10 targets with persistent local BuildKit cache.

Defaults are intentionally conservative for local work:
  GB10_MAX_JOBS=1
  GB10_NVCC_THREADS=1
  GB10_NATIVE_CUDA_ARCHS_ONLY=1
  GB10_OUTPUT=load

Set GB10_DRY_RUN=1 to resolve settings, validate the manifest, print the local
build plan, and exit before creating cache dirs or touching Docker/Buildx.
Set GB10_USE_REGISTRY_CACHE=1 to also import/export GHCR build cache.
Set GB10_BUILDX_BUILDER to override the local buildx builder name.
Set GB10_LOCAL_DIST_DIR to override where local wheel builds extract artifacts.
Set GB10_LOCAL_RELEASE_MANIFEST_DIR to override the local manifest output dir.
Failed builds preserve any exported cache for the next retry.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gb10_bool_flag() {
    local name="$1"
    local value="$2"
    local normalized_value="${value,,}"
    case "$normalized_value" in
        1|true|yes|on)
            printf '1'
            ;;
        0|false|no|off)
            printf '0'
            ;;
        *)
            echo "Unsupported ${name}=${value}. Use 1, 0, true, false, yes, no, on, or off." >&2
            return 2
            ;;
    esac
}

gb10_cache_ref_error() {
    local name="$1"
    local value="$2"
    echo "GB10 local cached build cache ref ${name} must be a GHCR image ref with a Docker-compatible tag, got ${value}." >&2
}

gb10_validate_cache_ref() {
    local name="$1"
    local value="$2"
    local image_name
    local image_tag
    local repository
    local repository_part
    local -a repository_parts

    if [[ "$value" =~ [[:space:]] ]] \
        || [[ "$value" != ghcr.io/* ]] \
        || [[ "$value" == *@* ]] \
        || [[ "$value" != *:* ]]; then
        gb10_cache_ref_error "$name" "$value"
        return 2
    fi

    image_name="${value%:*}"
    image_tag="${value##*:}"
    repository="${image_name#ghcr.io/}"

    if [[ "$image_name" == "$value" ]] \
        || [[ "$image_name" == *:* ]] \
        || ! [[ "$image_tag" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
        gb10_cache_ref_error "$name" "$value"
        return 2
    fi

    IFS='/' read -r -a repository_parts <<< "$repository"
    if [ "${#repository_parts[@]}" -lt 2 ]; then
        gb10_cache_ref_error "$name" "$value"
        return 2
    fi
    for repository_part in "${repository_parts[@]}"; do
        if ! [[ "$repository_part" =~ ^[a-z0-9]+([._-]+[a-z0-9]+)*$ ]]; then
            gb10_cache_ref_error "$name" "$value"
            return 2
        fi
    done
}

gb10_validate_release_output_paths() {
    GB10_LOCAL_RELEASE_MANIFEST_DIR="$GB10_LOCAL_RELEASE_MANIFEST_DIR" \
    GB10_RELEASE_MANIFEST_JSON="$GB10_RELEASE_MANIFEST_JSON" \
    GB10_RELEASE_CHECKSUMS="$GB10_RELEASE_CHECKSUMS" \
    GB10_RUNTIME_IMAGE_REF="$GB10_RUNTIME_IMAGE_REF" \
    GB10_RUNTIME_IMAGE_DIGEST="$GB10_RUNTIME_IMAGE_DIGEST" \
    GB10_RUNTIME_IMAGE_METADATA_JSON="$GB10_RUNTIME_IMAGE_METADATA_JSON" \
        python3 - <<'PY'
import os
import sys
from pathlib import Path

manifest_dir = Path(os.environ["GB10_LOCAL_RELEASE_MANIFEST_DIR"]).resolve()
output_paths = {
    "GB10_RELEASE_MANIFEST_JSON": os.environ["GB10_RELEASE_MANIFEST_JSON"],
    "GB10_RELEASE_CHECKSUMS": os.environ["GB10_RELEASE_CHECKSUMS"],
    "GB10_RUNTIME_IMAGE_REF": os.environ["GB10_RUNTIME_IMAGE_REF"],
    "GB10_RUNTIME_IMAGE_DIGEST": os.environ["GB10_RUNTIME_IMAGE_DIGEST"],
    "GB10_RUNTIME_IMAGE_METADATA_JSON": os.environ[
        "GB10_RUNTIME_IMAGE_METADATA_JSON"
    ],
}

errors = []
if manifest_dir.exists() and not manifest_dir.is_dir():
    errors.append(
        "GB10_LOCAL_RELEASE_MANIFEST_DIR must be a directory path; "
        f"existing target is not a directory: {manifest_dir}."
    )

resolved_output_paths = {}
for name, raw_path in output_paths.items():
    if not raw_path:
        errors.append(f"GB10 generated release output path {name} must be non-empty.")
        continue
    output_path = Path(raw_path).resolve()
    try:
        relative_path = output_path.relative_to(manifest_dir)
    except ValueError:
        errors.append(
            f"GB10 generated release output path {name} must stay under "
            f"GB10_LOCAL_RELEASE_MANIFEST_DIR ({manifest_dir}), got {output_path}."
        )
        continue
    if not relative_path.parts:
        errors.append(
            f"GB10 generated release output path {name} must be a file under "
            f"GB10_LOCAL_RELEASE_MANIFEST_DIR ({manifest_dir}), got {output_path}."
        )
        continue
    if output_path.is_dir():
        errors.append(
            f"GB10 generated release output path {name} must be a file path; "
            f"existing target is a directory: {output_path}."
        )
        continue
    resolved_output_paths.setdefault(output_path, []).append(name)

for output_path, names in sorted(
    resolved_output_paths.items(),
    key=lambda item: str(item[0]),
):
    if len(names) > 1:
        errors.append(
            "GB10 generated release output paths must be unique; "
            f"{', '.join(names)} all resolve to {output_path}."
        )

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(2)
PY
}

gb10_remove_stale_release_outputs() {
    rm -f \
        "$GB10_RELEASE_MANIFEST_JSON" \
        "$GB10_RELEASE_CHECKSUMS" \
        "$GB10_RUNTIME_IMAGE_REF" \
        "$GB10_RUNTIME_IMAGE_DIGEST" \
        "$GB10_RUNTIME_IMAGE_METADATA_JSON"
}

GB10_PREFLIGHT_CACHE_REF="${GB10_PREFLIGHT_CACHE_REF:-ghcr.io/gardner/vllm-gb10-buildcache:preflight}"
GB10_WHEEL_CACHE_REF="${GB10_WHEEL_CACHE_REF:-ghcr.io/gardner/vllm-gb10-buildcache:wheel}"
GB10_RUNTIME_CACHE_REF="${GB10_RUNTIME_CACHE_REF:-ghcr.io/gardner/vllm-gb10-buildcache:runtime}"
gb10_validate_cache_ref GB10_PREFLIGHT_CACHE_REF "$GB10_PREFLIGHT_CACHE_REF"
gb10_validate_cache_ref GB10_WHEEL_CACHE_REF "$GB10_WHEEL_CACHE_REF"
gb10_validate_cache_ref GB10_RUNTIME_CACHE_REF "$GB10_RUNTIME_CACHE_REF"

target_arg="${1:-preflight}"
case "$target_arg" in
    -h|--help)
        usage
        exit 0
        ;;
    preflight)
        docker_target="gb10-flashinfer-preflight"
        cache_key="preflight"
        image_tags=(--tag "vllm-gb10-flashinfer-preflight:local")
        registry_cache_refs=("$GB10_PREFLIGHT_CACHE_REF")
        ;;
    wheel|build)
        docker_target="build"
        cache_key="wheel"
        image_tags=(--tag "vllm-gb10-wheel:local")
        registry_cache_refs=("$GB10_WHEEL_CACHE_REF")
        ;;
    runtime|image|vllm-openai)
        docker_target="vllm-openai"
        cache_key="runtime"
        image_tags=()
        registry_cache_refs=(
            "$GB10_WHEEL_CACHE_REF"
            "$GB10_RUNTIME_CACHE_REF"
        )
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

GB10_MAX_JOBS="${GB10_MAX_JOBS:-1}"
GB10_NVCC_THREADS="${GB10_NVCC_THREADS:-1}"
GB10_NATIVE_CUDA_ARCHS_ONLY="${GB10_NATIVE_CUDA_ARCHS_ONLY:-1}"
GB10_NATIVE_CUDA_ARCHS_ONLY_ENABLED="$(gb10_bool_flag GB10_NATIVE_CUDA_ARCHS_ONLY "$GB10_NATIVE_CUDA_ARCHS_ONLY")"
if [ "$GB10_NATIVE_CUDA_ARCHS_ONLY_ENABLED" != "1" ]; then
    echo "GB10_NATIVE_CUDA_ARCHS_ONLY must be enabled for GB10 native builds; got $GB10_NATIVE_CUDA_ARCHS_ONLY." >&2
    exit 2
fi
GB10_NATIVE_CUDA_ARCHS_ONLY="$GB10_NATIVE_CUDA_ARCHS_ONLY_ENABLED"
GB10_DRY_RUN="${GB10_DRY_RUN:-0}"
GB10_DRY_RUN_ENABLED="$(gb10_bool_flag GB10_DRY_RUN "$GB10_DRY_RUN")"
GB10_USE_REGISTRY_CACHE="${GB10_USE_REGISTRY_CACHE:-0}"
GB10_USE_REGISTRY_CACHE_ENABLED="$(gb10_bool_flag GB10_USE_REGISTRY_CACHE "$GB10_USE_REGISTRY_CACHE")"
GB10_BUILDX_BUILDER="${GB10_BUILDX_BUILDER:-gb10-builder}"
GB10_LOCAL_DIST_DIR="${GB10_LOCAL_DIST_DIR:-$repo_root/dist}"
GB10_LOCAL_RELEASE_MANIFEST_DIR="${GB10_LOCAL_RELEASE_MANIFEST_DIR:-$repo_root/gb10-release-manifest-local}"
GB10_RUNTIME_IMAGE_METADATA_JSON="${GB10_RUNTIME_IMAGE_METADATA_JSON:-$GB10_LOCAL_RELEASE_MANIFEST_DIR/buildx-runtime-image-metadata.json}"
GB10_RELEASE_MANIFEST_JSON="$GB10_LOCAL_RELEASE_MANIFEST_DIR/gb10-release-manifest.json"
GB10_RELEASE_CHECKSUMS="$GB10_LOCAL_RELEASE_MANIFEST_DIR/gb10-vllm-release-SHA256SUMS"
GB10_RUNTIME_IMAGE_REF="$GB10_LOCAL_RELEASE_MANIFEST_DIR/gb10-runtime-image-ref.txt"
GB10_RUNTIME_IMAGE_DIGEST="$GB10_LOCAL_RELEASE_MANIFEST_DIR/gb10-runtime-image-digest.txt"
GITHUB_SHA="${GITHUB_SHA:-$(git rev-parse HEAD)}"
git_branch="$(git symbolic-ref -q --short HEAD || true)"
GITHUB_EVENT_NAME="${GITHUB_EVENT_NAME:-workflow_dispatch}"
GITHUB_REF="${GITHUB_REF:-refs/heads/${git_branch:-local}}"
GB10_SELF_HOSTED_RUNNER_LABELS="${GB10_SELF_HOSTED_RUNNER_LABELS:-[\"self-hosted\",\"linux\",\"aarch64\",\"cuda13\",\"dgx-spark\",\"sm121\"]}"

output_mode="${GB10_OUTPUT:-load}"

case "$output_mode" in
    load)
        output_args=(--load)
        ;;
    push)
        output_args=(--push)
        ;;
    cacheonly)
        output_args=(--output type=cacheonly)
        ;;
    *)
        echo "Unsupported GB10_OUTPUT=${GB10_OUTPUT}. Use load, push, or cacheonly." >&2
        exit 2
        ;;
esac
if [ "$output_mode" = "push" ] && [ "$docker_target" != "vllm-openai" ]; then
    echo "GB10_OUTPUT=push is only supported for runtime image builds." >&2
    echo "Target '$target_arg' resolves to local stage '$docker_target'." >&2
    exit 2
fi

if [ "$docker_target" = "gb10-flashinfer-preflight" ]; then
    gb10_preflight_only_default="true"
else
    gb10_preflight_only_default="false"
fi
if [ "$output_mode" = "push" ]; then
    gb10_push_image_default="true"
else
    gb10_push_image_default="false"
fi

export GITHUB_EVENT_NAME GITHUB_REF GITHUB_SHA
export GB10_INPUT_FLASH_ATTN_REF="${GB10_FLASH_ATTN_REF:-}"
export GB10_INPUT_FLASH_ATTN_REPO="${GB10_FLASH_ATTN_REPO:-}"
export GB10_INPUT_IMAGE_NAME="${GB10_IMAGE_NAME:-vllm-gb10}"
if [ -n "${GB10_IMAGE_TAG:-}" ]; then
    export GB10_INPUT_IMAGE_TAG="$GB10_IMAGE_TAG"
fi
export GB10_INPUT_MAX_JOBS="$GB10_MAX_JOBS"
export GB10_INPUT_NVCC_THREADS="$GB10_NVCC_THREADS"
export GB10_INPUT_PREFLIGHT_ONLY="${GB10_PREFLIGHT_ONLY:-$gb10_preflight_only_default}"
export GB10_INPUT_PREBUILT_WHEEL_URLS="${GB10_PREBUILT_WHEEL_URLS:-}"
export GB10_INPUT_PUSH_IMAGE="${GB10_PUSH_IMAGE:-$gb10_push_image_default}"
export GB10_INPUT_RELEASE_TAG="${GB10_RELEASE_TAG:-}"
export GB10_INPUT_RUNNER_LABELS="${GB10_RUNNER_LABELS:-$GB10_SELF_HOSTED_RUNNER_LABELS}"

gb10_validate_release_output_paths
gb10_remove_stale_release_outputs

set +e
resolved_settings="$(python3 scripts/gb10-resolve-release-settings.py --gb10-output-shell 2>&1)"
resolve_status=$?
set -e
if [ "$resolve_status" -ne 0 ]; then
    printf '%s\n' "$resolved_settings" >&2
    exit "$resolve_status"
fi
eval "$resolved_settings"

export GB10_NATIVE_CUDA_ARCHS_ONLY
export GB10_PREFLIGHT_CACHE_REF GB10_RUNTIME_CACHE_REF GB10_WHEEL_CACHE_REF

if [ "$docker_target" = "vllm-openai" ]; then
    image_tags=(--tag "${GB10_IMAGE_NAME}:${GB10_IMAGE_TAG}")
fi
metadata_args=()
if [ "$docker_target" = "vllm-openai" ]; then
    metadata_args=(--metadata-file "$GB10_RUNTIME_IMAGE_METADATA_JSON")
fi

VLLM_BUILD_COMMIT="${VLLM_BUILD_COMMIT:-$GITHUB_SHA}"
VLLM_BUILD_PIPELINE="${VLLM_BUILD_PIPELINE:-${GITHUB_WORKFLOW:-GB10 local cached build}}"
if [ -z "${VLLM_BUILD_URL:-}" ]; then
    if [ -n "${GITHUB_RUN_ID:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
        GITHUB_SERVER_URL="${GITHUB_SERVER_URL:-https://github.com}"
        VLLM_BUILD_URL="${GITHUB_SERVER_URL%/}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
    else
        VLLM_BUILD_URL=""
    fi
fi
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-$GB10_IMAGE_TAG}"
export VLLM_BUILD_COMMIT VLLM_BUILD_PIPELINE VLLM_BUILD_URL VLLM_IMAGE_TAG

mkdir -p "$GB10_LOCAL_RELEASE_MANIFEST_DIR"
scripts/gb10-write-release-manifest.py \
    --gb10-validate-release-inputs \
    --gb10-output-json "$GB10_RELEASE_MANIFEST_JSON"

cache_root="${GB10_LOCAL_CACHE_DIR:-$repo_root/.buildx-cache/gb10}"
cache_dir="$cache_root/$cache_key"
cache_next="$cache_root/${cache_key}.next"
cache_failed="$cache_root/${cache_key}.failed"
registry_cache_refs_string="${registry_cache_refs[*]}"

if [ "$GB10_DRY_RUN_ENABLED" = "1" ]; then
    source_dependency_settings="$(python3 - "$GB10_RELEASE_MANIFEST_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)

source_dependencies = manifest["dependencies"]["source_dependencies"]
for dependency_name, variable_prefix in (
    ("deepgemm", "DEEPGEMM"),
    ("flashmla", "FLASH_MLA"),
    ("triton_kernels", "TRITON_KERNELS"),
):
    dependency = source_dependencies[dependency_name]
    print(f"{variable_prefix}_GIT_REPOSITORY={dependency['repository']}")
    print(f"{variable_prefix}_GIT_TAG={dependency['ref']}")
PY
)"
    echo "GB10 local cached build dry run"
    echo "target_arg=$target_arg"
    echo "docker_target=$docker_target"
    echo "cache_key=$cache_key"
    echo "cache_root=$cache_root"
    echo "cache_dir=$cache_dir"
    echo "cache_next=$cache_next"
    echo "cache_failed=$cache_failed"
    echo "registry_cache_refs=$registry_cache_refs_string"
    echo "registry_cache_enabled=$GB10_USE_REGISTRY_CACHE_ENABLED"
    echo "output_mode=$output_mode"
    echo "GB10_OUTPUT=$output_mode"
    echo "GB10_RELEASE_TAG=$GB10_RELEASE_TAG"
    echo "GB10_PREFLIGHT_ONLY=$GB10_PREFLIGHT_ONLY"
    echo "GB10_PUSH_IMAGE=$GB10_PUSH_IMAGE"
    echo "GB10_IMAGE_NAME=$GB10_IMAGE_NAME"
    echo "GB10_IMAGE_TAG=$GB10_IMAGE_TAG"
    echo "GB10_VLLM_VERSION=$GB10_VLLM_VERSION"
    echo "GB10_RELEASE_MANIFEST_JSON=$GB10_RELEASE_MANIFEST_JSON"
    echo "GB10_RELEASE_CHECKSUMS=$GB10_RELEASE_CHECKSUMS"
    echo "GB10_RUNTIME_IMAGE_REF=$GB10_RUNTIME_IMAGE_REF"
    echo "GB10_RUNTIME_IMAGE_DIGEST=$GB10_RUNTIME_IMAGE_DIGEST"
    echo "VLLM_BUILD_COMMIT=$VLLM_BUILD_COMMIT"
    echo "VLLM_BUILD_PIPELINE=$VLLM_BUILD_PIPELINE"
    echo "VLLM_BUILD_URL=$VLLM_BUILD_URL"
    echo "VLLM_IMAGE_TAG=$VLLM_IMAGE_TAG"
    echo "GB10_RUNTIME_IMAGE_METADATA_JSON=$GB10_RUNTIME_IMAGE_METADATA_JSON"
    echo "GB10_MAX_JOBS=$GB10_MAX_JOBS"
    echo "GB10_NVCC_THREADS=$GB10_NVCC_THREADS"
    echo "GB10_NATIVE_CUDA_ARCHS_ONLY=$GB10_NATIVE_CUDA_ARCHS_ONLY"
    echo "GB10_LOCAL_DIST_DIR=$GB10_LOCAL_DIST_DIR"
    echo "GB10_LOCAL_RELEASE_MANIFEST_DIR=$GB10_LOCAL_RELEASE_MANIFEST_DIR"
    echo "GB10_FLASH_ATTN_REPO=$GB10_FLASH_ATTN_REPO"
    echo "GB10_FLASH_ATTN_REF=$GB10_FLASH_ATTN_REF"
    echo "GB10_RUNNER_LABELS=$GB10_RUNNER_LABELS"
    printf '%s\n' "$source_dependency_settings"
    echo "GB10_PREBUILT_WHEEL_URLS=$GB10_PREBUILT_WHEEL_URLS"
    exit 0
fi

if [ "$docker_target" = "vllm-openai" ] && [ "$output_mode" != "cacheonly" ]; then
    scripts/gb10-validate-vllm-wheel-artifact.py \
        --gb10-dist-dir "$GB10_LOCAL_DIST_DIR" \
        --gb10-vllm-version "$GB10_VLLM_VERSION" \
        --gb10-context "GB10 local runtime build before Docker/Buildx starts" \
        --gb10-remediation "Run scripts/gb10-build-cached.sh wheel first, or set GB10_LOCAL_DIST_DIR to a directory with one current vLLM wheel."
fi
if [ "$docker_target" = "build" ] && [ "$output_mode" = "load" ]; then
    scripts/gb10-validate-vllm-wheel-artifact.py \
        --gb10-dist-dir "$GB10_LOCAL_DIST_DIR" \
        --gb10-vllm-version "$GB10_VLLM_VERSION" \
        --gb10-context "GB10 local wheel build output directory before Docker/Buildx starts" \
        --gb10-remediation "Remove stale vLLM wheels from GB10_LOCAL_DIST_DIR, or set GB10_LOCAL_DIST_DIR to a clean directory." \
        --gb10-allow-empty
fi

mkdir -p "$cache_root"

cache_args=()
if [ -f "$cache_dir/index.json" ]; then
    cache_args+=(--cache-from "type=local,src=$cache_dir")
fi
if [ -f "$cache_failed/index.json" ]; then
    cache_args+=(--cache-from "type=local,src=$cache_failed")
fi
cache_args+=(--cache-to "type=local,dest=$cache_next,mode=max")

if [ "$GB10_USE_REGISTRY_CACHE_ENABLED" = "1" ]; then
    for ref in "${registry_cache_refs[@]}"; do
        cache_args+=(--cache-from "type=registry,ref=$ref")
    done
    cache_args+=(--cache-to "type=registry,ref=${registry_cache_refs[-1]},mode=max")
fi

rm -rf "$cache_next"

if ! docker buildx inspect "$GB10_BUILDX_BUILDER" >/dev/null 2>&1; then
    docker buildx create \
        --name "$GB10_BUILDX_BUILDER" \
        --driver docker-container \
        --use >/dev/null
fi
docker buildx inspect "$GB10_BUILDX_BUILDER" --bootstrap >/dev/null

set +e
scripts/gb10-run-with-heartbeat.sh "local ${cache_key} build" \
    docker buildx build \
    --builder "$GB10_BUILDX_BUILDER" \
    --file docker/Dockerfile \
    --target "$docker_target" \
    --platform linux/arm64 \
    "${output_args[@]}" \
    "${image_tags[@]}" \
    "${cache_args[@]}" \
    "${metadata_args[@]}" \
    --build-arg "max_jobs=$GB10_MAX_JOBS" \
    --build-arg "nvcc_threads=$GB10_NVCC_THREADS" \
    --build-arg "vllm_native_cuda_archs_only=$GB10_NATIVE_CUDA_ARCHS_ONLY" \
    --build-arg "gb10_prebuilt_wheel_urls=$GB10_PREBUILT_WHEEL_URLS" \
    --build-arg gb10_require_flashinfer_wheels=true \
    --build-arg "vllm_version_override=$GB10_VLLM_VERSION" \
    --build-arg "vllm_flash_attn_git_repository=$GB10_FLASH_ATTN_REPO" \
    --build-arg "vllm_flash_attn_git_tag=$GB10_FLASH_ATTN_REF" \
    --build-arg "VLLM_BUILD_COMMIT=$VLLM_BUILD_COMMIT" \
    --build-arg "VLLM_BUILD_PIPELINE=$VLLM_BUILD_PIPELINE" \
    --build-arg "VLLM_BUILD_URL=$VLLM_BUILD_URL" \
    --build-arg "VLLM_IMAGE_TAG=$VLLM_IMAGE_TAG" \
    .
build_status=$?
set -e

if [ -f "$cache_next/index.json" ]; then
    if [ "$build_status" -eq 0 ]; then
        rm -rf "$cache_failed" "$cache_dir"
        mv "$cache_next" "$cache_dir"
    else
        rm -rf "$cache_failed"
        mv "$cache_next" "$cache_failed"
        echo "GB10 BuildKit cache export from failed build preserved at $cache_failed." >&2
        echo "The next retry will import both the last successful cache and failed-build cache." >&2
    fi
else
    rm -rf "$cache_next"
fi

if [ "$build_status" -ne 0 ]; then
    exit "$build_status"
fi

if [ "$docker_target" = "build" ] && [ "$output_mode" = "load" ]; then
    mkdir -p "$GB10_LOCAL_DIST_DIR"
    wheel_container="$(docker create vllm-gb10-wheel:local)"
    docker cp "$wheel_container:/workspace/dist/." "$GB10_LOCAL_DIST_DIR/"
    docker rm "$wheel_container"
    scripts/gb10-validate-vllm-wheel-artifact.py \
        --gb10-dist-dir "$GB10_LOCAL_DIST_DIR" \
        --gb10-vllm-version "$GB10_VLLM_VERSION" \
        --gb10-context "GB10 local wheel build after artifact extraction"
    ls -lh "$GB10_LOCAL_DIST_DIR"
fi

if [ "$docker_target" = "vllm-openai" ] && [ "$output_mode" != "cacheonly" ]; then
    scripts/gb10-write-runtime-image-provenance.py \
        --gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON" \
        --gb10-release-manifest-dir "$GB10_LOCAL_RELEASE_MANIFEST_DIR" \
        --gb10-image-name "$GB10_IMAGE_NAME" \
        --gb10-image-tag "$GB10_IMAGE_TAG" \
        --gb10-push-image "$GB10_PUSH_IMAGE"
    scripts/gb10-write-vllm-release-checksums.py \
        --gb10-dist-dir "$GB10_LOCAL_DIST_DIR" \
        --gb10-release-manifest-dir "$GB10_LOCAL_RELEASE_MANIFEST_DIR" \
        --gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON"
    cat "$GB10_RELEASE_CHECKSUMS"
    if [ -n "$GB10_RELEASE_TAG" ]; then
        scripts/gb10-validate-vllm-release-assets.py \
            --gb10-dist-dir "$GB10_LOCAL_DIST_DIR" \
            --gb10-release-manifest-dir "$GB10_LOCAL_RELEASE_MANIFEST_DIR" \
            --gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON"
    fi
fi
