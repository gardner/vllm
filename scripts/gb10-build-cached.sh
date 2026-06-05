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
Set GB10_LOCAL_RELEASE_MANIFEST_DIR to override the local manifest output dir.
Failed builds preserve any exported cache for the next retry.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

GB10_PREFLIGHT_CACHE_REF="${GB10_PREFLIGHT_CACHE_REF:-ghcr.io/gardner/vllm-gb10-buildcache:preflight}"
GB10_WHEEL_CACHE_REF="${GB10_WHEEL_CACHE_REF:-ghcr.io/gardner/vllm-gb10-buildcache:wheel}"
GB10_RUNTIME_CACHE_REF="${GB10_RUNTIME_CACHE_REF:-ghcr.io/gardner/vllm-gb10-buildcache:runtime}"

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
GB10_DRY_RUN="${GB10_DRY_RUN:-0}"
GB10_USE_REGISTRY_CACHE="${GB10_USE_REGISTRY_CACHE:-0}"
GB10_BUILDX_BUILDER="${GB10_BUILDX_BUILDER:-gb10-builder}"
GB10_LOCAL_RELEASE_MANIFEST_DIR="${GB10_LOCAL_RELEASE_MANIFEST_DIR:-$repo_root/gb10-release-manifest-local}"
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

mkdir -p "$GB10_LOCAL_RELEASE_MANIFEST_DIR"
scripts/gb10-write-release-manifest.py \
    --gb10-validate-release-inputs \
    --gb10-output-json "$GB10_LOCAL_RELEASE_MANIFEST_DIR/gb10-release-manifest.json"

if [[ "$GB10_DRY_RUN" =~ ^(1|true|yes|on)$ ]]; then
    echo "GB10 local cached build dry run"
    echo "target_arg=$target_arg"
    echo "docker_target=$docker_target"
    echo "cache_key=$cache_key"
    echo "output_mode=$output_mode"
    echo "GB10_PREFLIGHT_ONLY=$GB10_PREFLIGHT_ONLY"
    echo "GB10_PUSH_IMAGE=$GB10_PUSH_IMAGE"
    echo "GB10_IMAGE_NAME=$GB10_IMAGE_NAME"
    echo "GB10_IMAGE_TAG=$GB10_IMAGE_TAG"
    echo "GB10_VLLM_VERSION=$GB10_VLLM_VERSION"
    echo "GB10_MAX_JOBS=$GB10_MAX_JOBS"
    echo "GB10_NVCC_THREADS=$GB10_NVCC_THREADS"
    echo "GB10_NATIVE_CUDA_ARCHS_ONLY=$GB10_NATIVE_CUDA_ARCHS_ONLY"
    echo "GB10_LOCAL_RELEASE_MANIFEST_DIR=$GB10_LOCAL_RELEASE_MANIFEST_DIR"
    echo "GB10_PREBUILT_WHEEL_URLS=$GB10_PREBUILT_WHEEL_URLS"
    exit 0
fi

cache_root="${GB10_LOCAL_CACHE_DIR:-$repo_root/.buildx-cache/gb10}"
cache_dir="$cache_root/$cache_key"
cache_next="$cache_root/${cache_key}.next"
cache_failed="$cache_root/${cache_key}.failed"
mkdir -p "$cache_root"

cache_args=()
if [ -f "$cache_dir/index.json" ]; then
    cache_args+=(--cache-from "type=local,src=$cache_dir")
fi
if [ -f "$cache_failed/index.json" ]; then
    cache_args+=(--cache-from "type=local,src=$cache_failed")
fi
cache_args+=(--cache-to "type=local,dest=$cache_next,mode=max")

if [ "$GB10_USE_REGISTRY_CACHE" = "1" ]; then
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
    --build-arg "max_jobs=$GB10_MAX_JOBS" \
    --build-arg "nvcc_threads=$GB10_NVCC_THREADS" \
    --build-arg "vllm_native_cuda_archs_only=$GB10_NATIVE_CUDA_ARCHS_ONLY" \
    --build-arg "gb10_prebuilt_wheel_urls=$GB10_PREBUILT_WHEEL_URLS" \
    --build-arg gb10_require_flashinfer_wheels=true \
    --build-arg "vllm_version_override=$GB10_VLLM_VERSION" \
    --build-arg "vllm_flash_attn_git_repository=$GB10_FLASH_ATTN_REPO" \
    --build-arg "vllm_flash_attn_git_tag=$GB10_FLASH_ATTN_REF" \
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
