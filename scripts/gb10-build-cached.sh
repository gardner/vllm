#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/gb10-build-cached.sh [preflight|wheel|runtime]

Build GB10 targets with persistent local BuildKit cache.

Defaults are intentionally conservative for local work:
  GB10_MAX_JOBS=1
  GB10_NVCC_THREADS=1
  GB10_OUTPUT=load

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
        image_name="${GB10_IMAGE_NAME:-vllm-gb10}"
        image_tag="${GB10_IMAGE_TAG:-local}"
        image_tags=(--tag "${image_name}:${image_tag}")
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

GB10_DEFAULT_PREBUILT_WHEEL_URLS="${GB10_DEFAULT_PREBUILT_WHEEL_URLS:-https://github.com/gardner/flashinfer/releases/download/gb10-flashinfer-v0.6.12-1c80efb3/flashinfer_python-0.6.12+cu130gb10-py3-none-any.whl https://github.com/gardner/flashinfer/releases/download/gb10-flashinfer-v0.6.12-1c80efb3/flashinfer_cubin-0.6.12+cu130gb10-py3-none-any.whl https://github.com/gardner/flashinfer/releases/download/gb10-flashinfer-v0.6.12-1c80efb3/flashinfer_jit_cache-0.6.12+cu130gb10-cp39-abi3-manylinux_2_28_aarch64.whl}"
GB10_PREBUILT_WHEEL_URLS="${GB10_PREBUILT_WHEEL_URLS:-$GB10_DEFAULT_PREBUILT_WHEEL_URLS}"
GB10_FLASH_ATTN_REPO="${GB10_FLASH_ATTN_REPO:-https://github.com/gardner/vllm-flash-attention.git}"
GB10_FLASH_ATTN_REF="${GB10_FLASH_ATTN_REF:-de3849e75d07edd1c00aec02c92ec852ba757adc}"
GB10_VLLM_VERSION="${GB10_VLLM_VERSION:-0.22.1rc0+gb10.local}"
GB10_MAX_JOBS="${GB10_MAX_JOBS:-1}"
GB10_NVCC_THREADS="${GB10_NVCC_THREADS:-1}"
GB10_USE_REGISTRY_CACHE="${GB10_USE_REGISTRY_CACHE:-0}"
GB10_BUILDX_BUILDER="${GB10_BUILDX_BUILDER:-gb10-builder}"
GB10_LOCAL_RELEASE_MANIFEST_DIR="${GB10_LOCAL_RELEASE_MANIFEST_DIR:-$repo_root/gb10-release-manifest-local}"
GB10_IMAGE_NAME="${GB10_IMAGE_NAME:-vllm-gb10}"
GB10_IMAGE_TAG="${GB10_IMAGE_TAG:-local}"
GB10_RELEASE_TAG="${GB10_RELEASE_TAG:-}"
GITHUB_SHA="${GITHUB_SHA:-$(git rev-parse HEAD)}"

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
    GB10_PREFLIGHT_ONLY="${GB10_PREFLIGHT_ONLY:-true}"
else
    GB10_PREFLIGHT_ONLY="${GB10_PREFLIGHT_ONLY:-false}"
fi
if [ "$output_mode" = "push" ]; then
    GB10_PUSH_IMAGE="${GB10_PUSH_IMAGE:-true}"
else
    GB10_PUSH_IMAGE="${GB10_PUSH_IMAGE:-false}"
fi

export GITHUB_SHA
export GB10_FLASH_ATTN_REF GB10_FLASH_ATTN_REPO
export GB10_IMAGE_NAME GB10_IMAGE_TAG GB10_PUSH_IMAGE
export GB10_MAX_JOBS GB10_NVCC_THREADS
export GB10_PREFLIGHT_CACHE_REF GB10_PREFLIGHT_ONLY
export GB10_PREBUILT_WHEEL_URLS GB10_RELEASE_TAG GB10_VLLM_VERSION
export GB10_RUNTIME_CACHE_REF GB10_WHEEL_CACHE_REF

mkdir -p "$GB10_LOCAL_RELEASE_MANIFEST_DIR"
scripts/gb10-write-release-manifest.py \
    --gb10-validate-release-inputs \
    --gb10-output-json "$GB10_LOCAL_RELEASE_MANIFEST_DIR/gb10-release-manifest.json"

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
