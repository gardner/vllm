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
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

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
        registry_cache_refs=("ghcr.io/gardner/vllm-gb10-buildcache:preflight")
        ;;
    wheel|build)
        docker_target="build"
        cache_key="wheel"
        image_tags=(--tag "vllm-gb10-wheel:local")
        registry_cache_refs=("ghcr.io/gardner/vllm-gb10-buildcache:wheel")
        ;;
    runtime|image|vllm-openai)
        docker_target="vllm-openai"
        cache_key="runtime"
        image_name="${GB10_IMAGE_NAME:-vllm-gb10}"
        image_tag="${GB10_IMAGE_TAG:-local}"
        image_tags=(--tag "${image_name}:${image_tag}")
        registry_cache_refs=(
            "ghcr.io/gardner/vllm-gb10-buildcache:wheel"
            "ghcr.io/gardner/vllm-gb10-buildcache:runtime"
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

case "${GB10_OUTPUT:-load}" in
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

cache_root="${GB10_LOCAL_CACHE_DIR:-$repo_root/.buildx-cache/gb10}"
cache_dir="$cache_root/$cache_key"
cache_next="$cache_root/${cache_key}.next"
mkdir -p "$cache_root"

cache_args=()
if [ -f "$cache_dir/index.json" ]; then
    cache_args+=(--cache-from "type=local,src=$cache_dir")
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

rm -rf "$cache_dir"
mv "$cache_next" "$cache_dir"
