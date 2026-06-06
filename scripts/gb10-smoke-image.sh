#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/gb10-smoke-image.sh IMAGE [-- SMOKE_ARGS...]

Run the GB10 NVFP4 model-smoke harness inside an already built vLLM image.
This is intended for local GB10 hardware or a self-hosted Spark runner, not
GitHub-hosted ARM runners.

Required:
  IMAGE                    vLLM runtime image to test.
  GB10_NVFP4_MODEL         Model id/path, unless --model is passed after --.

Useful environment:
  GB10_SMOKE_CACHE_DIR     Host cache dir mounted as /root/.cache (default: ~/.cache)
  GB10_SMOKE_REPORT_DIR    Host report dir mounted as /gb10-smoke-reports
                           (default: ./gb10-smoke-reports)
  GB10_GPU_MEMORY_UTILIZATION
                           Default vLLM gpu-memory-utilization for the smoke
                           harness (default: 0.88)
  GB10_SMOKE_SHM_SIZE      Docker --shm-size value (default: 16g)
  GB10_SMOKE_IPC           Docker --ipc value (default: host)
  GB10_SMOKE_ENV_FILE      Optional Docker --env-file
  GB10_SMOKE_EXTRA_DOCKER_ARGS
                           Extra docker arguments, split by the shell.

Examples:
  GB10_GPU_MEMORY_UTILIZATION=0.88 \
    GB10_NVFP4_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    scripts/gb10-smoke-image.sh ghcr.io/gardner/vllm-gb10:tag -- \
      --trust-remote-code --max-model-len 4096 \
      --gb10-require-path linear --gb10-require-path moe \
      --gb10-expect-backend linear=FlashInferB12x \
      --gb10-expect-backend moe=FLASHINFER_CUTLASS

The default JSON report is written to:
  ${GB10_SMOKE_REPORT_DIR:-./gb10-smoke-reports}/gb10-nvfp4-smoke.json
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
smoke_script="$repo_root/scripts/gb10-smoke-nvfp4.py"
release_contract_script="$repo_root/scripts/gb10_release_contract.py"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ $# -lt 1 ]; then
    usage >&2
    exit 2
fi

image="$1"
shift
if [ "${1:-}" = "--" ]; then
    shift
fi

if [ ! -f "$smoke_script" ]; then
    echo "Missing smoke harness: $smoke_script" >&2
    exit 1
fi
if [ ! -f "$release_contract_script" ]; then
    echo "Missing GB10 release contract helper: $release_contract_script" >&2
    exit 1
fi

has_model_arg=0
has_report_arg=0
for arg in "$@"; do
    case "$arg" in
        --model|--model=*)
            has_model_arg=1
            ;;
        --gb10-report-json|--gb10-report-json=*)
            has_report_arg=1
            ;;
    esac
done

if [ "$has_model_arg" = "0" ] && [ -z "${GB10_NVFP4_MODEL:-}" ]; then
    echo "GB10_NVFP4_MODEL is required unless --model is passed after --." >&2
    exit 2
fi

cache_dir="${GB10_SMOKE_CACHE_DIR:-$HOME/.cache}"
report_dir="${GB10_SMOKE_REPORT_DIR:-$PWD/gb10-smoke-reports}"
hf_cache_dir="$cache_dir/huggingface"
mkdir -p "$cache_dir"
mkdir -p "$report_dir"
mkdir -p "$hf_cache_dir"
hf_cache_dir="$(cd "$hf_cache_dir" && pwd -P)"

docker_args=(
    run
    --rm
    --entrypoint python3
    --gpus all
    --ipc "${GB10_SMOKE_IPC:-host}"
    --shm-size "${GB10_SMOKE_SHM_SIZE:-16g}"
    -e VLLM_FAIL_ON_NVFP4_FALLBACK=1
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0
    -e VLLM_NO_USAGE_STATS=1
    -e HF_HOME=/root/.cache
    -e TRANSFORMERS_CACHE=/root/.cache
    -v "$hf_cache_dir:/root/.cache"
    -v "$report_dir:/gb10-smoke-reports"
    -v "$smoke_script:/tmp/gb10-smoke-nvfp4.py:ro"
    -v "$release_contract_script:/tmp/gb10_release_contract.py:ro"
)

if [ -n "${GB10_NVFP4_MODEL:-}" ]; then
    docker_args+=(-e "GB10_NVFP4_MODEL=$GB10_NVFP4_MODEL")
fi

if [ -n "${GB10_GPU_MEMORY_UTILIZATION:-}" ]; then
    docker_args+=(-e "GB10_GPU_MEMORY_UTILIZATION=$GB10_GPU_MEMORY_UTILIZATION")
fi

if [ -n "${HF_TOKEN:-}" ]; then
    docker_args+=(-e "HF_TOKEN=$HF_TOKEN")
fi

if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    docker_args+=(-e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN")
fi

if [ -n "${GB10_SMOKE_ENV_FILE:-}" ]; then
    docker_args+=(--env-file "$GB10_SMOKE_ENV_FILE")
fi

if [ -n "${GB10_SMOKE_EXTRA_DOCKER_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_docker_args=($GB10_SMOKE_EXTRA_DOCKER_ARGS)
    docker_args+=("${extra_docker_args[@]}")
fi

if [ "$has_report_arg" = "0" ]; then
    set -- "$@" --gb10-report-json /gb10-smoke-reports/gb10-nvfp4-smoke.json
fi

exec docker "${docker_args[@]}" "$image" /tmp/gb10-smoke-nvfp4.py "$@"
