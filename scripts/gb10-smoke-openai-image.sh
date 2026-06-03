#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/gb10-smoke-openai-image.sh IMAGE [--serve SERVE_ARGS...] [--smoke SMOKE_ARGS...]

Run a GB10 OpenAI-compatible server smoke test against an already built vLLM
runtime image. The wrapper starts a temporary container on an isolated host
port, waits for /v1/models, runs scripts/gb10-smoke-openai-server.py from the
host, writes a durable JSON report, and removes the container on exit.

This is intended for local GB10 hardware or a self-hosted Spark runner, not
GitHub-hosted ARM runners.

Required:
  IMAGE                    vLLM runtime image to test.
  SERVE_ARGS               Arguments passed to the image entrypoint
                           `vllm serve`; include the model id/path here.

Useful environment:
  GB10_OPENAI_IMAGE_HOST_PORT       Host port for the temporary server
                                    (default: 18000)
  GB10_OPENAI_IMAGE_CONTAINER_PORT  Container vLLM port (default: 8000)
  GB10_OPENAI_IMAGE_NAME            Container name
                                    (default: gb10-openai-smoke-$$)
  GB10_OPENAI_IMAGE_CACHE_DIR       Host cache dir mounted as /root/.cache
                                    (default: ~/.cache)
  GB10_OPENAI_IMAGE_REPORT_DIR      Host report dir
                                    (default: ./gb10-smoke-reports)
  GB10_OPENAI_IMAGE_SHM_SIZE        Docker --shm-size value (default: 16g)
  GB10_OPENAI_IMAGE_IPC             Docker --ipc value (default: host)
  GB10_OPENAI_IMAGE_WAIT_SECONDS    Startup timeout in seconds (default: 900)
  GB10_OPENAI_IMAGE_WAIT_INTERVAL   Startup poll interval in seconds (default: 5)
  GB10_OPENAI_IMAGE_ENV_FILE        Optional Docker --env-file
  GB10_OPENAI_IMAGE_EXTRA_DOCKER_ARGS
                                    Extra docker arguments, split by the shell.
  GB10_OPENAI_IMAGE_KEEP_CONTAINER  Set to 1 to leave the container running.

Default smoke report:
  ${GB10_OPENAI_IMAGE_REPORT_DIR:-./gb10-smoke-reports}/gb10-openai-server-smoke-image.json

Examples:
  scripts/gb10-smoke-openai-image.sh ghcr.io/gardner/vllm-gb10:tag \
    --serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
      --served-model-name qwen3.6 --trust-remote-code \
      --quantization modelopt --attention-backend flashinfer \
      --kv-cache-dtype fp8 --max-model-len 4096 \
    --smoke --model qwen3.6 --gb10-endpoint chat \
      --gb10-repeat-count 2 --gb10-require-deterministic
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
smoke_script="$repo_root/scripts/gb10-smoke-openai-server.py"

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

serve_args=()
smoke_args=()
mode=serve

if [ "${1:-}" = "--serve" ]; then
    shift
fi

for arg in "$@"; do
    case "$arg" in
        --serve)
            mode=serve
            ;;
        --smoke)
            mode=smoke
            ;;
        *)
            if [ "$mode" = "serve" ]; then
                serve_args+=("$arg")
            else
                smoke_args+=("$arg")
            fi
            ;;
    esac
done

if [ "${#serve_args[@]}" -eq 0 ]; then
    echo "SERVE_ARGS are required; pass the model and vLLM serve flags after --serve." >&2
    usage >&2
    exit 2
fi

if [ ! -f "$smoke_script" ]; then
    echo "Missing OpenAI server smoke harness: $smoke_script" >&2
    exit 1
fi

command -v docker >/dev/null || {
    echo "docker is required to run the GB10 OpenAI image smoke." >&2
    exit 1
}
command -v curl >/dev/null || {
    echo "curl is required to wait for the temporary vLLM server." >&2
    exit 1
}
command -v python3 >/dev/null || {
    echo "python3 is required to run the OpenAI server smoke harness." >&2
    exit 1
}

has_arg() {
    local flag="$1"
    shift
    for arg in "$@"; do
        case "$arg" in
            "$flag"|"$flag"=*)
                return 0
                ;;
        esac
    done
    return 1
}

host_port="${GB10_OPENAI_IMAGE_HOST_PORT:-18000}"
container_port="${GB10_OPENAI_IMAGE_CONTAINER_PORT:-8000}"
container_name="${GB10_OPENAI_IMAGE_NAME:-gb10-openai-smoke-$$}"
cache_dir="${GB10_OPENAI_IMAGE_CACHE_DIR:-$HOME/.cache}"
report_dir="${GB10_OPENAI_IMAGE_REPORT_DIR:-$PWD/gb10-smoke-reports}"
wait_seconds="${GB10_OPENAI_IMAGE_WAIT_SECONDS:-900}"
wait_interval="${GB10_OPENAI_IMAGE_WAIT_INTERVAL:-5}"
base_url="http://127.0.0.1:${host_port}"
default_report_path="$report_dir/gb10-openai-server-smoke-image.json"

mkdir -p "$cache_dir"
mkdir -p "$report_dir"

if ! has_arg --host "${serve_args[@]}"; then
    serve_args+=(--host 0.0.0.0)
fi
if ! has_arg --port "${serve_args[@]}"; then
    serve_args+=(--port "$container_port")
fi

if ! has_arg --gb10-base-url "${smoke_args[@]}"; then
    smoke_args+=(--gb10-base-url "$base_url")
fi
if ! has_arg --gb10-report-json "${smoke_args[@]}"; then
    smoke_args+=(--gb10-report-json "$default_report_path")
fi
if ! has_arg --gb10-repeat-count "${smoke_args[@]}"; then
    smoke_args+=(--gb10-repeat-count "${GB10_OPENAI_IMAGE_REPEAT_COUNT:-2}")
fi
if ! has_arg --gb10-require-deterministic "${smoke_args[@]}" \
    && [ "${GB10_OPENAI_IMAGE_REQUIRE_DETERMINISTIC:-1}" = "1" ]; then
    smoke_args+=(--gb10-require-deterministic)
fi

docker_args=(
    run
    --detach
    --name "$container_name"
    --gpus all
    --ipc "${GB10_OPENAI_IMAGE_IPC:-host}"
    --shm-size "${GB10_OPENAI_IMAGE_SHM_SIZE:-16g}"
    --publish "127.0.0.1:${host_port}:${container_port}"
    -e VLLM_FAIL_ON_NVFP4_FALLBACK=1
    -e VLLM_NO_USAGE_STATS=1
    -e HF_HOME=/root/.cache/huggingface
    -e TRANSFORMERS_CACHE=/root/.cache/huggingface
    -v "$cache_dir:/root/.cache"
)

if [ -n "${HF_TOKEN:-}" ]; then
    docker_args+=(-e "HF_TOKEN=$HF_TOKEN")
fi

if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    docker_args+=(-e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN")
fi

if [ -n "${GB10_OPENAI_IMAGE_ENV_FILE:-}" ]; then
    docker_args+=(--env-file "$GB10_OPENAI_IMAGE_ENV_FILE")
fi

if [ -n "${GB10_OPENAI_IMAGE_EXTRA_DOCKER_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_docker_args=($GB10_OPENAI_IMAGE_EXTRA_DOCKER_ARGS)
    docker_args+=("${extra_docker_args[@]}")
fi

cleanup() {
    if [ "${GB10_OPENAI_IMAGE_KEEP_CONTAINER:-0}" != "1" ]; then
        docker rm -f "$container_name" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "Starting temporary GB10 OpenAI server container: $container_name" >&2
docker "${docker_args[@]}" "$image" "${serve_args[@]}" >/dev/null

deadline=$((SECONDS + wait_seconds))
models_url="${base_url}/v1/models"
while ! curl -fsS --max-time 5 "$models_url" >/dev/null; do
    running="$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)"
    if [ "$running" != "true" ]; then
        echo "Temporary vLLM server container exited before /v1/models was ready." >&2
        docker logs "$container_name" >&2 || true
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "Timed out waiting for $models_url." >&2
        docker logs "$container_name" >&2 || true
        exit 1
    fi
    sleep "$wait_interval"
done

echo "Temporary GB10 OpenAI server is ready at $base_url" >&2
python3 "$smoke_script" "${smoke_args[@]}"
