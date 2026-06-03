#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/gb10-smoke-release-image.sh IMAGE [--offline OFFLINE_ARGS...] --serve SERVE_ARGS... [--openai OPENAI_ARGS...] [--verify VERIFY_ARGS...]

Run the complete GB10 final-image smoke sequence for a candidate vLLM runtime
image:

  1. scripts/gb10-smoke-image.sh for offline NVFP4 backend-selection evidence.
  2. scripts/gb10-smoke-openai-image.sh for OpenAI-compatible server evidence.
  3. scripts/gb10-verify-release-evidence.py as the final report gate.
  4. scripts/gb10-bundle-release-evidence.py for checksummed artifacts.

This script is intended for local GB10 hardware or a self-hosted Spark runner.
It starts containers only when invoked; it does not touch any already-running
server.

Required:
  IMAGE                    vLLM runtime image to test.
  SERVE_ARGS               Arguments passed to the image entrypoint
                           `vllm serve`; include the model id/path here.
  GB10_NVFP4_MODEL         Model id/path for the offline smoke, unless
                           --model is included in OFFLINE_ARGS.

Useful environment:
  GB10_RELEASE_SMOKE_REPORT_DIR       Host report dir
                                      (default: ./gb10-smoke-reports)
  GB10_RELEASE_REQUIRE_MOE            Set to 0 to skip required MoE evidence
                                      in the final verifier (default: 1)
  GB10_RELEASE_REQUIRE_OPENAI_DETERMINISTIC
                                      Set to 0 to skip deterministic OpenAI
                                      evidence in the final verifier
                                      (default: 1)
  GB10_RELEASE_EVIDENCE_OUTPUT_DIR    Directory for the checksummed evidence
                                      bundle (default:
                                      ./dist/gb10-release-evidence)
  GB10_RELEASE_BUNDLE_ALLOW_PARTIAL   Set to 0 to make bundling require all
                                      final-image reports even after verifier
                                      failure (default: 1)
  GB10_RELEASE_MANIFEST_JSON          Optional release manifest path to verify
                                      and bundle.
  GB10_RUNTIME_IMAGE_METADATA_JSON    Optional BuildKit runtime-image metadata
                                      path to bundle.

Argument sections:
  --offline OFFLINE_ARGS...  Extra args for scripts/gb10-smoke-image.sh after --.
  --serve SERVE_ARGS...      Args passed through to vLLM serve by the OpenAI
                             image wrapper.
  --openai OPENAI_ARGS...    Extra smoke args for scripts/gb10-smoke-openai-image.sh
                             after --smoke.
  --verify VERIFY_ARGS...    Extra args for scripts/gb10-verify-release-evidence.py.

Default reports:
  ${GB10_RELEASE_SMOKE_REPORT_DIR:-./gb10-smoke-reports}/gb10-nvfp4-smoke.json
  ${GB10_RELEASE_SMOKE_REPORT_DIR:-./gb10-smoke-reports}/gb10-openai-server-smoke-image.json
  ${GB10_RELEASE_SMOKE_REPORT_DIR:-./gb10-smoke-reports}/gb10-release-evidence-image.json
  ${GB10_RELEASE_EVIDENCE_OUTPUT_DIR:-./dist/gb10-release-evidence}/gb10-release-evidence.tar.gz

Example:
  GB10_NVFP4_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4 \
    scripts/gb10-smoke-release-image.sh ghcr.io/gardner/vllm-gb10:tag \
      --offline --trust-remote-code --max-model-len 4096 \
        --gb10-require-path linear --gb10-require-path moe \
        --gb10-expect-backend linear=FlashInferB12x \
        --gb10-expect-backend moe=FLASHINFER_B12X \
      --serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
        --served-model-name qwen3.6 --trust-remote-code \
        --quantization modelopt --attention-backend flashinfer \
        --kv-cache-dtype fp8 --max-model-len 4096 \
      --openai --model qwen3.6 --gb10-endpoint chat
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
offline_wrapper="$repo_root/scripts/gb10-smoke-image.sh"
openai_wrapper="$repo_root/scripts/gb10-smoke-openai-image.sh"
verifier="$repo_root/scripts/gb10-verify-release-evidence.py"
bundler="$repo_root/scripts/gb10-bundle-release-evidence.py"

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

offline_args=()
serve_args=()
openai_args=()
verify_args=()
mode=offline

for arg in "$@"; do
    case "$arg" in
        --offline)
            mode=offline
            ;;
        --serve)
            mode=serve
            ;;
        --openai)
            mode=openai
            ;;
        --verify)
            mode=verify
            ;;
        *)
            case "$mode" in
                offline)
                    offline_args+=("$arg")
                    ;;
                serve)
                    serve_args+=("$arg")
                    ;;
                openai)
                    openai_args+=("$arg")
                    ;;
                verify)
                    verify_args+=("$arg")
                    ;;
            esac
            ;;
    esac
done

if [ "${#serve_args[@]}" -eq 0 ]; then
    echo "SERVE_ARGS are required; pass model and vLLM serve flags after --serve." >&2
    usage >&2
    exit 2
fi

for script in "$offline_wrapper" "$openai_wrapper" "$verifier" "$bundler"; do
    if [ ! -f "$script" ]; then
        echo "Missing required GB10 smoke helper: $script" >&2
        exit 1
    fi
done

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

report_dir="${GB10_RELEASE_SMOKE_REPORT_DIR:-$PWD/gb10-smoke-reports}"
nvfp4_report="$report_dir/gb10-nvfp4-smoke.json"
openai_report="$report_dir/gb10-openai-server-smoke-image.json"
evidence_report="$report_dir/gb10-release-evidence-image.json"
mkdir -p "$report_dir"

if ! has_arg --gb10-report-json "${offline_args[@]}"; then
    offline_args+=(--gb10-report-json /gb10-smoke-reports/gb10-nvfp4-smoke.json)
fi

if ! has_arg --gb10-report-json "${openai_args[@]}"; then
    openai_args+=(--gb10-report-json "$openai_report")
fi

if ! has_arg --gb10-nvfp4-report-json "${verify_args[@]}"; then
    verify_args+=(--gb10-nvfp4-report-json "$nvfp4_report")
fi
if ! has_arg --gb10-openai-report-json "${verify_args[@]}"; then
    verify_args+=(--gb10-openai-report-json "$openai_report")
fi
if ! has_arg --gb10-image-ref "${verify_args[@]}"; then
    verify_args+=(--gb10-image-ref "$image")
fi
if ! has_arg --gb10-output-json "${verify_args[@]}"; then
    verify_args+=(--gb10-output-json "$evidence_report")
fi
if [ -n "${GB10_RELEASE_MANIFEST_JSON:-}" ] \
    && ! has_arg --gb10-release-manifest-json "${verify_args[@]}"; then
    verify_args+=(--gb10-release-manifest-json "$GB10_RELEASE_MANIFEST_JSON")
fi
if ! has_arg --gb10-require-moe "${verify_args[@]}" \
    && [ "${GB10_RELEASE_REQUIRE_MOE:-1}" = "1" ]; then
    verify_args+=(--gb10-require-moe)
fi
if ! has_arg --gb10-require-openai-deterministic "${verify_args[@]}" \
    && [ "${GB10_RELEASE_REQUIRE_OPENAI_DETERMINISTIC:-1}" = "1" ]; then
    verify_args+=(--gb10-require-openai-deterministic)
fi

echo "Running GB10 offline NVFP4 image smoke..." >&2
GB10_SMOKE_REPORT_DIR="$report_dir" \
    "$offline_wrapper" "$image" -- "${offline_args[@]}"

echo "Running GB10 OpenAI-compatible image smoke..." >&2
GB10_OPENAI_IMAGE_REPORT_DIR="$report_dir" \
    "$openai_wrapper" "$image" --serve "${serve_args[@]}" --smoke "${openai_args[@]}"

echo "Verifying combined GB10 release evidence..." >&2
verify_status=0
"$verifier" "${verify_args[@]}" || verify_status=$?

echo "Bundling GB10 release evidence reports..." >&2
bundle_args=(
    --gb10-report-dir "$report_dir"
    --gb10-image-ref "$image"
)
if [ -n "${GB10_RELEASE_MANIFEST_JSON:-}" ]; then
    bundle_args+=(--gb10-release-manifest-json "$GB10_RELEASE_MANIFEST_JSON")
fi
if [ -n "${GB10_RUNTIME_IMAGE_METADATA_JSON:-}" ]; then
    bundle_args+=(--gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON")
fi
if [ "${GB10_RELEASE_BUNDLE_ALLOW_PARTIAL:-1}" = "1" ]; then
    bundle_args+=(--gb10-allow-partial)
fi
"$bundler" "${bundle_args[@]}"

exit "$verify_status"
