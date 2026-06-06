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
  GB10_RELEASE_EVIDENCE_BUNDLE_NAME   Base name for the generated evidence
                                      tarball and tarball checksum.
  GB10_RELEASE_BUNDLE_ALLOW_PARTIAL   Set to 0 to make bundling require all
                                      final-image reports even after verifier
                                      failure (default: 1)
  GB10_RELEASE_TAG                    Optional release tag to verify against
                                      the manifest and record in bundle metadata.
  GB10_RELEASE_MANIFEST_JSON          Optional release manifest path to verify
                                      and bundle.
  GB10_RUNTIME_IMAGE_METADATA_JSON    Optional BuildKit runtime-image metadata
                                      path to verify and bundle.
  GB10_IMAGE_DIGEST                   Optional immutable digest for IMAGE. If
                                      unset, the script records the first
                                      docker RepoDigest for IMAGE when
                                      available.
  GB10_RELEASE_ALLOW_EXISTING_VLLM_CONTAINERS
                                      Set to 1 to allow final-image smoke to
                                      start while another running Docker
                                      container appears to be a vLLM service.
                                      The default is 0, which fails fast to
                                      avoid loading a second large model on the
                                      same Spark.

Argument sections:
  --offline OFFLINE_ARGS...  Extra args for scripts/gb10-smoke-image.sh after --.
  --serve SERVE_ARGS...      Args passed through to vLLM serve by the OpenAI
                             image wrapper.
  --openai OPENAI_ARGS...    Extra smoke args for scripts/gb10-smoke-openai-image.sh
                             after --smoke.
  --verify VERIFY_ARGS...    Extra args for scripts/gb10-verify-release-evidence.py.

Default reports:
  Evidence report files listed by scripts/gb10-list-release-evidence-report-files.py.
  Evidence bundle assets listed by scripts/gb10-list-evidence-release-assets.py.

Example:
  GB10_GPU_MEMORY_UTILIZATION=0.88 \
  GB10_NVFP4_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    scripts/gb10-smoke-release-image.sh ghcr.io/gardner/vllm-gb10:tag \
      --offline --trust-remote-code --attention-backend flashinfer \
        --max-model-len 4096 \
        --gb10-require-path linear --gb10-require-path moe \
        --gb10-expect-backend linear=FlashInferB12x \
        --gb10-expect-backend moe=FLASHINFER_CUTLASS \
      --serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
        --served-model-name nemotron3.nano --trust-remote-code \
        --quantization modelopt --attention-backend flashinfer \
        --kv-cache-dtype fp8 --max-model-len 4096 \
      --openai --model nemotron3.nano --gb10-endpoint chat
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
offline_wrapper="$repo_root/scripts/gb10-smoke-image.sh"
openai_wrapper="$repo_root/scripts/gb10-smoke-openai-image.sh"
verifier="$repo_root/scripts/gb10-verify-release-evidence.py"
bundler="$repo_root/scripts/gb10-bundle-release-evidence.py"
report_file_lister="$repo_root/scripts/gb10-list-release-evidence-report-files.py"
asset_lister="$repo_root/scripts/gb10-list-evidence-release-assets.py"

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

for script in \
    "$offline_wrapper" \
    "$openai_wrapper" \
    "$verifier" \
    "$bundler" \
    "$report_file_lister" \
    "$asset_lister"
do
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
mkdir -p "$report_dir"
report_files_file="$(mktemp)"
trap 'rm -f "$report_files_file"' EXIT
"$report_file_lister" --gb10-report-dir "$report_dir" > "$report_files_file"
mapfile -t release_evidence_files < "$report_files_file"
if [ "${#release_evidence_files[@]}" -ne 4 ]; then
    echo "GB10 release evidence report lister returned ${#release_evidence_files[@]} files; expected 4." >&2
    exit 1
fi
nvfp4_report="${release_evidence_files[0]}"
openai_report="${release_evidence_files[1]}"
evidence_report="${release_evidence_files[2]}"
smoked_image_digest_report="${release_evidence_files[3]}"

bundle_available_release_evidence() {
    local bundle_status=0
    local -a partial_bundle_args=(
        --gb10-report-dir "$report_dir"
        --gb10-image-ref "$image"
        --gb10-allow-partial
    )
    if [ -n "${GB10_RELEASE_TAG:-}" ]; then
        partial_bundle_args+=(--gb10-release-tag "$GB10_RELEASE_TAG")
    fi
    if [ -n "${GB10_RELEASE_MANIFEST_JSON:-}" ]; then
        partial_bundle_args+=(--gb10-release-manifest-json "$GB10_RELEASE_MANIFEST_JSON")
    fi
    if [ -n "${GB10_RUNTIME_IMAGE_METADATA_JSON:-}" ]; then
        partial_bundle_args+=(--gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON")
    fi

    echo "Bundling available GB10 release evidence reports..." >&2
    "$bundler" "${partial_bundle_args[@]}" >&2 || bundle_status=$?
    if [ "$bundle_status" -eq 0 ]; then
        echo "GB10 release evidence assets:" >&2
        "$asset_lister" >&2 || true
    else
        echo "GB10 release evidence bundling failed with status $bundle_status." >&2
    fi
}

if { [ -n "${GB10_RELEASE_MANIFEST_JSON:-}" ] \
        && [ -z "${GB10_RUNTIME_IMAGE_METADATA_JSON:-}" ]; } \
    || { [ -z "${GB10_RELEASE_MANIFEST_JSON:-}" ] \
        && [ -n "${GB10_RUNTIME_IMAGE_METADATA_JSON:-}" ]; }; then
    python3 - "$evidence_report" "$image" <<'PY'
import json
import os
import sys

path, image_ref = sys.argv[1:3]
report = {
    "schema_version": 1,
    "status": "failed",
    "phase": "pre_smoke_provenance_guard",
    "message": (
        "GB10 release provenance requires both GB10_RELEASE_MANIFEST_JSON "
        "and GB10_RUNTIME_IMAGE_METADATA_JSON, or neither, before final-image "
        "smoke."
    ),
    "image_ref": image_ref,
    "release_tag": os.environ.get("GB10_RELEASE_TAG"),
    "release_manifest_json": os.environ.get("GB10_RELEASE_MANIFEST_JSON"),
    "runtime_image_metadata_json": os.environ.get(
        "GB10_RUNTIME_IMAGE_METADATA_JSON"
    ),
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
    bundle_available_release_evidence
    echo "GB10 release provenance requires both GB10_RELEASE_MANIFEST_JSON and GB10_RUNTIME_IMAGE_METADATA_JSON." >&2
    exit 1
fi

if [ -n "${GB10_RELEASE_TAG:-}" ] \
    && { [ -z "${GB10_RELEASE_MANIFEST_JSON:-}" ] \
        || [ -z "${GB10_RUNTIME_IMAGE_METADATA_JSON:-}" ]; }; then
    python3 - "$evidence_report" "$image" <<'PY'
import json
import os
import sys

path, image_ref = sys.argv[1:3]
report = {
    "schema_version": 1,
    "status": "failed",
    "phase": "pre_smoke_provenance_guard",
    "message": (
        "GB10 release-tag smoke requires release provenance. Set both "
        "GB10_RELEASE_MANIFEST_JSON and GB10_RUNTIME_IMAGE_METADATA_JSON "
        "before attaching a release tag to final-image smoke evidence."
    ),
    "image_ref": image_ref,
    "release_tag": os.environ.get("GB10_RELEASE_TAG"),
    "release_manifest_json": os.environ.get("GB10_RELEASE_MANIFEST_JSON"),
    "runtime_image_metadata_json": os.environ.get(
        "GB10_RUNTIME_IMAGE_METADATA_JSON"
    ),
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
    bundle_available_release_evidence
    echo "GB10 release-tag smoke requires release provenance: set both GB10_RELEASE_MANIFEST_JSON and GB10_RUNTIME_IMAGE_METADATA_JSON." >&2
    exit 1
fi

if [ "${GB10_RELEASE_ALLOW_EXISTING_VLLM_CONTAINERS:-0}" != "1" ] \
    && command -v docker >/dev/null 2>&1; then
    existing_vllm_containers="$(
        docker ps --format '{{.ID}} {{.Names}} {{.Image}} {{.Command}}' 2>/dev/null \
            | awk 'BEGIN { IGNORECASE = 1 } /vllm/ { print }'
    )"
    if [ -n "$existing_vllm_containers" ]; then
        export GB10_EXISTING_VLLM_CONTAINERS="$existing_vllm_containers"
        python3 - "$evidence_report" "$image" <<'PY'
import json
import os
import sys

path, image_ref = sys.argv[1:3]
report = {
    "schema_version": 1,
    "status": "failed",
    "phase": "pre_smoke_resource_guard",
    "message": (
        "GB10 final-image smoke refused to start while an existing vLLM "
        "Docker container was running."
    ),
    "image_ref": image_ref,
    "release_tag": os.environ.get("GB10_RELEASE_TAG"),
    "allow_override_env": "GB10_RELEASE_ALLOW_EXISTING_VLLM_CONTAINERS=1",
    "existing_vllm_containers": [
        line
        for line in os.environ.get("GB10_EXISTING_VLLM_CONTAINERS", "").splitlines()
        if line
    ],
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
        bundle_available_release_evidence
        echo "GB10 final-image smoke refused to start while an existing vLLM Docker container is running:" >&2
        printf '%s\n' "$existing_vllm_containers" >&2
        echo "Stop the existing service first, or set GB10_RELEASE_ALLOW_EXISTING_VLLM_CONTAINERS=1 for an intentionally isolated runner." >&2
        exit 1
    fi
fi

if [ -z "${GB10_IMAGE_DIGEST:-}" ] && command -v docker >/dev/null 2>&1; then
    image_digest="$(
        docker image inspect "$image" \
            --format '{{range .RepoDigests}}{{println .}}{{end}}' 2>/dev/null \
            | sed -n '1p'
    )"
    if [ -n "$image_digest" ]; then
        export GB10_IMAGE_DIGEST="$image_digest"
    fi
fi
if [ -n "${GB10_IMAGE_DIGEST:-}" ]; then
    printf '%s\n' "$GB10_IMAGE_DIGEST" > "$smoked_image_digest_report"
fi

if ! has_arg --gb10-report-json "${offline_args[@]}"; then
    offline_args+=(--gb10-report-json "/gb10-smoke-reports/$(basename "$nvfp4_report")")
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
if [ -n "${GB10_RELEASE_TAG:-}" ] \
    && ! has_arg --gb10-release-tag "${verify_args[@]}"; then
    verify_args+=(--gb10-release-tag "$GB10_RELEASE_TAG")
fi
if ! has_arg --gb10-output-json "${verify_args[@]}"; then
    verify_args+=(--gb10-output-json "$evidence_report")
fi
if [ -n "${GB10_RELEASE_MANIFEST_JSON:-}" ] \
    && ! has_arg --gb10-release-manifest-json "${verify_args[@]}"; then
    verify_args+=(--gb10-release-manifest-json "$GB10_RELEASE_MANIFEST_JSON")
fi
if [ -n "${GB10_RUNTIME_IMAGE_METADATA_JSON:-}" ] \
    && ! has_arg --gb10-runtime-image-metadata-json "${verify_args[@]}"; then
    verify_args+=(--gb10-runtime-image-metadata-json "$GB10_RUNTIME_IMAGE_METADATA_JSON")
fi
if [ -n "${GB10_IMAGE_DIGEST:-}" ] \
    && ! has_arg --gb10-image-digest "${verify_args[@]}"; then
    verify_args+=(--gb10-image-digest "$GB10_IMAGE_DIGEST")
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
if [ -n "${GB10_RELEASE_TAG:-}" ]; then
    bundle_args+=(--gb10-release-tag "$GB10_RELEASE_TAG")
fi
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

echo "GB10 release evidence assets:" >&2
"$asset_lister" >&2

exit "$verify_status"
