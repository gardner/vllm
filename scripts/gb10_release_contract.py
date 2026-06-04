"""Shared GB10 release contract constants for release and evidence tools."""

from __future__ import annotations

import os
import re
from pathlib import Path

FLASHINFER_RUNTIME_DISTRIBUTIONS = (
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)

REQUIRED_FLASHINFER_COMPONENTS = (
    "flashinfer_python",
    "flashinfer_cubin",
    "flashinfer_jit_cache",
)

REQUIRED_SOURCE_DEPENDENCIES = (
    "deepgemm",
    "flashmla",
    "triton_kernels",
)

REQUIRED_GB10_SUPPORT_MATRIX = {
    "flashinfer_nvfp4_dense": "supported_native",
    "flashinfer_nvfp4_quantization": "supported_native",
    "modelopt_fp4_quantization": "supported_native",
    "flashinfer_attention_fa2": "supported_native",
    "flashinfer_b12x_non_ep_moe": "supported_native",
    "flashmla_attention": "supported_native",
    "public_flashattention_runtime": "not_supported",
    "trtllm_gen_attention": "not_supported",
    "trtllm_gen_moe": "not_supported",
    "marlin_nvfp4_fallback": "not_supported",
    "flashinfer_b12x_ep_all2all_eplb": "deferred",
    "multi_spark_ep_all2all_eplb": "deferred",
}

GB10_NOT_SUPPORTED_PATH_REASONS = {
    "public_flashattention_runtime": (
        "Public FlashAttention runtime is not validated for the GB10 first "
        "release path; route attention through FlashInfer or FlashMLA."
    ),
    "trtllm_gen_attention": (
        "TRTLLM Gen attention artifacts and metadata do not support SM121; "
        "route GB10 attention through FlashInfer or FlashMLA."
    ),
    "trtllm_gen_moe": (
        "TRTLLM Gen MoE rejects SM121 today and must not satisfy native NVFP4 "
        "MoE release evidence."
    ),
    "marlin_nvfp4_fallback": (
        "Marlin can prove fallback serving reachability, but it is not native "
        "GB10 NVFP4 Tensor Core evidence."
    ),
}

GB10_SUPPORT_STATUSES = frozenset(
    {
        "supported_native",
        "supported_routed",
        "not_supported",
        "deferred",
    }
)

RELEASE_NVFP4_SMOKE_REPORT_FILE = "gb10-nvfp4-smoke.json"
RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE = "gb10-openai-server-smoke-image.json"
RELEASE_EVIDENCE_SUMMARY_REPORT_FILE = "gb10-release-evidence-image.json"
RELEASE_SMOKED_IMAGE_DIGEST_FILE = "gb10-smoked-image-digest.txt"

EXPECTED_RELEASE_REPORTS = (
    RELEASE_NVFP4_SMOKE_REPORT_FILE,
    RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE,
    RELEASE_EVIDENCE_SUMMARY_REPORT_FILE,
)

EXPECTED_RELEASE_EVIDENCE_FILES = (
    *EXPECTED_RELEASE_REPORTS,
    RELEASE_SMOKED_IMAGE_DIGEST_FILE,
)

PROVENANCE_FILES = {
    "release_manifest": "gb10-release-manifest.json",
    "runtime_image_metadata": "buildx-runtime-image-metadata.json",
}

VLLM_RELEASE_ASSET_FILES = {
    "runtime_image_ref": "gb10-runtime-image-ref.txt",
    "runtime_image_digest": "gb10-runtime-image-digest.txt",
    "checksums": "gb10-vllm-release-SHA256SUMS",
}

DEFAULT_RELEASE_MANIFEST_DIR = Path("gb10-release-manifest")
DEFAULT_VLLM_RELEASE_DIST_DIR = Path("dist")
DEFAULT_RELEASE_EVIDENCE_REPORT_DIR = Path("gb10-smoke-reports")
DEFAULT_RELEASE_EVIDENCE_OUTPUT_DIR = Path("dist/gb10-release-evidence")
DEFAULT_RELEASE_PROVENANCE_DIR = Path("gb10-release-provenance")
GITHUB_RELEASE_MANIFEST_ARTIFACT_NAME = "gb10-release-manifest"
GITHUB_RELEASE_INPUTS_ARTIFACT_NAME = "gb10-release-inputs"
GITHUB_RELEASE_EVIDENCE_ARTIFACT_NAME = "gb10-release-evidence"
RELEASE_EVIDENCE_BUNDLE_NAME = "gb10-release-evidence"
RELEASE_EVIDENCE_METADATA_FILE = "release-evidence-metadata.json"
RELEASE_EVIDENCE_CHECKSUM_FILE = "SHA256SUMS"
VLLM_RELEASE_WHEEL_GLOB = "vllm-*.whl"

PROVENANCE_RELATIVE_PATHS = {
    "release_manifest": "provenance/gb10-release-manifest.json",
    "runtime_image_metadata": "provenance/buildx-runtime-image-metadata.json",
}

REQUIRED_RELEASE_EVIDENCE_PROVENANCE = (
    "release_manifest",
    "runtime_image_metadata",
)

RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS = (
    "release_manifest",
    "runtime_image_metadata",
    "runtime_image_ref",
    "runtime_image_digest",
)

SHA256_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


def default_release_manifest_dir() -> Path:
    return Path(
        os.environ.get("GB10_RELEASE_MANIFEST_DIR", DEFAULT_RELEASE_MANIFEST_DIR)
    )


def default_release_manifest_json() -> Path:
    explicit = os.environ.get("GB10_RELEASE_MANIFEST_JSON")
    if explicit:
        return Path(explicit)
    return default_release_manifest_dir() / PROVENANCE_FILES["release_manifest"]


def default_runtime_image_metadata_json() -> Path:
    explicit = os.environ.get("GB10_RUNTIME_IMAGE_METADATA_JSON")
    if explicit:
        return Path(explicit)
    return default_release_manifest_dir() / PROVENANCE_FILES["runtime_image_metadata"]


def default_release_evidence_manifest_json() -> Path | None:
    explicit = os.environ.get("GB10_RELEASE_EVIDENCE_MANIFEST_JSON") or os.environ.get(
        "GB10_RELEASE_MANIFEST_JSON"
    )
    if explicit:
        return Path(explicit)
    manifest_dir = os.environ.get("GB10_RELEASE_MANIFEST_DIR")
    if manifest_dir:
        return Path(manifest_dir) / PROVENANCE_FILES["release_manifest"]
    return None


def default_release_evidence_runtime_image_metadata_json() -> Path | None:
    explicit = os.environ.get(
        "GB10_RELEASE_EVIDENCE_RUNTIME_IMAGE_METADATA_JSON"
    ) or os.environ.get("GB10_RUNTIME_IMAGE_METADATA_JSON")
    if explicit:
        return Path(explicit)
    return None


def default_release_evidence_report_dir() -> Path:
    return Path(
        os.environ.get(
            "GB10_RELEASE_EVIDENCE_REPORT_DIR",
            DEFAULT_RELEASE_EVIDENCE_REPORT_DIR,
        )
    )


def default_release_evidence_output_dir() -> Path:
    return Path(
        os.environ.get(
            "GB10_RELEASE_EVIDENCE_OUTPUT_DIR",
            DEFAULT_RELEASE_EVIDENCE_OUTPUT_DIR,
        )
    )


def default_release_provenance_dir() -> Path:
    return Path(
        os.environ.get(
            "GB10_RELEASE_PROVENANCE_DIR",
            DEFAULT_RELEASE_PROVENANCE_DIR,
        )
    )


def default_release_evidence_bundle_name() -> str:
    return os.environ.get(
        "GB10_RELEASE_EVIDENCE_BUNDLE_NAME",
        RELEASE_EVIDENCE_BUNDLE_NAME,
    )


def release_evidence_bundle_archive_name(
    bundle_name: str = RELEASE_EVIDENCE_BUNDLE_NAME,
) -> str:
    return f"{bundle_name}.tar.gz"


def release_evidence_bundle_archive_checksum_name(
    bundle_name: str = RELEASE_EVIDENCE_BUNDLE_NAME,
) -> str:
    return f"{release_evidence_bundle_archive_name(bundle_name)}.sha256"


def release_evidence_asset_paths(
    output_dir: Path,
    bundle_name: str = RELEASE_EVIDENCE_BUNDLE_NAME,
) -> list[Path]:
    return [
        output_dir / release_evidence_bundle_archive_name(bundle_name),
        output_dir / release_evidence_bundle_archive_checksum_name(bundle_name),
        output_dir / RELEASE_EVIDENCE_METADATA_FILE,
        output_dir / RELEASE_EVIDENCE_CHECKSUM_FILE,
    ]


def release_evidence_file_paths(report_dir: Path) -> list[Path]:
    return [
        report_dir / filename
        for filename in EXPECTED_RELEASE_EVIDENCE_FILES
    ]


def release_provenance_artifact_paths(provenance_dir: Path) -> dict[str, Path]:
    paths = {
        "release_manifest": provenance_dir / PROVENANCE_FILES["release_manifest"],
        "runtime_image_metadata": provenance_dir
        / PROVENANCE_FILES["runtime_image_metadata"],
        "runtime_image_ref": provenance_dir
        / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
        "runtime_image_digest": provenance_dir
        / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"],
    }
    return {
        kind: paths[kind]
        for kind in RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS
    }


def find_vllm_wheel_assets(dist_dir: Path) -> list[Path]:
    return sorted(dist_dir.glob(VLLM_RELEASE_WHEEL_GLOB))


def vllm_release_asset_paths(
    *,
    dist_dir: Path,
    release_manifest_dir: Path,
    runtime_image_metadata_json: Path,
) -> list[Path]:
    return [
        *find_vllm_wheel_assets(dist_dir),
        release_manifest_dir / PROVENANCE_FILES["release_manifest"],
        runtime_image_metadata_json,
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"],
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["checksums"],
    ]


def vllm_release_checksum_asset_paths(
    *,
    dist_dir: Path,
    release_manifest_dir: Path,
    runtime_image_metadata_json: Path,
) -> list[Path]:
    checksum_assets = [
        *find_vllm_wheel_assets(dist_dir),
        release_manifest_dir / PROVENANCE_FILES["release_manifest"],
        runtime_image_metadata_json,
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
    ]

    runtime_image_digest = (
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"]
    )
    if runtime_image_digest.is_file():
        checksum_assets.append(runtime_image_digest)
    return checksum_assets
