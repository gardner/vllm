"""Shared GB10 release contract constants for release and evidence tools."""

from __future__ import annotations

import re

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

GB10_SUPPORT_STATUSES = frozenset(
    {
        "supported_native",
        "supported_routed",
        "not_supported",
        "deferred",
    }
)

EXPECTED_RELEASE_REPORTS = (
    "gb10-nvfp4-smoke.json",
    "gb10-openai-server-smoke-image.json",
    "gb10-release-evidence-image.json",
)

EXPECTED_RELEASE_EVIDENCE_FILES = (
    *EXPECTED_RELEASE_REPORTS,
    "gb10-smoked-image-digest.txt",
)

PROVENANCE_FILES = {
    "release_manifest": "gb10-release-manifest.json",
    "runtime_image_metadata": "buildx-runtime-image-metadata.json",
}

PROVENANCE_RELATIVE_PATHS = {
    "release_manifest": "provenance/gb10-release-manifest.json",
    "runtime_image_metadata": "provenance/buildx-runtime-image-metadata.json",
}

REQUIRED_RELEASE_EVIDENCE_PROVENANCE = (
    "release_manifest",
    "runtime_image_metadata",
)

SHA256_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
