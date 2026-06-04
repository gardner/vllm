#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate GB10 evidence metadata before attaching it to a GitHub Release."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from gb10_release_contract import (
    REQUIRED_GB10_SUPPORT_MATRIX,
    REQUIRED_RELEASE_EVIDENCE_PROVENANCE,
    SHA256_DIGEST_RE,
)

MISSING_SOURCE_MESSAGE = (
    "GB10 evidence release metadata is missing source provenance."
)
INCOMPLETE_SUPPORT_MATRIX_MESSAGE = (
    "GB10 evidence release metadata does not prove a complete GB10 support matrix."
)
MISSING_SUPPORT_MATRIX_MESSAGE = (
    "GB10 evidence release metadata is missing a GB10 support matrix summary."
)
SUPPORT_MATRIX_MISMATCH_MESSAGE = (
    "GB10 evidence release metadata support matrix does not match "
    "the required GB10 release contract."
)
FAILED_RELEASE_GATE_MESSAGE = (
    "GB10 evidence release metadata does not prove a passed release gate."
)
IMAGE_REF_MISMATCH_MESSAGE = (
    "GB10 evidence release metadata image ref does not match input image-ref."
)
IMAGE_DIGEST_MISMATCH_MESSAGE = (
    "GB10 evidence release metadata image digest does not match pulled digest."
)
RELEASE_TAG_MISMATCH_MESSAGE = (
    "GB10 evidence release metadata tag does not match release-tag."
)
MISSING_PROVENANCE_MESSAGE = (
    "GB10 evidence release metadata is missing release provenance."
)


def normalize_image_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if "@" in value:
        value = value.rsplit("@", 1)[1].strip()
    marker = "sha256:"
    marker_index = value.find(marker)
    if marker_index >= 0:
        value = value[marker_index:]
    value = value.split()[0].strip().strip('",')
    if not SHA256_DIGEST_RE.fullmatch(value):
        return None
    return value


def validate_metadata(
    metadata: dict[str, Any],
    *,
    image_ref: str,
    image_digest: str,
    release_tag: str,
) -> list[str]:
    errors: list[str] = []
    source = metadata.get("source")
    if not isinstance(source, dict):
        return [MISSING_SOURCE_MESSAGE]

    support_matrix = metadata.get("support_matrix_summary")
    if metadata.get("support_matrix_complete") is not True:
        errors.append(
            f"{INCOMPLETE_SUPPORT_MATRIX_MESSAGE} support_matrix_complete="
            f"{metadata.get('support_matrix_complete')}"
        )

    if (
        not isinstance(support_matrix, dict)
        or support_matrix.get("present") is not True
        or support_matrix.get("release_manifest_present") is not True
        or support_matrix.get("architecture") != "sm_121a"
        or support_matrix.get("first_release_scope") != "single_spark_first_path"
        or support_matrix.get("entry_count", 0) <= 0
        or support_matrix.get("invalid_entries")
    ):
        errors.append(
            f"{MISSING_SUPPORT_MATRIX_MESSAGE} support_matrix_summary={support_matrix}"
        )
    elif (mismatched_support := _mismatched_support_entries(support_matrix)):
        errors.append(
            f"{SUPPORT_MATRIX_MISMATCH_MESSAGE} mismatched={mismatched_support}"
        )

    if metadata.get("status") != "complete" or (
        metadata.get("release_gate_passed") is not True
    ):
        errors.append(
            f"{FAILED_RELEASE_GATE_MESSAGE} status={metadata.get('status')} "
            f"release_gate_passed={metadata.get('release_gate_passed')}"
        )

    if source.get("image_ref") != image_ref:
        errors.append(
            f"{IMAGE_REF_MISMATCH_MESSAGE} metadata={source.get('image_ref')} "
            f"input={image_ref}"
        )

    metadata_digest = normalize_image_digest(source.get("image_digest"))
    pulled_digest = normalize_image_digest(image_digest)
    if metadata_digest is None or metadata_digest != pulled_digest:
        errors.append(
            f"{IMAGE_DIGEST_MISMATCH_MESSAGE} "
            f"metadata={source.get('image_digest')} pulled={image_digest}"
        )

    if source.get("release_tag") != release_tag:
        errors.append(
            f"{RELEASE_TAG_MISMATCH_MESSAGE} metadata={source.get('release_tag')} "
            f"input={release_tag}"
        )

    provenance_kinds = {
        item.get("kind")
        for item in metadata.get("included_provenance", [])
        if isinstance(item, dict)
    }
    required_provenance = set(REQUIRED_RELEASE_EVIDENCE_PROVENANCE)
    if not required_provenance.issubset(provenance_kinds):
        errors.append(
            f"{MISSING_PROVENANCE_MESSAGE} provenance={sorted(provenance_kinds)}"
        )

    return errors


def _mismatched_support_entries(support_matrix: dict[str, Any]) -> dict[str, Any]:
    entries = support_matrix.get("entries")
    return {
        name: {
            "expected": expected_status,
            "actual": entries.get(name) if isinstance(entries, dict) else None,
        }
        for name, expected_status in REQUIRED_GB10_SUPPORT_MATRIX.items()
        if not isinstance(entries, dict) or entries.get(name) != expected_status
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate release-evidence-metadata.json before attaching GB10 "
            "smoke evidence assets to a GitHub Release."
        )
    )
    parser.add_argument(
        "--gb10-metadata-json",
        type=Path,
        required=True,
        help="Path to release-evidence-metadata.json.",
    )
    parser.add_argument(
        "--gb10-image-ref",
        default=os.environ.get("GB10_IMAGE_REF", ""),
        help="Candidate runtime image ref being published.",
    )
    parser.add_argument(
        "--gb10-image-digest",
        default=os.environ.get("GB10_IMAGE_DIGEST", ""),
        help="Pulled immutable image digest for the candidate runtime image.",
    )
    parser.add_argument(
        "--gb10-release-tag",
        default=os.environ.get("GB10_RELEASE_TAG", ""),
        help="GitHub Release tag receiving the evidence assets.",
    )
    return parser


def _load_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise TypeError("release evidence metadata root must be a JSON object")
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metadata = _load_metadata(args.gb10_metadata_json)
    errors = validate_metadata(
        metadata,
        image_ref=args.gb10_image_ref,
        image_digest=args.gb10_image_digest,
        release_tag=args.gb10_release_tag,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
