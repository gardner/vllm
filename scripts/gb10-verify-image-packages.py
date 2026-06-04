#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify final GB10 image package provenance without importing CUDA modules."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Callable
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

GB10_FLASHINFER_DISTRIBUTIONS = (
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)
GB10_DISTRIBUTIONS = ("vllm", *GB10_FLASHINFER_DISTRIBUTIONS)


def _distribution_version(
    distribution_name: str,
    version_getter: Callable[[str], str],
) -> str | None:
    try:
        return version_getter(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_image_package_report(
    *,
    image_ref: str | None = None,
    image_digest: str | None = None,
    version_getter: Callable[[str], str] = importlib_metadata.version,
) -> dict[str, Any]:
    versions = {
        distribution_name: _distribution_version(
            distribution_name,
            version_getter,
        )
        for distribution_name in GB10_DISTRIBUTIONS
    }
    missing = [
        distribution_name
        for distribution_name, version in versions.items()
        if version is None
    ]
    non_gb10_flashinfer = {
        distribution_name: version
        for distribution_name, version in versions.items()
        if (
            distribution_name in GB10_FLASHINFER_DISTRIBUTIONS
            and version is not None
            and "+cu130gb10" not in version
        )
    }
    vllm_version = versions["vllm"]
    vllm_version_is_gb10 = bool(vllm_version and "+gb10." in vllm_version)
    status = (
        "passed"
        if not missing and not non_gb10_flashinfer and vllm_version_is_gb10
        else "failed"
    )

    return {
        "schema_version": 1,
        "status": status,
        "phase": "image_package_check",
        "message": (
            "GB10 final image package versions are compatible"
            if status == "passed"
            else "GB10 final image package versions are not compatible"
        ),
        "image_ref": image_ref,
        "image_digest": image_digest,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": versions,
        "missing_distributions": missing,
        "vllm_version_is_gb10": vllm_version_is_gb10,
        "flashinfer_versions_are_gb10_cuda13": {
            distribution_name: (
                versions[distribution_name] is not None
                and "+cu130gb10" in versions[distribution_name]
            )
            for distribution_name in GB10_FLASHINFER_DISTRIBUTIONS
        },
        "non_gb10_flashinfer_distributions": non_gb10_flashinfer,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a final GB10 image contains the GB10 vLLM and "
            "FlashInfer runtime distributions without importing CUDA modules."
        )
    )
    parser.add_argument(
        "--gb10-output-json",
        type=Path,
        required=True,
        help="Path to write the JSON package provenance report.",
    )
    parser.add_argument(
        "--gb10-image-ref",
        default=os.environ.get("GB10_IMAGE_REF"),
        help="Candidate image ref recorded in the report.",
    )
    parser.add_argument(
        "--gb10-image-digest",
        default=os.environ.get("GB10_IMAGE_DIGEST"),
        help="Candidate image digest recorded in the report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = collect_image_package_report(
        image_ref=args.gb10_image_ref,
        image_digest=args.gb10_image_digest,
    )
    args.gb10_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.gb10_output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
