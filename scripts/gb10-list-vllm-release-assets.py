#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""List GB10 vLLM release assets for GitHub Release upload."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from gb10_release_contract import PROVENANCE_FILES, VLLM_RELEASE_ASSET_FILES


def _default_manifest_dir() -> Path:
    return Path(os.environ.get("GB10_RELEASE_MANIFEST_DIR", "gb10-release-manifest"))


def _default_runtime_image_metadata_json() -> Path:
    explicit = os.environ.get("GB10_RUNTIME_IMAGE_METADATA_JSON")
    if explicit:
        return Path(explicit)
    return _default_manifest_dir() / PROVENANCE_FILES["runtime_image_metadata"]


def list_release_assets(
    *,
    dist_dir: Path,
    release_manifest_dir: Path,
    runtime_image_metadata_json: Path,
) -> tuple[list[Path], list[str]]:
    errors: list[str] = []
    wheel_assets = sorted(dist_dir.glob("vllm-*.whl"))
    if len(wheel_assets) != 1:
        errors.append(
            "GB10 release publication expects exactly one vLLM wheel. "
            f"found={len(wheel_assets)} dist_dir={dist_dir}"
        )

    release_assets = [
        *wheel_assets,
        release_manifest_dir / PROVENANCE_FILES["release_manifest"],
        runtime_image_metadata_json,
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"],
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["checksums"],
    ]
    for asset in release_assets:
        if not asset.is_file() or asset.stat().st_size <= 0:
            errors.append(f"GB10 release upload asset is missing or empty: {asset}")

    return release_assets, errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List GB10 vLLM release assets, one path per line."
    )
    parser.add_argument(
        "--gb10-dist-dir",
        type=Path,
        default=Path("dist"),
        help="Directory containing the built vLLM wheel.",
    )
    parser.add_argument(
        "--gb10-release-manifest-dir",
        type=Path,
        default=_default_manifest_dir(),
        help="Directory containing GB10 release manifest/provenance files.",
    )
    parser.add_argument(
        "--gb10-runtime-image-metadata-json",
        type=Path,
        default=_default_runtime_image_metadata_json(),
        help="Path to BuildKit runtime image metadata JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    release_assets, errors = list_release_assets(
        dist_dir=args.gb10_dist_dir,
        release_manifest_dir=args.gb10_release_manifest_dir,
        runtime_image_metadata_json=args.gb10_runtime_image_metadata_json,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    for asset in release_assets:
        print(asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
