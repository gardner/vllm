#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""List GB10 release-evidence assets for GitHub Release upload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gb10_release_contract import (
    default_release_evidence_bundle_name,
    default_release_evidence_output_dir,
    release_evidence_asset_paths,
)


def list_release_evidence_assets(
    *,
    output_dir: Path,
    bundle_name: str,
) -> tuple[list[Path], list[str]]:
    release_assets = release_evidence_asset_paths(output_dir, bundle_name)
    errors: list[str] = []
    for asset in release_assets:
        if not asset.is_file() or asset.stat().st_size <= 0:
            errors.append(
                f"GB10 evidence release upload asset is missing or empty: {asset}"
            )
    return release_assets, errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List GB10 release-evidence assets, one path per line."
    )
    parser.add_argument(
        "--gb10-output-dir",
        type=Path,
        default=default_release_evidence_output_dir(),
        help="Directory containing the GB10 release-evidence bundle assets.",
    )
    parser.add_argument(
        "--gb10-bundle-name",
        default=default_release_evidence_bundle_name(),
        help="Base name for the generated .tar.gz bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    release_assets, errors = list_release_evidence_assets(
        output_dir=args.gb10_output_dir,
        bundle_name=args.gb10_bundle_name,
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
