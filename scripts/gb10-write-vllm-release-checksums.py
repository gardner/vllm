#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write SHA256 checksums for GB10 vLLM release assets."""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_checksum_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _release_checksum_assets(
    *,
    dist_dir: Path,
    release_manifest_dir: Path,
    runtime_image_metadata_json: Path,
) -> tuple[list[Path], list[str]]:
    errors: list[str] = []
    wheel_assets = sorted(dist_dir.glob("vllm-*.whl"))
    if len(wheel_assets) != 1:
        errors.append(
            "GB10 release checksum generation expects exactly one vLLM wheel. "
            f"found={len(wheel_assets)} dist_dir={dist_dir}"
        )

    checksum_assets = [
        *wheel_assets,
        release_manifest_dir / PROVENANCE_FILES["release_manifest"],
        runtime_image_metadata_json,
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
    ]

    runtime_image_digest = (
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"]
    )
    if runtime_image_digest.is_file():
        checksum_assets.append(runtime_image_digest)

    for asset in checksum_assets:
        if not asset.is_file() or asset.stat().st_size <= 0:
            errors.append(f"GB10 release checksum asset is missing or empty: {asset}")

    return checksum_assets, errors


def write_release_checksums(
    *,
    dist_dir: Path,
    release_manifest_dir: Path,
    runtime_image_metadata_json: Path,
    repo_root: Path,
) -> list[str]:
    checksum_file = release_manifest_dir / VLLM_RELEASE_ASSET_FILES["checksums"]
    checksum_file.unlink(missing_ok=True)
    checksum_assets, errors = _release_checksum_assets(
        dist_dir=dist_dir,
        release_manifest_dir=release_manifest_dir,
        runtime_image_metadata_json=runtime_image_metadata_json,
    )
    if errors:
        return errors

    checksum_lines = [
        f"{_sha256(asset)}  {_relative_checksum_path(asset, repo_root)}"
        for asset in checksum_assets
    ]
    checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write gb10-vllm-release-SHA256SUMS for GB10 release assets."
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
    parser.add_argument(
        "--gb10-repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to write stable checksum paths.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = write_release_checksums(
        dist_dir=args.gb10_dist_dir,
        release_manifest_dir=args.gb10_release_manifest_dir,
        runtime_image_metadata_json=args.gb10_runtime_image_metadata_json,
        repo_root=args.gb10_repo_root,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
