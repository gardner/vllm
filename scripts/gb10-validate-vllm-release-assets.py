#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate GB10 vLLM release assets before GitHub Release upload."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

from gb10_release_contract import PROVENANCE_FILES, VLLM_RELEASE_ASSET_FILES

SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")


def _default_manifest_dir() -> Path:
    return Path(os.environ.get("GB10_RELEASE_MANIFEST_DIR", "gb10-release-manifest"))


def _default_runtime_image_metadata_json() -> Path:
    explicit = os.environ.get("GB10_RUNTIME_IMAGE_METADATA_JSON")
    if explicit:
        return Path(explicit)
    return _default_manifest_dir() / PROVENANCE_FILES["runtime_image_metadata"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the GB10 vLLM wheel, release manifest, runtime image "
            "provenance files, and release checksums before publication."
        )
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
        help="Repository root used to resolve checksum paths.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checksum_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _validate_checksums(
    *,
    checksum_file: Path,
    repo_root: Path,
    required_assets: list[Path],
) -> list[str]:
    errors: list[str] = []
    seen_paths: set[Path] = set()
    try:
        checksum_lines = checksum_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"GB10 release checksum file cannot be read: {checksum_file}: {exc}"]

    if not checksum_lines:
        errors.append(f"GB10 release checksum file is empty: {checksum_file}")

    for line in checksum_lines:
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"GB10 release checksum line is malformed: {line!r}")
            continue
        expected_digest, raw_path = parts
        raw_path = raw_path.lstrip("*")
        if not SHA256_HEX_RE.fullmatch(expected_digest):
            errors.append(
                "GB10 release checksum line has an invalid sha256 digest: "
                f"{line!r}"
            )
            continue
        asset_path = _resolve_checksum_path(repo_root, raw_path)
        seen_paths.add(asset_path.resolve())
        if not asset_path.is_file():
            errors.append(f"GB10 release checksum references missing file: {raw_path}")
            continue
        actual_digest = _sha256(asset_path)
        if actual_digest != expected_digest:
            errors.append(
                "GB10 release checksum mismatch for "
                f"{raw_path}: expected {expected_digest}, got {actual_digest}"
            )

    missing_checksum_assets = [
        asset
        for asset in required_assets
        if asset.resolve() not in seen_paths
    ]
    if missing_checksum_assets:
        missing = ", ".join(str(asset) for asset in missing_checksum_assets)
        errors.append(f"GB10 release checksums omit required assets: {missing}")

    return errors


def validate_release_assets(
    *,
    dist_dir: Path,
    release_manifest_dir: Path,
    runtime_image_metadata_json: Path,
    repo_root: Path,
) -> list[str]:
    errors: list[str] = []
    wheel_assets = sorted(dist_dir.glob("vllm-*.whl"))
    if len(wheel_assets) != 1:
        errors.append(
            "GB10 release publication expects exactly one vLLM wheel. "
            f"found={len(wheel_assets)} dist_dir={dist_dir}"
        )

    required_assets = [
        *wheel_assets,
        release_manifest_dir / PROVENANCE_FILES["release_manifest"],
        runtime_image_metadata_json,
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"],
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"],
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["checksums"],
    ]

    for asset in required_assets:
        if not asset.is_file() or asset.stat().st_size <= 0:
            errors.append(f"GB10 release asset is missing or empty: {asset}")

    checksum_file = release_manifest_dir / VLLM_RELEASE_ASSET_FILES["checksums"]
    if checksum_file.is_file() and checksum_file.stat().st_size > 0:
        checksummed_assets = [
            asset for asset in required_assets if asset != checksum_file
        ]
        errors.extend(
            _validate_checksums(
                checksum_file=checksum_file,
                repo_root=repo_root,
                required_assets=checksummed_assets,
            )
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = validate_release_assets(
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
