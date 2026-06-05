#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write GB10 runtime image ref/digest provenance files from BuildKit metadata."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from gb10_release_contract import (
    SHA256_DIGEST_RE,
    VLLM_RELEASE_ASSET_FILES,
    default_release_manifest_dir,
    default_runtime_image_metadata_json,
)


def _parse_bool(name: str, value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Unsupported GB10 boolean setting {name}={value}. "
        "Use 1, 0, true, false, yes, no, on, or off."
    )


def _mapping_value(mapping: Any, key: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    return mapping.get(key)


def extract_runtime_image_digest(metadata: dict[str, Any]) -> str | None:
    candidates = [
        metadata.get("containerimage.digest"),
        _mapping_value(metadata.get("containerimage.descriptor"), "digest"),
        _mapping_value(metadata.get("image"), "digest"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def write_runtime_image_provenance(
    *,
    runtime_image_metadata_json: Path,
    release_manifest_dir: Path,
    image_name: str,
    image_tag: str,
    push_image: bool,
) -> list[str]:
    errors: list[str] = []
    release_manifest_dir.mkdir(parents=True, exist_ok=True)

    image_ref = f"{image_name}:{image_tag}"
    image_ref_path = (
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_ref"]
    )
    image_digest_path = (
        release_manifest_dir / VLLM_RELEASE_ASSET_FILES["runtime_image_digest"]
    )
    image_ref_path.unlink(missing_ok=True)
    image_digest_path.unlink(missing_ok=True)

    try:
        metadata = json.loads(runtime_image_metadata_json.read_text(encoding="utf-8"))
    except OSError as exc:
        return [
            "GB10 runtime image metadata could not be read: "
            f"{runtime_image_metadata_json}: {exc}"
        ]
    except json.JSONDecodeError as exc:
        return [
            "GB10 runtime image metadata is not valid JSON: "
            f"{runtime_image_metadata_json}: {exc}"
        ]
    if not isinstance(metadata, dict):
        return ["GB10 runtime image metadata root must be a JSON object."]

    image_digest = extract_runtime_image_digest(metadata)
    if not image_digest:
        if push_image:
            errors.append(
                "GB10 pushed runtime image metadata did not include a digest."
            )
        else:
            image_ref_path.write_text(f"{image_ref}\n", encoding="utf-8")
        return errors

    if not SHA256_DIGEST_RE.fullmatch(image_digest):
        errors.append(
            "GB10 runtime image metadata digest is not a valid sha256 digest: "
            f"{image_digest}"
        )
        return errors

    image_ref_path.write_text(f"{image_ref}\n", encoding="utf-8")
    image_digest_path.write_text(f"{image_digest}\n", encoding="utf-8")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write gb10-runtime-image-ref.txt and gb10-runtime-image-digest.txt "
            "from BuildKit runtime image metadata."
        )
    )
    parser.add_argument(
        "--gb10-runtime-image-metadata-json",
        type=Path,
        default=default_runtime_image_metadata_json(),
        help="Path to BuildKit runtime image metadata JSON.",
    )
    parser.add_argument(
        "--gb10-release-manifest-dir",
        type=Path,
        default=default_release_manifest_dir(),
        help="Directory where runtime image provenance files are written.",
    )
    parser.add_argument(
        "--gb10-image-name",
        default=os.environ.get("GB10_IMAGE_NAME", ""),
        help="Runtime image repository name.",
    )
    parser.add_argument(
        "--gb10-image-tag",
        default=os.environ.get("GB10_IMAGE_TAG", ""),
        help="Runtime image tag.",
    )
    parser.add_argument(
        "--gb10-push-image",
        default=os.environ.get("GB10_PUSH_IMAGE", "false"),
        help="Whether the runtime image was pushed and must have a digest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        push_image = _parse_bool("--gb10-push-image", args.gb10_push_image)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    errors = write_runtime_image_provenance(
        runtime_image_metadata_json=args.gb10_runtime_image_metadata_json,
        release_manifest_dir=args.gb10_release_manifest_dir,
        image_name=args.gb10_image_name,
        image_tag=args.gb10_image_tag,
        push_image=push_image,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
