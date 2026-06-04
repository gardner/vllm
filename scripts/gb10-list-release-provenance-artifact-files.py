#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""List downloaded GB10 release provenance artifact files."""

from __future__ import annotations

import argparse
from pathlib import Path

from gb10_release_contract import (
    RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS,
    default_release_provenance_dir,
    release_provenance_artifact_paths,
)


def list_release_provenance_artifact_files(
    provenance_dir: Path,
) -> list[Path]:
    artifact_paths = release_provenance_artifact_paths(provenance_dir)
    return [
        artifact_paths[kind]
        for kind in RELEASE_PROVENANCE_ARTIFACT_FILE_KINDS
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List downloaded GB10 release provenance artifact files, "
            "one path per line."
        )
    )
    parser.add_argument(
        "--gb10-provenance-dir",
        type=Path,
        default=default_release_provenance_dir(),
        help="Directory containing downloaded GB10 release provenance files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for path in list_release_provenance_artifact_files(args.gb10_provenance_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
