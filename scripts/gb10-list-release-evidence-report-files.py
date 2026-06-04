#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""List GB10 final-image release evidence report files."""

from __future__ import annotations

import argparse
from pathlib import Path

from gb10_release_contract import (
    default_release_evidence_report_dir,
    release_evidence_file_paths,
)


def list_release_evidence_report_files(report_dir: Path) -> list[Path]:
    return release_evidence_file_paths(report_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List GB10 final-image release evidence files, one path per line."
        )
    )
    parser.add_argument(
        "--gb10-report-dir",
        type=Path,
        default=default_release_evidence_report_dir(),
        help="Directory containing GB10 final-image smoke reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for path in list_release_evidence_report_files(args.gb10_report_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
