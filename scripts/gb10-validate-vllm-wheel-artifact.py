#!/usr/bin/env python3
"""Validate the local GB10 vLLM wheel artifact for cached builds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gb10-dist-dir", required=True, type=Path)
    parser.add_argument("--gb10-vllm-version", required=True)
    parser.add_argument("--gb10-context", default="GB10 local build")
    parser.add_argument(
        "--gb10-remediation",
        default=(
            "Run scripts/gb10-build-cached.sh wheel first, or set "
            "GB10_LOCAL_DIST_DIR to a directory with one current vLLM wheel."
        ),
    )
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    wheels = sorted(args.gb10_dist_dir.glob("vllm-*.whl"))

    if len(wheels) != 1:
        print(
            f"{args.gb10_context} requires exactly one vLLM wheel in "
            f"{args.gb10_dist_dir}.",
            file=sys.stderr,
        )
        print(args.gb10_remediation, file=sys.stderr)
        return 1

    wheel = wheels[0]
    expected_prefix = f"vllm-{args.gb10_vllm_version}-"
    if not wheel.name.startswith(expected_prefix):
        print(
            f"{args.gb10_context} wheel {wheel.name} does not match "
            f"GB10_VLLM_VERSION={args.gb10_vllm_version}.",
            file=sys.stderr,
        )
        print(args.gb10_remediation, file=sys.stderr)
        return 1

    print(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
