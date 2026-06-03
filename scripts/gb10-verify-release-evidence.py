#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify GB10 smoke JSON reports before treating artifacts as releasable.

This script is intentionally report-only. It does not start vLLM, make HTTP
requests, import torch, or touch CUDA. It reads the durable JSON reports emitted
by the GB10 offline NVFP4 smoke and OpenAI-compatible server smoke scripts and
turns them into one release-gate summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify GB10 release evidence from the offline NVFP4 model smoke "
            "and OpenAI-compatible server smoke JSON reports."
        )
    )
    parser.add_argument(
        "--gb10-nvfp4-report-json",
        help="Path to gb10-nvfp4-smoke.json from scripts/gb10-smoke-nvfp4.py.",
    )
    parser.add_argument(
        "--gb10-openai-report-json",
        help=(
            "Path to gb10-openai-server-smoke.json from "
            "scripts/gb10-smoke-openai-server.py."
        ),
    )
    parser.add_argument(
        "--gb10-output-json",
        help="Optional path for the combined release-evidence summary.",
    )
    parser.add_argument(
        "--gb10-require-moe",
        action="store_true",
        help="Require native non-EP NVFP4 MoE evidence in the offline report.",
    )
    parser.add_argument(
        "--gb10-require-openai-deterministic",
        action="store_true",
        help="Require deterministic-generation evidence in the server report.",
    )
    parser.add_argument(
        "--gb10-allow-partial",
        action="store_true",
        help=(
            "Exit zero while marking the summary partial if evidence is "
            "missing or incomplete. Release jobs should not pass this flag."
        ),
    )
    return parser


def _load_report(path: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "report path was not provided"
    report_path = Path(path)
    if not report_path.exists():
        return None, f"report path does not exist: {report_path}"
    try:
        value = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"report is not valid JSON: {exc}"
    if not isinstance(value, dict):
        return None, "report root is not a JSON object"
    return value, None


def _nested_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _check(
    *,
    name: str,
    passed: bool,
    message: str,
    details: dict[str, Any] | None = None,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "required": required,
        "message": message,
        "details": details or {},
    }


def _missing_check(name: str, message: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "status": "missing",
        "required": required,
        "message": message,
        "details": {},
    }


def _is_fallback_selection(selection: Any) -> bool:
    return isinstance(selection, dict) and bool(selection.get("is_fallback"))


def _check_nvfp4_report(
    report: dict[str, Any] | None,
    error: str | None,
    *,
    require_moe: bool,
) -> list[dict[str, Any]]:
    if report is None:
        return [_missing_check("nvfp4_report_present", error or "NVFP4 report missing")]

    fallback_events = report.get("fallback_events", [])
    backend_selections = report.get("backend_selections", [])
    native_gemm_status = _nested_get(
        report,
        "backend_summary",
        "capabilities",
        "native_nvfp4_gemm",
        "status",
    )
    native_moe_status = _nested_get(
        report,
        "backend_summary",
        "capabilities",
        "native_nvfp4_moe_non_ep",
        "status",
    )
    release_summary = report.get("gb10_release_summary")

    checks = [
        _check(
            name="nvfp4_report_status",
            passed=report.get("status") == "passed",
            message="offline NVFP4 smoke report status is passed",
            details={"status": report.get("status")},
        ),
        _check(
            name="nvfp4_first_path_smoke",
            passed=_nested_get(release_summary, "first_path_smoke_passed") is True,
            message="offline NVFP4 first-path smoke summary passed",
            details={
                "first_path_smoke_passed": _nested_get(
                    release_summary,
                    "first_path_smoke_passed",
                ),
                "smoke_blockers": _nested_get(release_summary, "smoke_blockers"),
            },
        ),
        _check(
            name="native_nvfp4_gemm_observed",
            passed=native_gemm_status == "observed",
            message="native NVFP4 dense GEMM backend was observed",
            details={"status": native_gemm_status},
        ),
        _check(
            name="nvfp4_fallback_free",
            passed=not fallback_events
            and not any(_is_fallback_selection(event) for event in backend_selections),
            message="offline NVFP4 smoke recorded no fallback events or selections",
            details={
                "fallback_event_count": len(fallback_events)
                if isinstance(fallback_events, list)
                else None,
                "fallback_selection_count": sum(
                    1
                    for event in backend_selections
                    if _is_fallback_selection(event)
                )
                if isinstance(backend_selections, list)
                else None,
            },
        ),
        _check(
            name="kv_cache_fp8_e4m3",
            passed=_nested_get(
                report,
                "gb10_release_summary",
                "checks",
                "kv_cache_dtype",
                "status",
            )
            == "passed",
            message="offline smoke observed requested FP8 KV cache dtype",
            details=_nested_get(
                report,
                "gb10_release_summary",
                "checks",
                "kv_cache_dtype",
            )
            or {},
        ),
    ]

    if require_moe:
        checks.append(
            _check(
                name="native_nvfp4_moe_non_ep_observed",
                passed=native_moe_status == "observed",
                message="native non-EP NVFP4 MoE backend was observed",
                details={"status": native_moe_status},
            )
        )
    else:
        checks.append(
            {
                "name": "native_nvfp4_moe_non_ep_observed",
                "status": "not_required",
                "required": False,
                "message": "native non-EP NVFP4 MoE evidence was not required",
                "details": {"status": native_moe_status},
            }
        )

    return checks


def _check_openai_report(
    report: dict[str, Any] | None,
    error: str | None,
    *,
    require_deterministic: bool,
) -> list[dict[str, Any]]:
    if report is None:
        return [
            _missing_check(
                "openai_report_present",
                error or "OpenAI-compatible server report missing",
            )
        ]

    generated_text = _nested_get(report, "response", "generated_text")
    deterministic_status = _nested_get(report, "deterministic_generation", "status")
    checks = [
        _check(
            name="openai_report_status",
            passed=report.get("status") == "passed",
            message="OpenAI-compatible server smoke report status is passed",
            details={"status": report.get("status")},
        ),
        _check(
            name="openai_server_smoke_passed",
            passed=_nested_get(
                report,
                "gb10_release_evidence",
                "openai_compatible_server_smoke",
                "status",
            )
            == "passed",
            message="OpenAI-compatible server smoke evidence passed",
            details=_nested_get(
                report,
                "gb10_release_evidence",
                "openai_compatible_server_smoke",
            )
            or {},
        ),
        _check(
            name="openai_http_200",
            passed=_nested_get(report, "response", "status") == 200,
            message="OpenAI-compatible generation request returned HTTP 200",
            details={"status": _nested_get(report, "response", "status")},
        ),
        _check(
            name="openai_generated_text",
            passed=isinstance(generated_text, str) and bool(generated_text),
            message="OpenAI-compatible generation returned generated text",
            details={
                "generated_text_source": _nested_get(
                    report,
                    "response",
                    "generated_text_source",
                )
            },
        ),
        _check(
            name="openai_served_model_metadata",
            passed=bool(_nested_get(report, "models", "selected", "root")),
            message="/v1/models selected served-model metadata is present",
            details=_nested_get(report, "models", "selected") or {},
        ),
        _check(
            name="openai_system_fingerprint",
            passed=bool(_nested_get(report, "response", "system_fingerprint")),
            message="OpenAI-compatible response includes vLLM system fingerprint",
            details={
                "system_fingerprint": _nested_get(
                    report,
                    "response",
                    "system_fingerprint",
                )
            },
        ),
    ]

    if require_deterministic:
        checks.append(
            _check(
                name="openai_deterministic_generation",
                passed=deterministic_status == "passed",
                message=(
                    "repeated seeded OpenAI-compatible generation was "
                    "deterministic"
                ),
                details=report.get("deterministic_generation") or {},
            )
        )
    else:
        checks.append(
            {
                "name": "openai_deterministic_generation",
                "status": "not_required",
                "required": False,
                "message": (
                    "deterministic OpenAI-compatible generation was not required"
                ),
                "details": report.get("deterministic_generation") or {},
            }
        )

    return checks


def _required_failures(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        check
        for check in checks
        if check.get("required") is not False
        and check.get("status") not in {"passed", "not_required"}
    ]


def _build_summary(
    *,
    nvfp4_report: dict[str, Any] | None,
    nvfp4_error: str | None,
    openai_report: dict[str, Any] | None,
    openai_error: str | None,
    require_moe: bool,
    require_openai_deterministic: bool,
    allow_partial: bool,
) -> dict[str, Any]:
    checks = [
        *_check_nvfp4_report(
            nvfp4_report,
            nvfp4_error,
            require_moe=require_moe,
        ),
        *_check_openai_report(
            openai_report,
            openai_error,
            require_deterministic=require_openai_deterministic,
        ),
    ]
    failures = _required_failures(checks)
    status = "passed"
    if failures:
        status = "partial" if allow_partial else "failed"

    return {
        "schema_version": 1,
        "status": status,
        "release_gate_passed": not failures,
        "allow_partial": allow_partial,
        "requirements": {
            "require_moe": require_moe,
            "require_openai_deterministic": require_openai_deterministic,
        },
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "remaining_release_evidence": [
            "final runtime image smoke with the published GB10 dependency wheels",
            "CUDA graph capture/replay validation",
            "prefill/decode benchmark evidence",
        ],
    }


def _write_summary(output_path: str | None, summary: dict[str, Any]) -> None:
    if output_path is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"GB10 release evidence summary written to {path}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    nvfp4_report, nvfp4_error = _load_report(args.gb10_nvfp4_report_json)
    openai_report, openai_error = _load_report(args.gb10_openai_report_json)
    summary = _build_summary(
        nvfp4_report=nvfp4_report,
        nvfp4_error=nvfp4_error,
        openai_report=openai_report,
        openai_error=openai_error,
        require_moe=args.gb10_require_moe,
        require_openai_deterministic=args.gb10_require_openai_deterministic,
        allow_partial=args.gb10_allow_partial,
    )
    _write_summary(args.gb10_output_json, summary)
    if summary["release_gate_passed"] or args.gb10_allow_partial:
        return 0

    print("GB10 release evidence gate failed", file=sys.stderr)
    for failure in summary["failures"]:
        print(f" - {failure['name']}: {failure['message']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
