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
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from gb10_release_contract import (
    FLASHINFER_RUNTIME_DISTRIBUTIONS,
    GB10_DEFERRED_PATH_REASONS,
    GB10_NOT_SUPPORTED_PATH_REASONS,
    RELEASE_NVFP4_SMOKE_REPORT_FILE,
    RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE,
    REQUIRED_GB10_SUPPORT_MATRIX,
    REQUIRED_SOURCE_DEPENDENCIES,
    SHA256_DIGEST_RE,
)


def _is_gb10_cuda13_version(value: Any) -> bool:
    return isinstance(value, str) and "+cu13" in value and "gb10" in value


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_observed_model_shape(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("status") != "observed":
        return False
    return all(
        _is_positive_int(value.get(field))
        for field in (
            "max_model_len",
            "head_size",
            "num_attention_heads",
            "num_kv_heads",
        )
    )


def _is_passed_cuda_graph_check(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("status") != "passed":
        return False
    return (
        value.get("configured_enabled") is True
        and _is_positive_int(value.get("num_cudagraph_captured"))
        and _is_positive_int(value.get("num_cudagraph_replayed"))
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify GB10 release evidence from the offline NVFP4 model smoke "
            "and OpenAI-compatible server smoke JSON reports."
        )
    )
    parser.add_argument(
        "--gb10-nvfp4-report-json",
        help=(
            f"Path to {RELEASE_NVFP4_SMOKE_REPORT_FILE} from "
            "scripts/gb10-smoke-nvfp4.py."
        ),
    )
    parser.add_argument(
        "--gb10-openai-report-json",
        help=(
            f"Path to {RELEASE_OPENAI_SERVER_SMOKE_REPORT_FILE} from "
            "scripts/gb10-smoke-openai-server.py."
        ),
    )
    parser.add_argument(
        "--gb10-release-manifest-json",
        help=(
            "Optional gb10-release-manifest.json from "
            "scripts/gb10-write-release-manifest.py. When provided, release "
            "provenance becomes part of the gate."
        ),
    )
    parser.add_argument(
        "--gb10-runtime-image-metadata-json",
        help=(
            "Optional BuildKit runtime-image metadata JSON. When provided, "
            "the runtime image digest recorded by the release build becomes "
            "part of the gate."
        ),
    )
    parser.add_argument(
        "--gb10-image-ref",
        help=(
            "Optional runtime image ref being smoked. When provided with a "
            "release manifest, it must match image.name:image.tag."
        ),
    )
    parser.add_argument(
        "--gb10-image-digest",
        help=(
            "Optional immutable digest for the runtime image that was pulled "
            "and smoked, for example ghcr.io/gardner/vllm-gb10@sha256:..."
        ),
    )
    parser.add_argument(
        "--gb10-release-tag",
        help=(
            "Optional release tag receiving the smoke evidence. When provided "
            "with a release manifest, it must match release.tag."
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
    runtime = report.get("runtime")
    device_capability = _nested_get(report, "runtime", "device_capability")
    device_major = _nested_get(report, "runtime", "device_capability", "major")
    device_minor = _nested_get(report, "runtime", "device_capability", "minor")
    device_arch = _nested_get(report, "runtime", "device_capability", "arch")
    flashinfer_version = _nested_get(report, "runtime", "flashinfer_version")
    flashinfer_distribution_versions = {
        distribution: _nested_get(
            report,
            "runtime",
            "flashinfer_distributions",
            distribution,
        )
        for distribution in FLASHINFER_RUNTIME_DISTRIBUTIONS
    }
    attention_backend_check = _nested_get(
        report,
        "gb10_release_summary",
        "checks",
        "attention_backend",
    )
    cuda_graph_check = _nested_get(
        report,
        "gb10_release_summary",
        "checks",
        "cuda_graph",
    )
    model_shape_check = _nested_get(
        report,
        "gb10_release_summary",
        "checks",
        "model_shape",
    )
    quantization_check = _nested_get(
        report,
        "gb10_release_summary",
        "checks",
        "quantization",
    )

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
            name="gb10_device_sm121",
            passed=_nested_get(report, "runtime", "cuda_available") is True
            and device_major == 12
            and device_minor == 1
            and device_arch == "sm_121",
            message="offline NVFP4 smoke ran on GB10 / SM121",
            details={
                "runtime": runtime if isinstance(runtime, dict) else {},
                "device_capability": device_capability
                if isinstance(device_capability, dict)
                else {},
            },
        ),
        _check(
            name="flashinfer_gb10_runtime_version",
            passed=_is_gb10_cuda13_version(flashinfer_version),
            message=(
                "offline NVFP4 smoke imported the GB10 CUDA 13 FlashInfer "
                "runtime package"
            ),
            details={"flashinfer_version": flashinfer_version},
        ),
        _check(
            name="flashinfer_gb10_distribution_versions",
            passed=all(
                _is_gb10_cuda13_version(version)
                for version in flashinfer_distribution_versions.values()
            ),
            message=(
                "offline NVFP4 smoke installed the GB10 CUDA 13 FlashInfer "
                "Python, cubin, and JIT-cache packages"
            ),
            details={"flashinfer_distributions": flashinfer_distribution_versions},
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
        _check(
            name="cuda_graph_capture_replay",
            passed=_is_passed_cuda_graph_check(cuda_graph_check),
            message="offline smoke observed CUDA graph capture and replay",
            details=cuda_graph_check if isinstance(cuda_graph_check, dict) else {},
        ),
        _check(
            name="model_shape_reported",
            passed=_is_observed_model_shape(model_shape_check),
            message="offline smoke reported complete target model shape metadata",
            details=model_shape_check if isinstance(model_shape_check, dict) else {},
        ),
        _check(
            name="attention_backend_flashinfer",
            passed=isinstance(attention_backend_check, dict)
            and attention_backend_check.get("status") == "passed"
            and attention_backend_check.get("expected") == "FLASHINFER"
            and attention_backend_check.get("requested_backend") == "FLASHINFER",
            message="offline smoke observed requested FlashInfer attention backend",
            details=attention_backend_check
            if isinstance(attention_backend_check, dict)
            else {},
        ),
        _check(
            name="quantization_modelopt_fp4",
            passed=isinstance(quantization_check, dict)
            and quantization_check.get("status") == "passed"
            and quantization_check.get("expected") == "modelopt_fp4"
            and quantization_check.get("configured") == "modelopt_fp4",
            message="offline smoke observed requested ModelOpt FP4 quantization",
            details=quantization_check if isinstance(quantization_check, dict) else {},
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


def _source_refs_pinned(manifest: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        return False, {"reason": "dependencies object missing"}

    source_dependencies = dependencies.get("source_dependencies")
    if not isinstance(source_dependencies, dict):
        return False, {"reason": "source_dependencies object missing"}

    refs: dict[str, Any] = {}
    for name, dependency in source_dependencies.items():
        if isinstance(dependency, dict):
            refs[name] = {
                "ref": dependency.get("ref"),
                "ref_is_full_git_sha": dependency.get("ref_is_full_git_sha"),
            }
        else:
            refs[name] = {"ref": None, "ref_is_full_git_sha": False}

    flash_attn = dependencies.get("vllm_flash_attn")
    if isinstance(flash_attn, dict):
        refs["vllm_flash_attn"] = {
            "ref": flash_attn.get("ref"),
            "ref_is_full_git_sha": flash_attn.get("ref_is_full_git_sha"),
        }
    else:
        refs["vllm_flash_attn"] = {"ref": None, "ref_is_full_git_sha": False}

    missing = [
        name
        for name, details in refs.items()
        if details.get("ref_is_full_git_sha") is not True
    ]
    return not missing, {"refs": refs, "unpinned": missing}


def _source_dependencies_present(
    manifest: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        return False, {
            "required": list(REQUIRED_SOURCE_DEPENDENCIES),
            "present": [],
            "missing": list(REQUIRED_SOURCE_DEPENDENCIES),
            "reason": "dependencies object missing",
        }

    source_dependencies = dependencies.get("source_dependencies")
    if not isinstance(source_dependencies, dict):
        return False, {
            "required": list(REQUIRED_SOURCE_DEPENDENCIES),
            "present": [],
            "missing": list(REQUIRED_SOURCE_DEPENDENCIES),
            "reason": "source_dependencies object missing",
        }

    present = sorted(
        dependency
        for dependency in REQUIRED_SOURCE_DEPENDENCIES
        if dependency in source_dependencies
    )
    missing = sorted(set(REQUIRED_SOURCE_DEPENDENCIES) - set(present))
    return not missing, {
        "required": list(REQUIRED_SOURCE_DEPENDENCIES),
        "present": present,
        "missing": missing,
    }


def _gb10_support_matrix_present(
    manifest: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    support_matrix = manifest.get("gb10_support_matrix")
    if not isinstance(support_matrix, dict):
        return False, {
            "required": REQUIRED_GB10_SUPPORT_MATRIX,
            "present": {},
            "missing": sorted(REQUIRED_GB10_SUPPORT_MATRIX),
            "mismatched": {},
            "reason": "gb10_support_matrix object missing",
        }

    entries = support_matrix.get("entries")
    if not isinstance(entries, dict):
        return False, {
            "required": REQUIRED_GB10_SUPPORT_MATRIX,
            "present": {},
            "missing": sorted(REQUIRED_GB10_SUPPORT_MATRIX),
            "mismatched": {},
            "reason": "gb10_support_matrix.entries object missing",
        }

    present = {
        name: entry.get("status") if isinstance(entry, dict) else None
        for name, entry in entries.items()
        if name in REQUIRED_GB10_SUPPORT_MATRIX
    }
    missing = sorted(set(REQUIRED_GB10_SUPPORT_MATRIX) - set(present))
    mismatched = {
        name: {
            "expected": expected_status,
            "actual": present.get(name),
        }
        for name, expected_status in REQUIRED_GB10_SUPPORT_MATRIX.items()
        if name in present and present.get(name) != expected_status
    }
    return not missing and not mismatched, {
        "architecture": support_matrix.get("architecture"),
        "first_release_scope": support_matrix.get("first_release_scope"),
        "required": REQUIRED_GB10_SUPPORT_MATRIX,
        "present": present,
        "missing": missing,
        "mismatched": mismatched,
    }


def _support_matrix_entry_status(
    manifest: dict[str, Any] | None,
    entry_name: str,
) -> str | None:
    entries = _nested_get(manifest, "gb10_support_matrix", "entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(entry_name)
    if not isinstance(entry, dict):
        return None
    status = entry.get("status")
    return status if isinstance(status, str) else None


def _support_matrix_entry_for_selection(selection: Any) -> str | None:
    if not isinstance(selection, dict):
        return None

    path = selection.get("path")
    backend = selection.get("backend")
    path_name = str(path).lower() if path is not None else ""
    backend_name = str(backend).replace("-", "_").upper() if backend is not None else ""

    if "MARLIN" in backend_name or "EMULATION" in backend_name:
        return "marlin_nvfp4_fallback"

    trtllm_gen_markers = ("TRTLLM_GEN", "TRTLLMGEN", "TRT_LLM_GEN")
    if any(marker in backend_name for marker in trtllm_gen_markers):
        if path_name == "moe":
            return "trtllm_gen_moe"
        return "trtllm_gen_attention"

    if path_name in {"linear", "linear_w4a16"} and "FLASHINFER" in backend_name:
        return "flashinfer_nvfp4_dense"

    if path_name == "moe" and "FLASHINFER_B12X" in backend_name:
        return "flashinfer_b12x_non_ep_moe"

    return None


def _normalize_backend_marker(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("-", "_").replace(" ", "_").upper()


def _support_matrix_entry_for_attention_backend(backend: Any) -> str | None:
    backend_name = _normalize_backend_marker(backend)
    if not backend_name:
        return None

    if "FLASHINFER" in backend_name:
        return "flashinfer_attention_fa2"
    if "FLASHMLA" in backend_name or backend_name == "MLA":
        return "flashmla_attention"
    if "FLASH_ATTN" in backend_name or "FLASHATTN" in backend_name:
        return "public_flashattention_runtime"
    trtllm_markers = ("TRTLLM", "TRT_LLM", "TRTLLM_GEN", "TRT_LLM_GEN")
    if any(marker in backend_name for marker in trtllm_markers):
        return "trtllm_gen_attention"
    return None


def _check_attention_backend_against_support_matrix(
    report: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    *,
    required: bool,
) -> list[dict[str, Any]]:
    if manifest is None:
        return [
            {
                "name": "attention_backend_allowed_by_support_matrix",
                "status": "not_required",
                "required": False,
                "message": (
                    "release manifest support matrix was not provided for "
                    "attention backend classification"
                ),
                "details": {},
            }
        ]
    if report is None:
        return [
            _missing_check(
                "attention_backend_allowed_by_support_matrix",
                "NVFP4 report missing, so attention backend cannot be checked "
                "against the support matrix",
                required=required,
            )
        ]

    attention_backend = _nested_get(
        report,
        "gb10_release_summary",
        "checks",
        "attention_backend",
    )
    if not isinstance(attention_backend, dict):
        return [
            _check(
                name="attention_backend_allowed_by_support_matrix",
                passed=False,
                message=(
                    "NVFP4 report does not include attention backend evidence "
                    "to check against the support matrix"
                ),
                details={},
                required=required,
            )
        ]

    candidates = {
        "requested_backend": attention_backend.get("requested_backend"),
        "mla_prefill_backend": attention_backend.get("mla_prefill_backend"),
    }
    classified: list[dict[str, Any]] = []
    unclassified: dict[str, Any] = {}
    disallowed: list[dict[str, Any]] = []
    allowed_statuses = {"supported_native", "supported_routed"}
    for source, backend in candidates.items():
        if backend in {None, ""}:
            continue
        entry_name = _support_matrix_entry_for_attention_backend(backend)
        if entry_name is None:
            unclassified[source] = backend
            continue
        status = _support_matrix_entry_status(manifest, entry_name)
        details = {
            "source": source,
            "backend": backend,
            "support_matrix_entry": entry_name,
            "support_matrix_status": status,
        }
        classified.append(details)
        if status not in allowed_statuses:
            disallowed.append(details)

    return [
        _check(
            name="attention_backend_allowed_by_support_matrix",
            passed=bool(classified) and not unclassified and not disallowed,
            message=(
                "attention backend evidence is classified by the GB10 support "
                "matrix and only uses supported native or routed entries"
            ),
            details={
                "attention_backend": attention_backend,
                "classified": classified,
                "unclassified": unclassified,
                "disallowed": disallowed,
            },
            required=required,
        )
    ]


def _check_quantization_against_support_matrix(
    report: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    *,
    required: bool,
) -> list[dict[str, Any]]:
    if manifest is None:
        return [
            {
                "name": "quantization_allowed_by_support_matrix",
                "status": "not_required",
                "required": False,
                "message": (
                    "release manifest support matrix was not provided for "
                    "quantization classification"
                ),
                "details": {},
            }
        ]
    if report is None:
        return [
            _missing_check(
                "quantization_allowed_by_support_matrix",
                "NVFP4 report missing, so quantization cannot be checked "
                "against the support matrix",
                required=required,
            )
        ]

    quantization = _nested_get(
        report,
        "gb10_release_summary",
        "checks",
        "quantization",
    )
    if not isinstance(quantization, dict):
        return [
            _check(
                name="quantization_allowed_by_support_matrix",
                passed=False,
                message=(
                    "NVFP4 report does not include quantization evidence "
                    "to check against the support matrix"
                ),
                details={},
                required=required,
            )
        ]

    configured = quantization.get("configured")
    expected = quantization.get("expected")
    status = quantization.get("status")
    configured_name = str(configured).lower() if configured is not None else ""
    expected_name = str(expected).lower() if expected is not None else ""
    entry_name = (
        "modelopt_fp4_quantization"
        if configured_name == "modelopt_fp4" and expected_name == "modelopt_fp4"
        else None
    )
    support_status = (
        _support_matrix_entry_status(manifest, entry_name)
        if entry_name is not None
        else None
    )
    allowed_statuses = {"supported_native", "supported_routed"}
    return [
        _check(
            name="quantization_allowed_by_support_matrix",
            passed=(
                status == "passed"
                and entry_name is not None
                and support_status in allowed_statuses
            ),
            message=(
                "quantization evidence is classified by the GB10 support "
                "matrix and only uses supported native or routed entries"
            ),
            details={
                "quantization": quantization,
                "support_matrix_entry": entry_name,
                "support_matrix_status": support_status,
            },
            required=required,
        )
    ]


def _check_path_statuses_reported(
    report: dict[str, Any] | None,
    *,
    report_key: str,
    check_name: str,
    reasons: dict[str, str],
    expected_status: str,
    expected_handling: str,
    details_prefix: str,
    message_status: str,
) -> list[dict[str, Any]]:
    paths = _nested_get(
        report,
        "gb10_release_summary",
        report_key,
    )
    required_entries = sorted(reasons)
    if not isinstance(paths, dict):
        return [
            _check(
                name=check_name,
                passed=False,
                message=(
                    f"offline smoke report lists {message_status} GB10 paths and "
                    "their expected handling"
                ),
                details={
                    f"required_{details_prefix}_entries": required_entries,
                    f"reported_{details_prefix}_entries": [],
                    f"missing_{details_prefix}_entries": required_entries,
                    f"malformed_{details_prefix}_entries": required_entries,
                },
            )
        ]

    reported_entries: list[str] = []
    missing_entries: list[str] = []
    malformed_entries: list[str] = []
    for entry_name in required_entries:
        entry = paths.get(entry_name)
        if not isinstance(entry, dict):
            missing_entries.append(entry_name)
            malformed_entries.append(entry_name)
            continue
        reported_entries.append(entry_name)
        reason = entry.get("reason")
        if (
            entry.get("status") != expected_status
            or entry.get("expected_handling") != expected_handling
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            malformed_entries.append(entry_name)

    return [
        _check(
            name=check_name,
            passed=not missing_entries and not malformed_entries,
            message=(
                f"offline smoke report lists {message_status} GB10 paths and "
                "their expected handling"
            ),
            details={
                f"required_{details_prefix}_entries": required_entries,
                f"reported_{details_prefix}_entries": reported_entries,
                f"missing_{details_prefix}_entries": missing_entries,
                f"malformed_{details_prefix}_entries": malformed_entries,
            },
        )
    ]


def _check_unsupported_paths_reported(
    report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return _check_path_statuses_reported(
        report,
        report_key="unsupported_paths",
        check_name="unsupported_paths_reported",
        reasons=GB10_NOT_SUPPORTED_PATH_REASONS,
        expected_status="not_supported",
        expected_handling="route_or_reject_before_release_evidence",
        details_prefix="not_supported",
        message_status="Not Supported",
    )


def _check_deferred_paths_reported(
    report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return _check_path_statuses_reported(
        report,
        report_key="deferred_paths",
        check_name="deferred_paths_reported",
        reasons=GB10_DEFERRED_PATH_REASONS,
        expected_status="deferred",
        expected_handling="block_until_hardware_validated",
        details_prefix="deferred",
        message_status="Deferred",
    )


def _check_backend_selections_against_support_matrix(
    report: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    *,
    required: bool,
) -> list[dict[str, Any]]:
    if manifest is None:
        return [
            {
                "name": "nvfp4_backend_selections_allowed_by_support_matrix",
                "status": "not_required",
                "required": False,
                "message": (
                    "release manifest support matrix was not provided for "
                    "backend-selection classification"
                ),
                "details": {},
            }
        ]
    if report is None:
        return [
            _missing_check(
                "nvfp4_backend_selections_allowed_by_support_matrix",
                "NVFP4 report missing, so backend selections cannot be checked "
                "against the support matrix",
                required=required,
            )
        ]

    backend_selections = report.get("backend_selections")
    if not isinstance(backend_selections, list):
        return [
            _check(
                name="nvfp4_backend_selections_allowed_by_support_matrix",
                passed=False,
                message=(
                    "NVFP4 report backend selections must be a list before "
                    "they can be checked against the support matrix"
                ),
                details={"backend_selections": backend_selections},
                required=required,
            )
        ]

    classified: list[dict[str, Any]] = []
    unclassified: list[dict[str, Any]] = []
    disallowed: list[dict[str, Any]] = []
    allowed_statuses = {"supported_native", "supported_routed"}
    for selection in backend_selections:
        entry_name = _support_matrix_entry_for_selection(selection)
        if entry_name is None:
            if isinstance(selection, dict):
                unclassified.append(selection)
            else:
                unclassified.append({"selection": selection})
            continue

        status = _support_matrix_entry_status(manifest, entry_name)
        selection_details = {
            "path": selection.get("path") if isinstance(selection, dict) else None,
            "backend": selection.get("backend")
            if isinstance(selection, dict)
            else None,
            "is_fallback": selection.get("is_fallback")
            if isinstance(selection, dict)
            else None,
            "support_matrix_entry": entry_name,
            "support_matrix_status": status,
        }
        classified.append(selection_details)
        if status not in allowed_statuses or selection_details["is_fallback"] is True:
            disallowed.append(selection_details)

    return [
        _check(
            name="nvfp4_backend_selections_allowed_by_support_matrix",
            passed=not unclassified and not disallowed,
            message=(
                "NVFP4 backend selections are classified by the GB10 support "
                "matrix and only use supported native or routed entries"
            ),
            details={
                "classified": classified,
                "unclassified": unclassified,
                "disallowed": disallowed,
            },
            required=required,
        )
    ]


def _release_manifest_validation_errors(manifest: dict[str, Any]) -> list[str]:
    manifest_writer_path = Path(__file__).with_name("gb10-write-release-manifest.py")
    spec = importlib.util.spec_from_file_location(
        "gb10_write_release_manifest",
        manifest_writer_path,
    )
    if spec is None or spec.loader is None:
        return [f"could not load manifest validator: {manifest_writer_path}"]

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validate_manifest = getattr(module, "validate_manifest", None)
    if validate_manifest is None:
        return [f"manifest validator missing from {manifest_writer_path}"]
    return list(validate_manifest(manifest))


def _normalize_image_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidates = [line.strip() for line in value.splitlines() if line.strip()]
    if not candidates:
        return None
    digest = candidates[0]
    if "@" in digest:
        digest = digest.rsplit("@", 1)[1].strip()
    marker = "sha256:"
    marker_index = digest.find(marker)
    if marker_index >= 0:
        digest = digest[marker_index:]
    digest = digest.split()[0].strip().strip('",')
    if SHA256_DIGEST_RE.fullmatch(digest) is None:
        return None
    return digest


def _runtime_image_metadata_digest(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None

    candidates: list[Any] = [
        metadata.get("containerimage.digest"),
        _nested_get(metadata, "containerimage.descriptor", "digest"),
        _nested_get(metadata, "image", "digest"),
    ]
    for candidate in candidates:
        digest = _normalize_image_digest(candidate)
        if digest:
            return digest
    return None


def _check_runtime_image_metadata(
    metadata: dict[str, Any] | None,
    error: str | None,
    *,
    required: bool,
    image_digest: str | None,
) -> list[dict[str, Any]]:
    if metadata is None:
        if required:
            return [
                _missing_check(
                    "runtime_image_metadata_present",
                    error or "runtime image metadata missing",
                )
            ]
        return [
            {
                "name": "runtime_image_metadata_present",
                "status": "not_required",
                "required": False,
                "message": "runtime image metadata was not provided",
                "details": {"image_digest": image_digest},
            }
        ]

    metadata_digest = _runtime_image_metadata_digest(metadata)
    smoked_digest = _normalize_image_digest(image_digest)
    checks = [
        _check(
            name="runtime_image_metadata_present",
            passed=True,
            message="runtime image metadata is valid JSON",
            details={"metadata_keys": sorted(metadata.keys())},
        ),
        _check(
            name="runtime_image_metadata_has_digest",
            passed=metadata_digest is not None,
            message="runtime image metadata records the pushed image digest",
            details={"runtime_image_metadata_digest": metadata_digest},
        ),
    ]

    if image_digest is None:
        checks.append(
            _check(
                name="runtime_image_digest_matches_smoke",
                passed=False,
                message=(
                    "runtime image metadata was provided, but the smoked "
                    "image digest was not recorded"
                ),
                details={"runtime_image_metadata_digest": metadata_digest},
                required=required,
            )
        )
    else:
        checks.append(
            _check(
                name="runtime_image_digest_matches_smoke",
                passed=(
                    metadata_digest is not None
                    and smoked_digest is not None
                    and smoked_digest == metadata_digest
                ),
                message=(
                    "runtime image digest from BuildKit metadata matches the "
                    "image digest that was pulled and smoked"
                ),
                details={
                    "runtime_image_metadata_digest": metadata_digest,
                    "smoked_image_digest": smoked_digest,
                    "smoked_image_digest_raw": image_digest,
                },
                required=required,
            )
        )

    return checks


def _check_release_manifest(
    manifest: dict[str, Any] | None,
    error: str | None,
    *,
    required: bool,
    image_ref: str | None,
    expected_release_tag: str | None,
) -> list[dict[str, Any]]:
    if manifest is None:
        if required:
            return [
                _missing_check(
                    "release_manifest_present",
                    error or "release manifest missing",
                )
            ]
        return [
            {
                "name": "release_manifest_present",
                "status": "not_required",
                "required": False,
                "message": "release manifest was not provided",
                "details": {},
            }
        ]

    flashinfer = _nested_get(manifest, "dependencies", "flashinfer")
    flashinfer_components_present = (
        isinstance(flashinfer, dict)
        and flashinfer.get("all_required_components_present") is True
    )
    source_dependencies_present, source_dependencies_details = (
        _source_dependencies_present(manifest)
    )
    support_matrix_present, support_matrix_details = _gb10_support_matrix_present(
        manifest
    )
    source_refs_pinned, source_refs_details = _source_refs_pinned(manifest)
    manifest_validation_errors = _release_manifest_validation_errors(manifest)
    release_tag = _nested_get(manifest, "release", "tag")
    preflight_only = _nested_get(manifest, "release", "preflight_only") is True
    image_push = _nested_get(manifest, "image", "push") is True
    manifest_image_name = _nested_get(manifest, "image", "name")
    manifest_image_tag = _nested_get(manifest, "image", "tag")
    manifest_image_ref = (
        f"{manifest_image_name}:{manifest_image_tag}"
        if manifest_image_name and manifest_image_tag
        else None
    )
    tagged_full_release = bool(release_tag) and not preflight_only

    return [
        _check(
            name="release_manifest_present",
            passed=True,
            message="release manifest is valid JSON",
            details={"schema_version": manifest.get("schema_version")},
        ),
        _check(
            name="release_manifest_flashinfer_components",
            passed=flashinfer_components_present,
            message="release manifest includes all required GB10 FlashInfer wheels",
            details=flashinfer if isinstance(flashinfer, dict) else {},
        ),
        _check(
            name="release_manifest_durable_inputs",
            passed=not manifest_validation_errors,
            message=(
                "release manifest records durable GB10 inputs: GitHub Release "
                "FlashInfer wheels, full-SHA source refs, no local dependency "
                "checkouts, and pushed images for tagged full releases"
            ),
            details={"errors": manifest_validation_errors},
        ),
        _check(
            name="release_manifest_source_dependencies_present",
            passed=source_dependencies_present,
            message=(
                "release manifest includes the required GB10 source-built "
                "dependencies"
            ),
            details=source_dependencies_details,
        ),
        _check(
            name="release_manifest_gb10_support_matrix",
            passed=support_matrix_present,
            message=(
                "release manifest records the GB10 backend support matrix for "
                "native, routed, unsupported, and deferred paths"
            ),
            details=support_matrix_details,
        ),
        _check(
            name="release_manifest_source_refs_pinned",
            passed=source_refs_pinned,
            message=(
                "release manifest records pinned source refs for GB10 "
                "source-built dependencies"
            ),
            details=source_refs_details,
        ),
        _check(
            name="release_manifest_no_local_deps",
            passed=_nested_get(
                manifest,
                "build",
                "local_gb10_dependency_checkouts",
            )
            is False,
            message="release build did not depend on local GB10 sibling checkouts",
            details={
                "local_gb10_dependency_checkouts": _nested_get(
                    manifest,
                    "build",
                    "local_gb10_dependency_checkouts",
                )
            },
        ),
        _check(
            name="release_manifest_image_pushed_for_tagged_release",
            passed=not tagged_full_release or image_push,
            message=(
                "tagged full release manifest records a pushed durable "
                "runtime image"
            ),
            details={
                "release_tag": release_tag,
                "preflight_only": preflight_only,
                "image_push": image_push,
            },
        ),
        _check(
            name="release_manifest_image_ref_matches_smoke",
            passed=image_ref is None or image_ref == manifest_image_ref,
            message=(
                "release manifest image ref matches the runtime image ref "
                "being smoked"
            ),
            details={
                "image_ref": image_ref,
                "manifest_image_ref": manifest_image_ref,
            },
            required=image_ref is not None,
        ),
        _check(
            name="release_manifest_tag_matches_expected",
            passed=expected_release_tag is None or expected_release_tag == release_tag,
            message="release manifest tag matches the release receiving evidence",
            details={
                "release_tag": expected_release_tag,
                "manifest_release_tag": release_tag,
            },
            required=expected_release_tag is not None,
        ),
    ]


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
    release_manifest: dict[str, Any] | None = None,
    release_manifest_error: str | None = None,
    runtime_image_metadata: dict[str, Any] | None = None,
    runtime_image_metadata_error: str | None = None,
    image_ref: str | None = None,
    image_digest: str | None = None,
    release_tag: str | None = None,
    require_release_manifest: bool = False,
    require_runtime_image_metadata: bool = False,
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
        *_check_release_manifest(
            release_manifest,
            release_manifest_error,
            required=require_release_manifest,
            image_ref=image_ref,
            expected_release_tag=release_tag,
        ),
        *_check_attention_backend_against_support_matrix(
            nvfp4_report,
            release_manifest,
            required=require_release_manifest,
        ),
        *_check_quantization_against_support_matrix(
            nvfp4_report,
            release_manifest,
            required=require_release_manifest,
        ),
        *_check_unsupported_paths_reported(nvfp4_report),
        *_check_deferred_paths_reported(nvfp4_report),
        *_check_backend_selections_against_support_matrix(
            nvfp4_report,
            release_manifest,
            required=require_release_manifest,
        ),
        *_check_runtime_image_metadata(
            runtime_image_metadata,
            runtime_image_metadata_error,
            required=require_runtime_image_metadata,
            image_digest=image_digest,
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
            "require_release_manifest": require_release_manifest,
            "require_runtime_image_metadata": require_runtime_image_metadata,
            "image_ref": image_ref,
            "image_digest": image_digest,
            "release_tag": release_tag,
        },
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "remaining_release_evidence": [
            "final runtime image smoke with the published GB10 dependency wheels",
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
    release_manifest, release_manifest_error = _load_report(
        args.gb10_release_manifest_json
    )
    runtime_image_metadata, runtime_image_metadata_error = _load_report(
        args.gb10_runtime_image_metadata_json
    )
    summary = _build_summary(
        nvfp4_report=nvfp4_report,
        nvfp4_error=nvfp4_error,
        openai_report=openai_report,
        openai_error=openai_error,
        release_manifest=release_manifest,
        release_manifest_error=release_manifest_error,
        runtime_image_metadata=runtime_image_metadata,
        runtime_image_metadata_error=runtime_image_metadata_error,
        image_ref=args.gb10_image_ref,
        image_digest=args.gb10_image_digest,
        release_tag=args.gb10_release_tag,
        require_release_manifest=args.gb10_release_manifest_json is not None,
        require_runtime_image_metadata=(
            args.gb10_runtime_image_metadata_json is not None
        ),
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
