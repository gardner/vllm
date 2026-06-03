#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke a GB10 NVFP4 model and assert native backend selection.

This script is intentionally small and release-oriented: it starts an offline
vLLM engine, runs one short generation by default, and then checks the
process-local NVFP4 backend recorder. It defaults to single-process V1 engine
execution so backend-selection events recorded during model initialization are
visible to the assertion code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = "NVIDIA DGX Spark native NVFP4 support means"
DEFAULT_REQUIRED_PATHS = ("linear",)


def _parse_backend_expectation(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "expected PATH=BACKEND_SUBSTRING, for example "
            "linear=FlashInferB12x"
        )
    path, backend_substring = value.split("=", 1)
    if not path or not backend_substring:
        raise argparse.ArgumentTypeError(
            "expected non-empty PATH and BACKEND_SUBSTRING"
        )
    return path, backend_substring


def _preparse_allow_fallback(argv: Sequence[str] | None) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gb10-allow-fallback", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return bool(args.gb10_allow_fallback)


def _configure_env(*, allow_fallback: bool) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

    if allow_fallback:
        os.environ["VLLM_FAIL_ON_NVFP4_FALLBACK"] = "0"
    else:
        os.environ["VLLM_FAIL_ON_NVFP4_FALLBACK"] = "1"


def _build_parser(engine_args_cls: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a GB10 NVFP4 vLLM smoke test and fail if native backend "
            "selection evidence is missing or fallback was selected."
        )
    )
    parser.add_argument(
        "--gb10-prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for the smoke generation.",
    )
    parser.add_argument(
        "--gb10-max-tokens",
        type=int,
        default=8,
        help="Maximum generated tokens for the smoke request.",
    )
    parser.add_argument(
        "--gb10-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the smoke request.",
    )
    parser.add_argument(
        "--gb10-sampling-seed",
        type=int,
        default=0,
        help="Sampling seed for the smoke request.",
    )
    parser.add_argument(
        "--gb10-require-path",
        action="append",
        choices=("linear", "linear_w4a16", "moe"),
        help=(
            "Require at least one NVFP4 backend-selection event for this "
            "path. Defaults to linear. Pass again for moe on MoE models."
        ),
    )
    parser.add_argument(
        "--gb10-expect-backend",
        action="append",
        default=[],
        type=_parse_backend_expectation,
        metavar="PATH=BACKEND_SUBSTRING",
        help=(
            "Require a selected backend for PATH to contain BACKEND_SUBSTRING, "
            "for example linear=FlashInferB12x or moe=FLASHINFER_B12X."
        ),
    )
    parser.add_argument(
        "--gb10-allow-fallback",
        action="store_true",
        help=(
            "Allow fallback events. This is useful only while debugging; "
            "release smoke tests should omit it."
        ),
    )
    parser.add_argument(
        "--gb10-skip-generate",
        action="store_true",
        help="Only initialize the model and assert backend selections.",
    )
    parser.add_argument(
        "--gb10-report-json",
        help=(
            "Optional path for a JSON report containing runtime metadata, "
            "backend selections, fallback events, and generated text."
        ),
    )

    parser = engine_args_cls.add_cli_args(parser)
    parser.set_defaults(
        model=os.environ.get("GB10_NVFP4_MODEL"),
        quantization="modelopt_fp4",
        kv_cache_dtype="fp8_e4m3",
        enable_prefix_caching=False,
    )
    return parser


def _format_events(events: Iterable[Any]) -> str:
    lines = []
    for event in events:
        fields = [f"path={event.path}", f"backend={event.backend}"]
        if hasattr(event, "is_fallback"):
            fields.append(f"is_fallback={event.is_fallback}")
        if hasattr(event, "message"):
            fields.append(f"message={event.message}")
        lines.append("  - " + ", ".join(fields))
    return "\n".join(lines) if lines else "  - none"


def _print_generation_outputs(outputs: Sequence[Any]) -> None:
    for idx, output in enumerate(outputs):
        completions = getattr(output, "outputs", [])
        text = completions[0].text if completions else ""
        print(f"GB10 smoke output[{idx}]: {text!r}")


def _print_event_summary(selections: Sequence[Any], fallbacks: Sequence[Any]) -> None:
    print("GB10 NVFP4 backend selections:")
    print(_format_events(selections))
    print("GB10 NVFP4 fallback events:")
    print(_format_events(fallbacks))


def _events_to_dicts(events: Sequence[Any]) -> list[dict[str, Any]]:
    result = []
    for event in events:
        event_dict = {
            "path": event.path,
            "backend": event.backend,
        }
        if hasattr(event, "is_fallback"):
            event_dict["is_fallback"] = event.is_fallback
        if hasattr(event, "message"):
            event_dict["message"] = event.message
        result.append(event_dict)
    return result


def _collect_runtime_metadata() -> dict[str, Any]:
    import torch

    from vllm.version import __version__ as vllm_version

    metadata: dict[str, Any] = {
        "vllm_version": vllm_version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "env": {
            "VLLM_FAIL_ON_NVFP4_FALLBACK": os.environ.get(
                "VLLM_FAIL_ON_NVFP4_FALLBACK"
            ),
            "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ),
            "VLLM_NO_USAGE_STATS": os.environ.get("VLLM_NO_USAGE_STATS"),
        },
    }
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        metadata["device_name"] = torch.cuda.get_device_name()
        metadata["device_capability"] = {
            "major": capability[0],
            "minor": capability[1],
            "arch": f"sm_{capability[0]}{capability[1]}",
        }

    try:
        import flashinfer  # type: ignore[import-untyped]
    except ImportError:
        metadata["flashinfer_version"] = None
    else:
        metadata["flashinfer_version"] = getattr(flashinfer, "__version__", None)

    return metadata


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _summarize_int_list(values: Sequence[int] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    values = list(values)
    return {
        "count": len(values),
        "first": values[:8],
        "last": values[-8:],
        "max": max(values) if values else None,
    }


def _call_or_none(obj: Any, method_name: str, *args: Any) -> Any:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method(*args)
    except Exception:
        return None


def _collect_vllm_config_summary(llm: Any | None) -> dict[str, Any] | None:
    llm_engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(llm_engine, "vllm_config", None)
    if vllm_config is None:
        return None

    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    attention_config = getattr(vllm_config, "attention_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    observability_config = getattr(vllm_config, "observability_config", None)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)

    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    cudagraph_mode_name = _json_value(cudagraph_mode)
    cudagraph_enabled = (
        cudagraph_mode_name is not None
        and not str(cudagraph_mode_name).upper().endswith("NONE")
    )

    return {
        "model": {
            "dtype": _json_value(getattr(model_config, "dtype", None)),
            "quantization": _json_value(getattr(model_config, "quantization", None)),
            "max_model_len": getattr(model_config, "max_model_len", None),
            "enforce_eager": getattr(model_config, "enforce_eager", None),
            "use_mla": getattr(model_config, "use_mla", None),
            "is_attention_free": getattr(model_config, "is_attention_free", None),
            "head_size": _call_or_none(model_config, "get_head_size"),
            "num_attention_heads": _call_or_none(
                model_config,
                "get_num_attention_heads",
                parallel_config,
            ),
            "num_kv_heads": _call_or_none(
                model_config,
                "get_num_kv_heads",
                parallel_config,
            ),
        },
        "attention": {
            "requested_backend": _json_value(
                getattr(attention_config, "backend", None)
            ),
            "mla_prefill_backend": _json_value(
                getattr(attention_config, "mla_prefill_backend", None)
            ),
            "use_trtllm_attention": getattr(
                attention_config,
                "use_trtllm_attention",
                None,
            ),
            "use_prefill_query_quantization": getattr(
                attention_config,
                "use_prefill_query_quantization",
                None,
            ),
            "use_non_causal": getattr(attention_config, "use_non_causal", None),
        },
        "cache": {
            "cache_dtype": _json_value(getattr(cache_config, "cache_dtype", None)),
            "block_size": getattr(cache_config, "block_size", None),
            "enable_prefix_caching": getattr(
                cache_config,
                "enable_prefix_caching",
                None,
            ),
            "kv_cache_dtype_skip_layers": list(
                getattr(cache_config, "kv_cache_dtype_skip_layers", []) or []
            ),
            "mamba_cache_dtype": _json_value(
                getattr(cache_config, "mamba_cache_dtype", None)
            ),
            "mamba_ssm_cache_dtype": _json_value(
                getattr(cache_config, "mamba_ssm_cache_dtype", None)
            ),
        },
        "parallel": {
            "tensor_parallel_size": getattr(
                parallel_config,
                "tensor_parallel_size",
                None,
            ),
            "pipeline_parallel_size": getattr(
                parallel_config,
                "pipeline_parallel_size",
                None,
            ),
            "data_parallel_size": getattr(parallel_config, "data_parallel_size", None),
            "decode_context_parallel_size": getattr(
                parallel_config,
                "decode_context_parallel_size",
                None,
            ),
            "world_size": getattr(parallel_config, "world_size", None),
            "distributed_executor_backend": _json_value(
                getattr(parallel_config, "distributed_executor_backend", None)
            ),
        },
        "scheduler": {
            "max_num_seqs": getattr(scheduler_config, "max_num_seqs", None),
            "max_num_batched_tokens": getattr(
                scheduler_config,
                "max_num_batched_tokens",
                None,
            ),
            "enable_chunked_prefill": getattr(
                scheduler_config,
                "enable_chunked_prefill",
                None,
            ),
        },
        "compilation": {
            "cudagraph_mode": cudagraph_mode_name,
            "cudagraph_enabled": cudagraph_enabled,
            "max_cudagraph_capture_size": getattr(
                compilation_config,
                "max_cudagraph_capture_size",
                None,
            ),
            "cudagraph_capture_sizes": _summarize_int_list(
                getattr(compilation_config, "cudagraph_capture_sizes", None)
            ),
            "cudagraph_num_of_warmups": getattr(
                compilation_config,
                "cudagraph_num_of_warmups",
                None,
            ),
        },
        "observability": {
            "cudagraph_metrics": getattr(
                observability_config,
                "cudagraph_metrics",
                None,
            ),
        },
        "kv_transfer": {
            "enabled": kv_transfer_config is not None,
            "is_kv_transfer_instance": getattr(
                kv_transfer_config,
                "is_kv_transfer_instance",
                None,
            ),
        },
    }


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def _backend_status_for_path(
    path: str,
    selections: Sequence[Any],
    fallbacks: Sequence[Any],
    *,
    missing_status: str,
) -> str:
    path_selections = [event for event in selections if event.path == path]
    path_fallbacks = [event for event in fallbacks if event.path == path]
    if any(not event.is_fallback for event in path_selections):
        return "observed"
    if path_fallbacks or any(event.is_fallback for event in path_selections):
        return "fallback_observed"
    return missing_status


def _build_backend_summary(
    *,
    status: str,
    required_paths: Sequence[str],
    selections: Sequence[Any],
    fallbacks: Sequence[Any],
    vllm_config_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    paths = _unique_sorted(
        [
            *required_paths,
            *(event.path for event in selections),
            *(event.path for event in fallbacks),
        ]
    )
    path_summaries = {}
    for path in paths:
        path_selections = [event for event in selections if event.path == path]
        path_fallbacks = [event for event in fallbacks if event.path == path]
        path_summaries[path] = {
            "selected_backends": _unique_sorted(
                event.backend for event in path_selections
            ),
            "fallback_backends": _unique_sorted(
                [
                    *(event.backend for event in path_selections if event.is_fallback),
                    *(event.backend for event in path_fallbacks),
                ]
            ),
            "selection_count": len(path_selections),
            "fallback_event_count": len(path_fallbacks),
            "native_backend_selected": any(
                not event.is_fallback for event in path_selections
            ),
            "fallback_selected": bool(path_fallbacks)
            or any(event.is_fallback for event in path_selections),
        }

    moe_missing_status = "not_observed" if "moe" in required_paths else "not_requested"
    return {
        "paths": path_summaries,
        "capabilities": {
            "native_nvfp4_gemm": {
                "status": _backend_status_for_path(
                    "linear",
                    selections,
                    fallbacks,
                    missing_status="not_observed",
                ),
                "evidence_path": "linear",
            },
            "native_nvfp4_moe_non_ep": {
                "status": _backend_status_for_path(
                    "moe",
                    selections,
                    fallbacks,
                    missing_status=moe_missing_status,
                ),
                "evidence_path": "moe",
            },
            "native_nvfp4_moe_ep": {
                "status": "not_validated",
                "reason": (
                    "Expert-parallel/all2all/EPLB NVFP4 MoE is blocked until "
                    "multi-Spark contracts are validated."
                ),
            },
            "cuda_graph": {
                "status": "not_validated_by_smoke",
                "configured_cudagraph_mode": (
                    vllm_config_summary.get("compilation", {}).get("cudagraph_mode")
                    if vllm_config_summary is not None
                    else None
                ),
                "configured_cudagraph_enabled": (
                    vllm_config_summary.get("compilation", {}).get(
                        "cudagraph_enabled"
                    )
                    if vllm_config_summary is not None
                    else None
                ),
                "reason": (
                    "The GB10 smoke harness validates model startup and one "
                    "short generation only."
                ),
            },
            "model_shape": {
                "status": "smoke_passed" if status == "passed" else "smoke_failed",
                "required_paths": list(required_paths),
            },
        },
    }


def _capability_status(
    backend_summary: dict[str, Any],
    capability_name: str,
) -> str | None:
    capability = backend_summary.get("capabilities", {}).get(capability_name)
    if not isinstance(capability, dict):
        return None
    status = capability.get("status")
    return str(status) if status is not None else None


def _nested_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _build_gb10_release_summary(
    *,
    args: argparse.Namespace,
    status: str,
    required_paths: Sequence[str],
    selections: Sequence[Any],
    fallbacks: Sequence[Any],
    backend_summary: dict[str, Any],
    vllm_config_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    native_gemm_status = _capability_status(backend_summary, "native_nvfp4_gemm")
    native_moe_status = _capability_status(
        backend_summary,
        "native_nvfp4_moe_non_ep",
    )
    moe_required = "moe" in required_paths
    fallback_selection_count = sum(1 for event in selections if event.is_fallback)
    fallback_event_count = len(fallbacks)
    fallback_free = fallback_selection_count == 0 and fallback_event_count == 0
    configured_cache_dtype = _nested_get(vllm_config_summary, "cache", "cache_dtype")
    configured_cudagraph_mode = _nested_get(
        vllm_config_summary,
        "compilation",
        "cudagraph_mode",
    )
    configured_cudagraph_enabled = _nested_get(
        vllm_config_summary,
        "compilation",
        "cudagraph_enabled",
    )

    smoke_checks = {
        "model_smoke": {
            "status": "passed" if status == "passed" else "failed",
        },
        "native_nvfp4_gemm": {
            "status": native_gemm_status,
            "required": True,
        },
        "native_nvfp4_moe_non_ep": {
            "status": native_moe_status,
            "required": moe_required,
        },
        "fallback_free": {
            "status": "passed" if fallback_free else "failed",
            "fallback_selection_count": fallback_selection_count,
            "fallback_event_count": fallback_event_count,
        },
        "kv_cache_dtype": {
            "status": (
                "passed"
                if configured_cache_dtype == args.kv_cache_dtype
                else "not_observed"
            ),
            "expected": args.kv_cache_dtype,
            "configured": configured_cache_dtype,
        },
        "attention_backend": {
            "status": (
                "configured"
                if vllm_config_summary is not None
                else "not_observed_by_report"
            ),
            "requested_backend": _nested_get(
                vllm_config_summary,
                "attention",
                "requested_backend",
            ),
            "mla_prefill_backend": _nested_get(
                vllm_config_summary,
                "attention",
                "mla_prefill_backend",
            ),
        },
        "cuda_graph": {
            "status": "not_validated_by_smoke",
            "configured_mode": configured_cudagraph_mode,
            "configured_enabled": configured_cudagraph_enabled,
        },
    }

    smoke_blockers = []
    if smoke_checks["model_smoke"]["status"] != "passed":
        smoke_blockers.append("model smoke failed")
    if native_gemm_status != "observed":
        smoke_blockers.append("native NVFP4 dense GEMM was not observed")
    if moe_required and native_moe_status != "observed":
        smoke_blockers.append("required native NVFP4 non-EP MoE was not observed")
    if not fallback_free:
        smoke_blockers.append("NVFP4 fallback events or selections were observed")
    if smoke_checks["kv_cache_dtype"]["status"] != "passed":
        smoke_blockers.append("configured KV cache dtype did not match smoke request")

    return {
        "first_path_smoke_passed": not smoke_blockers,
        "smoke_blockers": smoke_blockers,
        "checks": smoke_checks,
        "release_ready": False,
        "remaining_release_evidence": [
            "final runtime image smoke with the published GB10 dependency wheels",
            "OpenAI-compatible server smoke",
            "CUDA graph capture/replay validation",
            "correctness or deterministic generation evidence for the target model",
            "prefill/decode benchmark evidence",
        ],
    }


def _build_report(
    *,
    args: argparse.Namespace,
    required_paths: Sequence[str],
    selections: Sequence[Any],
    fallbacks: Sequence[Any],
    outputs: Sequence[Any],
    status: str,
    vllm_config_summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    generated_texts = []
    for output in outputs:
        completions = getattr(output, "outputs", [])
        generated_texts.append(completions[0].text if completions else "")

    report = {
        "schema_version": 1,
        "status": status,
        "runtime": _collect_runtime_metadata(),
        "model": args.model,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "vllm_config": vllm_config_summary,
        "sampling": {
            "max_tokens": args.gb10_max_tokens,
            "temperature": args.gb10_temperature,
            "seed": args.gb10_sampling_seed,
            "skip_generate": args.gb10_skip_generate,
        },
        "checks": {
            "allow_fallback": args.gb10_allow_fallback,
            "required_paths": list(required_paths),
            "expected_backends": [
                {"path": path, "backend_substring": backend_substring}
                for path, backend_substring in args.gb10_expect_backend
            ],
        },
        "backend_selections": _events_to_dicts(selections),
        "fallback_events": _events_to_dicts(fallbacks),
        "generated_texts": generated_texts,
    }
    backend_summary = _build_backend_summary(
        status=status,
        required_paths=required_paths,
        selections=selections,
        fallbacks=fallbacks,
        vllm_config_summary=vllm_config_summary,
    )
    report["backend_summary"] = backend_summary
    report["gb10_release_summary"] = _build_gb10_release_summary(
        args=args,
        status=status,
        required_paths=required_paths,
        selections=selections,
        fallbacks=fallbacks,
        backend_summary=backend_summary,
        vllm_config_summary=vllm_config_summary,
    )
    if error is not None:
        report["error"] = error
    return report


def _write_report(report_path: str | None, report: dict[str, Any]) -> None:
    if report_path is None:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"GB10 NVFP4 smoke report written to {path}")


def _assert_backend_events(
    *,
    selections: Sequence[Any],
    fallbacks: Sequence[Any],
    required_paths: Sequence[str],
    expected_backends: Sequence[tuple[str, str]],
    allow_fallback: bool,
) -> None:
    if fallbacks and not allow_fallback:
        raise RuntimeError(
            "NVFP4 fallback events were recorded during GB10 smoke:\n"
            f"{_format_events(fallbacks)}"
        )

    fallback_selections = [event for event in selections if event.is_fallback]
    if fallback_selections and not allow_fallback:
        raise RuntimeError(
            "NVFP4 backend selections included fallback paths:\n"
            f"{_format_events(fallback_selections)}"
        )

    for path in required_paths:
        path_events = [event for event in selections if event.path == path]
        if not path_events:
            raise RuntimeError(
                f"No NVFP4 backend-selection event was recorded for {path!r}. "
                "Confirm this is an NVFP4 model path and keep "
                "VLLM_ENABLE_V1_MULTIPROCESSING=0 for this smoke test."
            )
        if not allow_fallback and all(event.is_fallback for event in path_events):
            raise RuntimeError(
                f"Only fallback NVFP4 backend selections were recorded for {path!r}:\n"
                f"{_format_events(path_events)}"
            )

    for path, backend_substring in expected_backends:
        path_events = [event for event in selections if event.path == path]
        if not any(backend_substring in event.backend for event in path_events):
            raise RuntimeError(
                f"No NVFP4 backend-selection event for {path!r} contained "
                f"{backend_substring!r}:\n{_format_events(path_events)}"
            )


def _shutdown_llm(llm: Any | None) -> None:
    if llm is None:
        return
    llm_engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(llm_engine, "shutdown", None)
    if callable(shutdown):
        shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    allow_fallback = _preparse_allow_fallback(argv)
    _configure_env(allow_fallback=allow_fallback)

    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.quantization.utils.nvfp4_fallback import (
        clear_nvfp4_backend_events,
        get_nvfp4_backend_selection_events,
        get_nvfp4_fallback_events,
    )

    parser = _build_parser(EngineArgs)
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model or GB10_NVFP4_MODEL is required")

    required_paths = tuple(args.gb10_require_path or DEFAULT_REQUIRED_PATHS)
    clear_nvfp4_backend_events()

    engine_args = EngineArgs.from_cli_args(args)
    sampling_params = SamplingParams(
        max_tokens=args.gb10_max_tokens,
        temperature=args.gb10_temperature,
        seed=args.gb10_sampling_seed,
    )

    llm = None
    outputs: Sequence[Any] = ()
    selections: Sequence[Any] = ()
    fallbacks: Sequence[Any] = ()
    vllm_config_summary: dict[str, Any] | None = None
    try:
        llm = LLM.from_engine_args(engine_args)
        vllm_config_summary = _collect_vllm_config_summary(llm)
        if not args.gb10_skip_generate:
            outputs = llm.generate(
                [args.gb10_prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            _print_generation_outputs(outputs)

        selections = get_nvfp4_backend_selection_events()
        fallbacks = get_nvfp4_fallback_events()
        _print_event_summary(selections, fallbacks)
        _assert_backend_events(
            selections=selections,
            fallbacks=fallbacks,
            required_paths=required_paths,
            expected_backends=tuple(args.gb10_expect_backend),
            allow_fallback=args.gb10_allow_fallback,
        )
        _write_report(
            args.gb10_report_json,
            _build_report(
                args=args,
                required_paths=required_paths,
                selections=selections,
                fallbacks=fallbacks,
                outputs=outputs,
                status="passed",
                vllm_config_summary=vllm_config_summary,
            ),
        )
    except Exception as exc:
        selections = get_nvfp4_backend_selection_events()
        fallbacks = get_nvfp4_fallback_events()
        if vllm_config_summary is None:
            vllm_config_summary = _collect_vllm_config_summary(llm)
        _write_report(
            args.gb10_report_json,
            _build_report(
                args=args,
                required_paths=required_paths,
                selections=selections,
                fallbacks=fallbacks,
                outputs=outputs,
                status="failed",
                vllm_config_summary=vllm_config_summary,
                error=str(exc),
            ),
        )
        raise
    finally:
        _shutdown_llm(llm)

    print("GB10 NVFP4 smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
