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
import os
import sys
from collections.abc import Iterable, Sequence
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
    try:
        llm = LLM.from_engine_args(engine_args)
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
    finally:
        _shutdown_llm(llm)

    print("GB10 NVFP4 smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
