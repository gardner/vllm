#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke an already running GB10 OpenAI-compatible vLLM server.

This script intentionally does not manage the server lifecycle. It probes the
OpenAI-compatible HTTP API exposed by a running vLLM process, sends one short
generation request, and writes a durable JSON report that can be archived with
the offline NVFP4 backend-selection smoke report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PROMPT = "NVIDIA DGX Spark native NVFP4 support means"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a GB10 OpenAI-compatible vLLM server smoke test against an "
            "already running server. The script never starts or stops vLLM."
        )
    )
    parser.add_argument(
        "--gb10-base-url",
        default=os.environ.get("GB10_OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help=(
            "OpenAI-compatible server base URL. Defaults to "
            "GB10_OPENAI_BASE_URL or http://127.0.0.1:8000."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GB10_OPENAI_MODEL"),
        help=(
            "Model id to request. Defaults to GB10_OPENAI_MODEL, then the "
            "first id returned by /v1/models."
        ),
    )
    parser.add_argument(
        "--gb10-endpoint",
        choices=("completions", "chat"),
        default="completions",
        help="Generation endpoint to smoke: /v1/completions or /v1/chat/completions.",
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
        "--gb10-seed",
        type=int,
        default=0,
        help="Sampling seed sent to vLLM for deterministic smoke requests.",
    )
    parser.add_argument(
        "--gb10-timeout",
        type=float,
        default=30.0,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--gb10-retries",
        type=int,
        default=1,
        help="Number of retries after the first request attempt.",
    )
    parser.add_argument(
        "--gb10-retry-delay",
        type=float,
        default=2.0,
        help="Delay in seconds between failed request attempts.",
    )
    parser.add_argument(
        "--gb10-repeat-count",
        type=int,
        default=1,
        help=(
            "Number of identical generation requests to send. Values greater "
            "than one add deterministic-generation evidence to the report."
        ),
    )
    parser.add_argument(
        "--gb10-require-deterministic",
        action="store_true",
        help=(
            "Fail if repeated generation requests do not return identical "
            "generated text."
        ),
    )
    parser.add_argument(
        "--gb10-allow-empty",
        action="store_true",
        help="Allow an empty generated text field in the response.",
    )
    parser.add_argument(
        "--gb10-report-json",
        help="Optional path for a JSON report containing server smoke evidence.",
    )
    return parser


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _json_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    api_key = os.environ.get("GB10_OPENAI_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _runtime_env() -> dict[str, str | None]:
    return {
        "FLASHINFER_DISABLE_JIT": os.environ.get("FLASHINFER_DISABLE_JIT"),
        "GB10_GPU_MEMORY_UTILIZATION": os.environ.get(
            "GB10_GPU_MEMORY_UTILIZATION"
        ),
    }


def _http_json(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=_json_headers(),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            return response.status, json.loads(response_body) if response_body else {}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        error_body = error_body[:2000]
        raise RuntimeError(
            f"{method} {url} failed with HTTP {exc.code}: {error_body}"
        ) from exc


def _request_with_retries(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> tuple[int, dict[str, Any]]:
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return _http_json(
                method=method,
                url=url,
                payload=payload,
                timeout=timeout,
            )
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(retry_delay)

    assert last_error is not None
    raise last_error


def _extract_model_ids(models_body: dict[str, Any] | None) -> list[str]:
    if not isinstance(models_body, dict):
        return []
    data = models_body.get("data", [])
    if not isinstance(data, list):
        return []
    return [
        model["id"]
        for model in data
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    ]


def _summarize_model_entry(model_entry: Any) -> dict[str, Any] | None:
    if not isinstance(model_entry, dict):
        return None
    model_id = model_entry.get("id")
    if not isinstance(model_id, str):
        return None
    return {
        "id": model_id,
        "object": model_entry.get("object"),
        "created": model_entry.get("created"),
        "owned_by": model_entry.get("owned_by"),
        "root": model_entry.get("root"),
        "parent": model_entry.get("parent"),
        "max_model_len": model_entry.get("max_model_len"),
    }


def _summarize_models(
    models_body: dict[str, Any] | None,
    requested_model: str | None,
) -> dict[str, Any]:
    if not isinstance(models_body, dict):
        return {
            "status": None,
            "ids": [],
            "raw_count": None,
            "served_models": [],
            "selected": None,
        }
    data = models_body.get("data", [])
    if not isinstance(data, list):
        data = []

    served_models = [
        summary
        for summary in (_summarize_model_entry(model_entry) for model_entry in data)
        if summary is not None
    ]
    selected = None
    if requested_model is not None:
        selected = next(
            (
                served_model
                for served_model in served_models
                if served_model["id"] == requested_model
            ),
            None,
        )

    return {
        "status": None,
        "ids": [served_model["id"] for served_model in served_models],
        "raw_count": len(data),
        "served_models": served_models,
        "selected": selected,
    }


def _completion_path(endpoint: str) -> str:
    if endpoint == "chat":
        return "/v1/chat/completions"
    return "/v1/completions"


def _build_generation_payload(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed

    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt

    return payload


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        text = part.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_generated_text(endpoint: str, response: dict[str, Any]) -> str:
    generated_text, _ = _extract_generated_text_with_source(endpoint, response)
    return generated_text


def _extract_generated_text_with_source(
    endpoint: str,
    response: dict[str, Any],
) -> tuple[str, str | None]:
    first_choice = _first_choice(response)
    if first_choice is None:
        return "", None

    if endpoint == "chat":
        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            return "", None
        content = message.get("content", "")
        if isinstance(content, list):
            generated_text = "".join(_content_part_text(part) for part in content)
            if generated_text:
                return generated_text, "message.content"
        elif isinstance(content, str) and content:
            return content, "message.content"

        for field_name in ("reasoning_content", "reasoning"):
            reasoning = message.get(field_name)
            if isinstance(reasoning, str) and reasoning:
                return reasoning, f"message.{field_name}"
        return "", None

    text = first_choice.get("text", "")
    return (text, "text") if isinstance(text, str) and text else ("", None)


def _first_choice(response: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    return first_choice if isinstance(first_choice, dict) else None


def _extract_choice_metadata(response: dict[str, Any] | None) -> dict[str, Any]:
    first_choice = _first_choice(response)
    if first_choice is None:
        return {
            "index": None,
            "finish_reason": None,
            "stop_reason": None,
        }
    return {
        "index": first_choice.get("index"),
        "finish_reason": first_choice.get("finish_reason"),
        "stop_reason": first_choice.get("stop_reason"),
    }


def _build_response_summary(
    *,
    completion_status: int | None,
    completion_body: dict[str, Any] | None,
    generated_text: str,
    generated_text_source: str | None,
) -> dict[str, Any]:
    completion_body = completion_body or {}
    return {
        "status": completion_status,
        "id": completion_body.get("id"),
        "object": completion_body.get("object"),
        "created": completion_body.get("created"),
        "model": completion_body.get("model"),
        "system_fingerprint": completion_body.get("system_fingerprint"),
        "choice": _extract_choice_metadata(completion_body),
        "generated_text": generated_text,
        "generated_text_source": generated_text_source,
        "usage": completion_body.get("usage"),
    }


def _build_deterministic_summary(
    response_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_texts = [
        str(response_summary.get("generated_text", ""))
        for response_summary in response_summaries
    ]
    unique_generated_texts = sorted(set(generated_texts))
    status = "not_requested"
    if len(response_summaries) > 1:
        status = "passed" if len(unique_generated_texts) == 1 else "failed"

    return {
        "status": status,
        "repeat_count": len(response_summaries),
        "unique_generated_text_count": len(unique_generated_texts),
        "generated_texts_match": len(unique_generated_texts) <= 1,
    }


def _build_report(
    *,
    args: argparse.Namespace,
    base_url: str,
    model: str | None,
    models_status: int | None,
    models_body: dict[str, Any] | None,
    completion_status: int | None,
    completion_body: dict[str, Any] | None,
    generated_text: str,
    generated_text_source: str | None,
    response_summaries: list[dict[str, Any]] | None = None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    models_summary = _summarize_models(models_body, model)
    models_summary["status"] = models_status
    endpoint_path = _completion_path(args.gb10_endpoint)
    fallback_response_summary = _build_response_summary(
        completion_status=completion_status,
        completion_body=completion_body,
        generated_text=generated_text,
        generated_text_source=generated_text_source,
    )
    response_summaries = response_summaries or [fallback_response_summary]
    first_response_summary = response_summaries[0]
    deterministic_summary = _build_deterministic_summary(response_summaries)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "base_url": base_url,
        "endpoint": {
            "name": args.gb10_endpoint,
            "path": endpoint_path,
        },
        "model": model,
        "models": models_summary,
        "request": {
            "prompt": args.gb10_prompt,
            "max_tokens": args.gb10_max_tokens,
            "temperature": args.gb10_temperature,
            "seed": args.gb10_seed,
            "timeout": args.gb10_timeout,
            "retries": args.gb10_retries,
            "repeat_count": args.gb10_repeat_count,
            "require_deterministic": args.gb10_require_deterministic,
        },
        "runtime": {
            "env": _runtime_env(),
        },
        "response": first_response_summary,
        "responses": response_summaries,
        "deterministic_generation": deterministic_summary,
        "gb10_release_evidence": {
            "openai_compatible_server_smoke": {
                "status": "passed" if status == "passed" else "failed",
                "endpoint": endpoint_path,
                "generated_text_observed": bool(generated_text),
            },
            "deterministic_generation": deterministic_summary,
            "release_ready": False,
            "remaining_release_evidence": [
                "final runtime image smoke with the published GB10 dependency wheels",
                "offline NVFP4 backend-selection smoke report",
                "CUDA graph capture/replay validation",
                "correctness or deterministic generation evidence for the target model",
                "prefill/decode benchmark evidence",
            ],
            "note": (
                "This OpenAI-compatible server smoke proves API reachability "
                "and generation only; native backend evidence comes from the "
                "offline GB10 NVFP4 smoke report."
            ),
        },
    }
    if error is not None:
        report["error"] = error
    return report


def _write_report(report_path: str | None, report: dict[str, Any]) -> None:
    if report_path is None:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"GB10 OpenAI-compatible server smoke report written to {path}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base_url = args.gb10_base_url
    model = args.model
    models_status: int | None = None
    models_body: dict[str, Any] | None = None
    completion_status: int | None = None
    completion_body: dict[str, Any] | None = None
    generated_text = ""
    generated_text_source: str | None = None
    response_summaries: list[dict[str, Any]] = []

    try:
        if args.gb10_repeat_count < 1:
            raise RuntimeError("--gb10-repeat-count must be at least 1")

        models_status, models_body = _request_with_retries(
            method="GET",
            url=_join_url(base_url, "/v1/models"),
            timeout=args.gb10_timeout,
            retries=args.gb10_retries,
            retry_delay=args.gb10_retry_delay,
        )
        model_ids = _extract_model_ids(models_body)
        if model is None:
            if not model_ids:
                raise RuntimeError(
                    "No model id was provided and /v1/models returned no ids"
                )
            model = model_ids[0]

        payload = _build_generation_payload(
            endpoint=args.gb10_endpoint,
            model=model,
            prompt=args.gb10_prompt,
            max_tokens=args.gb10_max_tokens,
            temperature=args.gb10_temperature,
            seed=args.gb10_seed,
        )
        for _ in range(args.gb10_repeat_count):
            completion_status, completion_body = _request_with_retries(
                method="POST",
                url=_join_url(base_url, _completion_path(args.gb10_endpoint)),
                payload=payload,
                timeout=args.gb10_timeout,
                retries=args.gb10_retries,
                retry_delay=args.gb10_retry_delay,
            )
            generated_text, generated_text_source = (
                _extract_generated_text_with_source(
                    args.gb10_endpoint,
                    completion_body,
                )
            )
            if not generated_text and not args.gb10_allow_empty:
                raise RuntimeError(
                    "Generation response did not contain non-empty generated text"
                )
            response_summaries.append(
                _build_response_summary(
                    completion_status=completion_status,
                    completion_body=completion_body,
                    generated_text=generated_text,
                    generated_text_source=generated_text_source,
                )
            )

        deterministic_summary = _build_deterministic_summary(response_summaries)
        if (
            args.gb10_require_deterministic
            and deterministic_summary["status"] != "passed"
        ):
            raise RuntimeError("Repeated generation responses were not deterministic")

        report = _build_report(
            args=args,
            base_url=base_url,
            model=model,
            models_status=models_status,
            models_body=models_body,
            completion_status=completion_status,
            completion_body=completion_body,
            generated_text=generated_text,
            generated_text_source=generated_text_source,
            response_summaries=response_summaries,
            status="passed",
        )
        _write_report(args.gb10_report_json, report)
        print(f"GB10 OpenAI-compatible server smoke output: {generated_text!r}")
        return 0
    except Exception as exc:
        report = _build_report(
            args=args,
            base_url=base_url,
            model=model,
            models_status=models_status,
            models_body=models_body,
            completion_status=completion_status,
            completion_body=completion_body,
            generated_text=generated_text,
            generated_text_source=generated_text_source,
            response_summaries=response_summaries,
            status="failed",
            error=str(exc),
        )
        _write_report(args.gb10_report_json, report)
        print(
            f"GB10 OpenAI-compatible server smoke failed: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
