#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Durable GSM8K accuracy harness for GB10 NVFP4 release evidence.

This replaces the ad-hoc GSM8K evaluator that produced the early
``docs/gb10-qwen36-native-evidence/gsm8k_*.json`` numbers but was never
committed. It is intentionally self-contained (no ``lm_eval`` dependency) so it
runs reproducibly inside the published GB10 runtime image, and it is
model-agnostic so the same harness measures an NVFP4 model and its BF16/FP8
reference to produce an accuracy *delta*.

It runs an offline vLLM engine, evaluates GSM8K with N-shot prompting and greedy
decoding, extracts the final numeric answer, and writes a JSON report whose
schema matches the existing evidence files:

    accuracy, invalid_rate, latency, questions_per_second, total_output_tokens,
    tokens_per_second, num_questions, num_shots, max_tokens, timestamp

Pair two runs with ``--gb10-reference-json`` to emit an accuracy delta block.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

GSM8K_BASE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data"
)
# Number token: optional sign, digits with thousands separators, optional
# decimal. Used both for the gold answer and to scrape the model completion.
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_ANSWER_IS_RE = re.compile(r"answer is[^\d\-]*(-?\$?\d[\d,]*(?:\.\d+)?)", re.I)
# GSM8K gold answers end with "#### <final answer>".
_GOLD_RE = re.compile(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)")


def _normalize_number(raw: str) -> str | None:
    if raw is None:
        return None
    cleaned = raw.replace(",", "").replace("$", "").strip().rstrip(".")
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    # Canonicalize so "42", "42.0" and "42.00" compare equal.
    if value.is_integer():
        return str(int(value))
    return repr(value)


def _extract_gold(answer_field: str) -> str | None:
    match = _GOLD_RE.search(answer_field)
    if match:
        return _normalize_number(match.group(1))
    numbers = _NUMBER_RE.findall(answer_field)
    return _normalize_number(numbers[-1]) if numbers else None


def _extract_prediction(completion: str) -> str | None:
    # Prefer the demonstrated "#### N" form, then an explicit "answer is N",
    # then fall back to the last number in the completion.
    gold_form = _GOLD_RE.search(completion)
    if gold_form:
        return _normalize_number(gold_form.group(1))
    answer_is = _ANSWER_IS_RE.search(completion)
    if answer_is:
        return _normalize_number(answer_is.group(1))
    numbers = _NUMBER_RE.findall(completion)
    return _normalize_number(numbers[-1]) if numbers else None


def _load_jsonl(url: str, cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {cache_path}")
        urllib.request.urlretrieve(url, cache_path)  # noqa: S310 (trusted URL)
    records = []
    with cache_path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_fewshot_preamble(train: Sequence[dict[str, Any]], num_shots: int) -> str:
    blocks = []
    for example in train[:num_shots]:
        blocks.append(
            f"Question: {example['question'].strip()}\n"
            f"Answer: {example['answer'].strip()}"
        )
    return "\n\n".join(blocks)


def _build_parser(engine_args_cls: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GB10 GSM8K accuracy harness for NVFP4 release evidence."
    )
    parser.add_argument(
        "--gb10-num-questions",
        type=int,
        default=200,
        help="Number of GSM8K test questions to evaluate (default: 200).",
    )
    parser.add_argument(
        "--gb10-num-shots",
        type=int,
        default=5,
        help="Number of few-shot demonstrations (default: 5).",
    )
    parser.add_argument(
        "--gb10-max-tokens",
        type=int,
        default=512,
        help="Max generated tokens per question (default: 512).",
    )
    parser.add_argument(
        "--gb10-data-dir",
        default=os.environ.get("GB10_GSM8K_DATA_DIR", "/tmp/gb10-gsm8k"),
        help="Directory to cache the GSM8K train/test jsonl files.",
    )
    parser.add_argument(
        "--gb10-output-json",
        help="Path to write the GSM8K result JSON (also printed to stdout).",
    )
    parser.add_argument(
        "--gb10-reference-json",
        help=(
            "Optional path to a previous result JSON (e.g. the BF16 reference); "
            "when set, an accuracy delta block is added to the output."
        ),
    )
    parser.add_argument(
        "--gb10-label",
        default=os.environ.get("GB10_GSM8K_LABEL"),
        help="Optional human label recorded in the report (e.g. qwen36_nvfp4).",
    )
    parser = engine_args_cls.add_cli_args(parser)
    parser.set_defaults(
        model=os.environ.get("GB10_NVFP4_MODEL"),
        gpu_memory_utilization=float(
            os.environ.get("GB10_GPU_MEMORY_UTILIZATION", "0.88")
        ),
    )
    return parser


def _evaluate(
    llm: Any,
    sampling_params: Any,
    preamble: str,
    test: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    prompts = [
        f"{preamble}\n\nQuestion: {item['question'].strip()}\nAnswer:"
        for item in test
    ]
    golds = [_extract_gold(item["answer"]) for item in test]

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    latency = time.perf_counter() - start

    correct = 0
    invalid = 0
    total_output_tokens = 0
    for output, gold in zip(outputs, golds):
        completion = output.outputs[0]
        total_output_tokens += len(completion.token_ids)
        pred = _extract_prediction(completion.text)
        if pred is None:
            invalid += 1
            continue
        if gold is not None and pred == gold:
            correct += 1

    num = len(test)
    return {
        "accuracy": correct / num if num else 0.0,
        "invalid_rate": invalid / num if num else 0.0,
        "latency": latency,
        "questions_per_second": num / latency if latency else 0.0,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": total_output_tokens / latency if latency else 0.0,
        "num_questions": num,
        "num_shots": None,  # filled by caller
        "max_tokens": sampling_params.max_tokens,
        "timestamp": time.time(),
    }


def _attach_delta(result: dict[str, Any], reference_path: str) -> None:
    with open(reference_path, encoding="utf-8") as stream:
        reference = json.load(stream)
    ref_acc = reference.get("accuracy")
    if ref_acc is None:
        return
    delta = result["accuracy"] - ref_acc
    result["reference_delta"] = {
        "reference_json": reference_path,
        "reference_accuracy": ref_acc,
        "accuracy_delta": delta,
        "accuracy_delta_pp": delta * 100.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    # NOTE: do NOT force FLASHINFER_DISABLE_JIT here. This is a correctness
    # harness, not the release smoke; some native GB10 kernels (e.g. the
    # FlashInfer Mamba SSU used by Nemotron-3-Nano) are not in the prebuilt
    # jit-cache yet and must be allowed to JIT-compile (nvcc is present in the
    # runtime image). Leaving it unset keeps FlashInfer's default (JIT on).

    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    parser = _build_parser(EngineArgs)
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model or GB10_NVFP4_MODEL is required")

    data_dir = Path(args.gb10_data_dir)
    train = _load_jsonl(f"{GSM8K_BASE_URL}/train.jsonl", data_dir / "train.jsonl")
    test = _load_jsonl(f"{GSM8K_BASE_URL}/test.jsonl", data_dir / "test.jsonl")
    test = test[: args.gb10_num_questions]
    preamble = _build_fewshot_preamble(train, args.gb10_num_shots)
    print(
        f"Running GSM8K evaluation: {len(test)} questions, "
        f"{args.gb10_num_shots}-shot, model={args.model}"
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.gb10_max_tokens,
        stop=["\n\nQuestion:", "\nQuestion:"],
    )
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM.from_engine_args(engine_args)

    result = _evaluate(llm, sampling_params, preamble, test)
    result["num_shots"] = args.gb10_num_shots
    result["model"] = args.model
    if args.gb10_label:
        result["label"] = args.gb10_label
    if args.gb10_reference_json:
        _attach_delta(result, args.gb10_reference_json)

    print(
        "\nResults:\n"
        f"Accuracy: {result['accuracy']:.3f}\n"
        f"Invalid responses: {result['invalid_rate']:.3f}\n"
        f"Total latency: {result['latency']:.3f} s\n"
        f"Output tokens per second: {result['tokens_per_second']:.3f}"
    )
    if "reference_delta" in result:
        print(
            f"Accuracy delta vs reference: "
            f"{result['reference_delta']['accuracy_delta_pp']:+.1f} pp"
        )

    if args.gb10_output_json:
        out_path = Path(args.gb10_output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
