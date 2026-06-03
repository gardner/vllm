# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import vllm.envs as envs


@dataclass(frozen=True)
class NvFp4FallbackEvent:
    path: str
    backend: str
    message: str


_NVFP4_FALLBACK_EVENTS: list[NvFp4FallbackEvent] = []


def record_nvfp4_fallback(path: str, backend: str, message: str) -> None:
    """Record an NVFP4 fallback and optionally fail fast.

    The process-local event list gives smoke tests a cheap way to assert that
    model initialization did not select a non-native NVFP4 path. Release
    validation can set VLLM_FAIL_ON_NVFP4_FALLBACK=1 to make any such
    selection fatal.
    """
    event = NvFp4FallbackEvent(path=path, backend=backend, message=message)
    _NVFP4_FALLBACK_EVENTS.append(event)
    if envs.VLLM_FAIL_ON_NVFP4_FALLBACK:
        raise RuntimeError(message)


def get_nvfp4_fallback_events() -> tuple[NvFp4FallbackEvent, ...]:
    return tuple(_NVFP4_FALLBACK_EVENTS)


def clear_nvfp4_fallback_events() -> None:
    _NVFP4_FALLBACK_EVENTS.clear()
