# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import vllm.envs as envs


@dataclass(frozen=True)
class NvFp4FallbackEvent:
    path: str
    backend: str
    message: str


@dataclass(frozen=True)
class NvFp4BackendSelectionEvent:
    path: str
    backend: str
    is_fallback: bool


_NVFP4_BACKEND_SELECTION_EVENTS: list[NvFp4BackendSelectionEvent] = []
_NVFP4_FALLBACK_EVENTS: list[NvFp4FallbackEvent] = []


def record_nvfp4_backend_selection(
    path: str,
    backend: str,
    *,
    is_fallback: bool,
) -> None:
    _NVFP4_BACKEND_SELECTION_EVENTS.append(
        NvFp4BackendSelectionEvent(
            path=path,
            backend=backend,
            is_fallback=is_fallback,
        )
    )


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


def get_nvfp4_backend_selection_events() -> tuple[NvFp4BackendSelectionEvent, ...]:
    return tuple(_NVFP4_BACKEND_SELECTION_EVENTS)


def clear_nvfp4_fallback_events() -> None:
    _NVFP4_FALLBACK_EVENTS.clear()


def clear_nvfp4_backend_selection_events() -> None:
    _NVFP4_BACKEND_SELECTION_EVENTS.clear()


def clear_nvfp4_backend_events() -> None:
    clear_nvfp4_backend_selection_events()
    clear_nvfp4_fallback_events()
