# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm.utils import flashinfer


@pytest.fixture(autouse=True)
def clear_flashinfer_caches():
    flashinfer.has_flashinfer.cache_clear()
    flashinfer.has_flashinfer_cubin.cache_clear()
    yield
    flashinfer.has_flashinfer.cache_clear()
    flashinfer.has_flashinfer_cubin.cache_clear()


def _patch_flashinfer_specs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    flashinfer_found: bool,
    flashinfer_cubin_found: bool,
) -> None:
    def fake_find_spec(module_name: str):
        if module_name == "flashinfer":
            return object() if flashinfer_found else None
        if module_name == "flashinfer_cubin":
            return object() if flashinfer_cubin_found else None
        return None

    monkeypatch.setattr(flashinfer.envs, "VLLM_HAS_FLASHINFER_CUBIN", False)
    monkeypatch.setattr(flashinfer.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(flashinfer.shutil, "which", lambda _name: "/usr/bin/nvcc")


def test_has_flashinfer_uses_nvcc_when_jit_is_enabled_without_cubin(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    _patch_flashinfer_specs(
        monkeypatch,
        flashinfer_found=True,
        flashinfer_cubin_found=False,
    )

    assert flashinfer.has_flashinfer() is True


def test_has_flashinfer_rejects_nvcc_fallback_when_jit_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    _patch_flashinfer_specs(
        monkeypatch,
        flashinfer_found=True,
        flashinfer_cubin_found=False,
    )

    assert flashinfer.has_flashinfer() is False


def test_has_flashinfer_accepts_cubin_when_jit_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    _patch_flashinfer_specs(
        monkeypatch,
        flashinfer_found=True,
        flashinfer_cubin_found=True,
    )

    assert flashinfer.has_flashinfer() is True


def test_has_flashinfer_warm_cache_survives_jit_env_toggle(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    _patch_flashinfer_specs(
        monkeypatch,
        flashinfer_found=True,
        flashinfer_cubin_found=False,
    )

    assert flashinfer.has_flashinfer() is True

    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    assert flashinfer.has_flashinfer() is True

    flashinfer.has_flashinfer.cache_clear()
    assert flashinfer.has_flashinfer() is False
