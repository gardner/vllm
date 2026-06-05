# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod


def _make_config(
    *,
    requested_backend: str = "auto",
    head_k_dim: int = 128,
):
    return SimpleNamespace(
        additional_config={"gdn_prefill_backend": requested_backend},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(linear_key_head_dim=head_k_dim)
        ),
    )


def _mock_cuda_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    family: int,
    runtime_major: int = 13,
) -> None:
    monkeypatch.setattr(gdn_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        gdn_mod.current_platform,
        "is_device_capability",
        lambda capability: capability == 90 and family == 90,
    )
    monkeypatch.setattr(
        gdn_mod.current_platform,
        "is_device_capability_family",
        lambda capability: capability == family,
    )
    monkeypatch.setattr(
        gdn_mod.current_platform,
        "get_cuda_runtime_major",
        lambda: runtime_major,
    )


def test_gdn_prefill_backend_sm12x_auto_rejects_missing_flashinfer_kernel(
    monkeypatch,
):
    _mock_cuda_platform(monkeypatch, family=120)
    monkeypatch.setattr(gdn_mod, "_has_flashinfer_sm12x_gdn_prefill", lambda: False)

    with pytest.raises(ValueError, match="GDN prefill.*GB10/SM12x"):
        gdn_mod._resolve_gdn_prefill_backend(_make_config())


def test_gdn_prefill_backend_sm12x_auto_uses_flashinfer_when_kernel_exists(
    monkeypatch,
):
    _mock_cuda_platform(monkeypatch, family=120)
    monkeypatch.setattr(gdn_mod, "_has_flashinfer_sm12x_gdn_prefill", lambda: True)

    requested_backend, active_backend = gdn_mod._resolve_gdn_prefill_backend(
        _make_config()
    )

    assert requested_backend == "auto"
    assert active_backend == "flashinfer"


@pytest.mark.parametrize("requested_backend", ["triton", "cutedsl"])
def test_gdn_prefill_backend_sm12x_rejects_non_flashinfer_backends(
    monkeypatch,
    requested_backend,
):
    _mock_cuda_platform(monkeypatch, family=120)
    monkeypatch.setattr(gdn_mod, "_has_flashinfer_sm12x_gdn_prefill", lambda: True)

    with pytest.raises(ValueError, match="GDN prefill.*GB10/SM12x"):
        gdn_mod._resolve_gdn_prefill_backend(
            _make_config(requested_backend=requested_backend)
        )


def test_gdn_prefill_backend_sm12x_rejects_non_qwen_head_dim(monkeypatch):
    _mock_cuda_platform(monkeypatch, family=120)
    monkeypatch.setattr(gdn_mod, "_has_flashinfer_sm12x_gdn_prefill", lambda: True)

    with pytest.raises(ValueError, match="head_k_dim=64"):
        gdn_mod._resolve_gdn_prefill_backend(_make_config(head_k_dim=64))
