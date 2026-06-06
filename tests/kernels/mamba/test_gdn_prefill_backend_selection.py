# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch")

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

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


def test_fi_chunk_gated_delta_rule_forwards_raw_q_and_k(monkeypatch):
    import flashinfer.gdn_prefill as flashinfer_gdn_prefill

    raw_q = torch.randn(1, 3, 2, 4, dtype=torch.float32)
    raw_k = torch.randn_like(raw_q)
    raw_v = torch.randn_like(raw_q)
    raw_g = torch.randn(1, 3, 2, dtype=torch.float32)
    raw_beta = torch.randn_like(raw_g)
    initial_state = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)

    seen = {}

    def _forbidden_l2norm(_tensor):
        raise AssertionError("wrapper must not normalize FlashInfer inputs")

    def fake_chunk_gated_delta_rule(
        *,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens,
        **kwargs,
    ):
        seen["q"] = q
        seen["k"] = k
        seen["g"] = g
        seen["beta"] = beta
        seen["initial_state"] = initial_state
        seen["output_final_state"] = output_final_state
        seen["cu_seqlens"] = cu_seqlens
        return (
            torch.zeros_like(q),
            torch.zeros(1, q.shape[1], q.shape[2], q.shape[2], dtype=q.dtype),
        )

    monkeypatch.setattr(
        flashinfer_gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    monkeypatch.setattr(gdn_mod, "l2norm_fwd", _forbidden_l2norm)

    output, final_state = gdn_mod.fi_chunk_gated_delta_rule(
        q=raw_q,
        k=raw_k,
        v=raw_v,
        g=raw_g,
        beta=raw_beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(seen["q"], raw_q.squeeze(0))
    torch.testing.assert_close(seen["k"], raw_k.squeeze(0))
    torch.testing.assert_close(seen["g"], torch.exp(raw_g.squeeze(0)))
    torch.testing.assert_close(seen["beta"], raw_beta.squeeze(0))
    torch.testing.assert_close(seen["initial_state"], initial_state)
    assert seen["initial_state"].dtype == torch.float32
    assert seen["output_final_state"] is True
    torch.testing.assert_close(seen["cu_seqlens"], cu_seqlens)
    assert output.shape == (1, 3, 2, 4)
    assert final_state.shape == (1, 2, 4, 4)
