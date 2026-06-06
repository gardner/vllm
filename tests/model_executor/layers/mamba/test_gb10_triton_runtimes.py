from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.mamba.linear_attn import (
    MiniMaxText01LinearAttention,
)
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.platforms import current_platform


def _mock_gb10(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda family, device_id=0: family == 120,
    )


def test_gb10_mamba1_triton_runtime_rejects(monkeypatch: pytest.MonkeyPatch):
    _mock_gb10(monkeypatch)

    with pytest.raises(ValueError, match="Mamba1 triton runtime.*GB10/SM12x"):
        MambaMixer(
            hidden_size=8,
            ssm_state_size=4,
            conv_kernel_size=4,
            intermediate_size=8,
            time_step_rank=2,
            use_conv_bias=False,
            use_bias=False,
            use_rms_norm=False,
            model_config=None,
            cache_config=None,
        )


def test_gb10_mamba2_triton_runtime_rejects(monkeypatch: pytest.MonkeyPatch):
    _mock_gb10(monkeypatch)

    with pytest.raises(
        ValueError, match="Mamba2 triton SSD runtime.*GB10/SM12x"
    ):
        MambaMixer2(
            hidden_size=8,
            ssm_state_size=4,
            conv_kernel_size=4,
            intermediate_size=8,
            use_conv_bias=False,
            use_bias=False,
            n_groups=1,
            num_heads=4,
            head_dim=2,
            model_config=None,
            cache_config=None,
        )


def test_gb10_short_conv_triton_runtime_rejects(monkeypatch: pytest.MonkeyPatch):
    _mock_gb10(monkeypatch)

    short_conv_config = SimpleNamespace(conv_L_cache=4, conv_bias=False)

    with pytest.raises(ValueError, match="short_conv triton runtime.*GB10/SM12x"):
        ShortConv(
            short_conv_config,
            dim=8,
            layer_idx=0,
            model_config=None,
            cache_config=None,
        )


def test_gb10_linear_attention_triton_runtime_rejects(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_gb10(monkeypatch)

    with pytest.raises(
        ValueError, match="linear attention triton runtime.*GB10/SM12x"
    ):
        MiniMaxText01LinearAttention(
            hidden_size=8,
            hidden_inner_size=8,
            num_heads=4,
            head_dim=2,
            max_position=16,
            block_size=4,
            num_hidden_layer=1,
            model_config=None,
            cache_config=None,
            quant_config=None,
        )
