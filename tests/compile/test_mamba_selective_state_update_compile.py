# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import vllm.envs as envs
from tests.kernels.mamba.utils import selective_state_update_ref
from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.mamba.ops import mamba_ssm as mamba_ssm_mod
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip(
        reason="Mamba selective-state-update compile coverage requires SM120 "
        "(RTX Pro 6000 / DGX Spark).",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _set_compile_cache_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    triton_cache_dir = Path("/tmp/triton-cache")
    triton_cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton_cache_dir))

    torchinductor_cache_dir = Path("/tmp/torchinductor-cache")
    torchinductor_cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_cache_dir))


@support_torch_compile(
    dynamic_arg_dims={
        "state": 0,
        "x": 0,
        "dt": 0,
        "B": 0,
        "C": 0,
    }
)
class SelectiveStateUpdateCompileModule(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__()

    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            out=out,
        )
        return out


def test_selective_state_update_torch_compile_with_aot_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1")
    
    @contextmanager
    def _noop_device_index(_device_index):
        yield

    monkeypatch.setattr(torch.accelerator, "device_index", _noop_device_index)
    monkeypatch.setattr(
        mamba_ssm_mod,
        "try_get_optimal_ssm_config",
        lambda *args, **kwargs: (8, 4),
    )

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.float32),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_num_of_warmups=0,
        ),
    )

    device = current_platform.device_type
    state = torch.randn(1, 4, 8, device=device, dtype=torch.float32)
    x = torch.randn(1, 4, device=device, dtype=torch.float32)
    dt = torch.randn(1, 4, device=device, dtype=torch.float32)
    A = -0.5 * torch.rand(4, 8, device=device, dtype=torch.float32)
    B = torch.randn(1, 8, device=device, dtype=torch.float32)
    C = torch.randn(1, 8, device=device, dtype=torch.float32)
    D = torch.randn(4, device=device, dtype=torch.float32)
    dt_bias = torch.randn(4, device=device, dtype=torch.float32)

    state_ref = state.clone()
    expected = selective_state_update_ref(
        state_ref,
        x.clone(),
        dt.clone(),
        A.clone(),
        B.clone(),
        C.clone(),
        D=D.clone(),
        dt_bias=dt_bias.clone(),
    )

    with set_current_vllm_config(vllm_config):
        model = SelectiveStateUpdateCompileModule(vllm_config=vllm_config)
        state_actual = state.clone()
        actual = model(
            state_actual,
            x.clone(),
            dt.clone(),
            A.clone(),
            B.clone(),
            C.clone(),
            D.clone(),
            dt_bias.clone(),
        )

    assert model._compiled_bytecode is not None
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state_actual, state_ref, atol=1e-5, rtol=1e-5)
