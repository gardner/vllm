# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.mamba.ops import ssu_dispatch
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


class _FakeFlashInferSSUBackend:
    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(self, *args, **kwargs) -> None:
        raise AssertionError("backend dispatch should not run during selection tests")


@pytest.fixture(autouse=True)
def reset_mamba_ssu_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ssu_dispatch, "_mamba_ssu_backend", None)
    yield
    monkeypatch.setattr(ssu_dispatch, "_mamba_ssu_backend", None)


def _mock_sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ssu_dispatch,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda family: family == 120,
            get_device_capability=lambda: DeviceCapability(major=12, minor=1),
        ),
        raising=False,
    )


def _mamba_kv_cache_config(
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layers.0"],
                kv_cache_spec=MambaSpec(
                    block_size=1,
                    shapes=((1,), (1,)),
                    dtypes=(torch.float32, torch.float32),
                    mamba_type=mamba_type,
                ),
            )
        ],
    )


@pytest.mark.parametrize(
    "mamba_type",
    [MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2],
)
def test_sm12x_rejects_triton_mamba_ssu_backend(
    monkeypatch: pytest.MonkeyPatch,
    mamba_type: MambaAttentionBackendEnum,
):
    _mock_sm12x_platform(monkeypatch)

    with pytest.raises(ValueError, match="Triton Mamba SSU.*GB10/SM12x"):
        ssu_dispatch.initialize_mamba_ssu_backend(
            MambaConfig(backend=MambaBackendEnum.TRITON),
            _mamba_kv_cache_config(mamba_type),
        )


def test_sm12x_allows_flashinfer_mamba_ssu_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_sm12x_platform(monkeypatch)
    monkeypatch.setitem(
        ssu_dispatch._BACKEND_REGISTRY,
        MambaBackendEnum.FLASHINFER,
        _FakeFlashInferSSUBackend,
    )

    ssu_dispatch.initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.FLASHINFER),
        _mamba_kv_cache_config(),
    )

    assert ssu_dispatch.get_mamba_ssu_backend().name == "flashinfer"
