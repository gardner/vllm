# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.model_executor.layers.fused_moe.oracle.fp8 as fp8_oracle
import vllm.model_executor.layers.fused_moe.oracle.int_wna16 as int_wna16_oracle
import vllm.model_executor.layers.fused_moe.oracle.mxfp8 as mxfp8_oracle
import vllm.model_executor.layers.fused_moe.oracle.unquantized as unquantized_oracle
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    select_mxfp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kInt4Static32,
)


class _SupportedExperts:
    @staticmethod
    def is_supported_config(*_args, **_kwargs):
        return True, None


class _UnsupportedExperts:
    @staticmethod
    def is_supported_config(*_args, **_kwargs):
        return False, "mock unsupported"


class _Sm12xPlatform:
    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def is_rocm() -> bool:
        return False

    @staticmethod
    def is_xpu() -> bool:
        return False

    @staticmethod
    def is_cpu() -> bool:
        return False

    @staticmethod
    def is_tpu() -> bool:
        return False

    @staticmethod
    def is_out_of_tree() -> bool:
        return False

    @staticmethod
    def is_device_capability_family(capability: int) -> bool:
        return capability == 120

    @staticmethod
    def is_device_capability(capability: int) -> bool:
        return capability == 121


def _mock_sm12x_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _Sm12xPlatform()
    monkeypatch.setattr(int_wna16_oracle, "current_platform", platform)
    monkeypatch.setattr(unquantized_oracle, "current_platform", platform)
    monkeypatch.setattr(fp8_oracle, "current_platform", platform)
    monkeypatch.setattr(mxfp8_oracle, "current_platform", platform, raising=False)


def _mock_unquantized_backend_support(
    monkeypatch: pytest.MonkeyPatch,
    supported_backends: set[UnquantizedMoeBackend],
) -> dict[UnquantizedMoeBackend, type]:
    def _experts_base(backend: UnquantizedMoeBackend) -> type:
        if backend in supported_backends:
            return _SupportedExperts
        return _UnsupportedExperts

    kernel_by_backend = {
        backend: type(
            f"{backend.name}Experts",
            (_experts_base(backend),),
            {},
        )
        for backend in UnquantizedMoeBackend
        if backend
        not in {
            UnquantizedMoeBackend.CPU,
            UnquantizedMoeBackend.TPU,
            UnquantizedMoeBackend.OOT,
        }
    }

    monkeypatch.setattr(
        unquantized_oracle,
        "backend_to_kernel_cls",
        lambda backend: kernel_by_backend[backend],
    )
    return kernel_by_backend


def _mock_fp8_backend_support(
    monkeypatch: pytest.MonkeyPatch,
    supported_backends: set[Fp8MoeBackend],
) -> dict[Fp8MoeBackend, type]:
    def _experts_base(backend: Fp8MoeBackend) -> type:
        if backend in supported_backends:
            return _SupportedExperts
        return _UnsupportedExperts

    kernel_by_backend = {
        backend: type(
            f"{backend.name}Experts",
            (_experts_base(backend),),
            {},
        )
        for backend in Fp8MoeBackend
        if backend != Fp8MoeBackend.NONE
    }

    def _backend_to_kernel_cls(backend: Fp8MoeBackend) -> list[type]:
        return [kernel_by_backend[backend]]

    monkeypatch.setattr(fp8_oracle, "backend_to_kernel_cls", _backend_to_kernel_cls)
    monkeypatch.setattr(mxfp8_oracle, "backend_to_kernel_cls", _backend_to_kernel_cls)
    return kernel_by_backend


def _mock_wna16_backend_support(
    monkeypatch: pytest.MonkeyPatch,
    supported_backends: set[WNA16MoEBackend],
) -> dict[WNA16MoEBackend, type]:
    def _experts_base(backend: WNA16MoEBackend) -> type:
        if backend in supported_backends:
            return _SupportedExperts
        return _UnsupportedExperts

    kernel_by_backend = {
        backend: type(
            f"{backend.name}Experts",
            (_experts_base(backend),),
            {},
        )
        for backend in WNA16MoEBackend
    }

    monkeypatch.setattr(
        int_wna16_oracle,
        "backend_to_kernel_cls",
        lambda backend: [kernel_by_backend[backend]],
    )
    return kernel_by_backend


def test_gb10_explicit_unquantized_trtllm_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_unquantized_backend_support(
        monkeypatch,
        {UnquantizedMoeBackend.FLASHINFER_TRTLLM},
    )
    config = make_dummy_moe_config()
    config.moe_backend = "flashinfer_trtllm"

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        select_unquantized_moe_backend(config)


def test_gb10_env_explicit_unquantized_trtllm_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    monkeypatch.setattr(
        unquantized_oracle,
        "get_flashinfer_moe_backend",
        lambda: FlashinferMoeBackend.TENSORRT_LLM,
    )
    _mock_unquantized_backend_support(
        monkeypatch,
        {UnquantizedMoeBackend.FLASHINFER_TRTLLM},
    )

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        select_unquantized_moe_backend(make_dummy_moe_config())


def test_gb10_auto_unquantized_moe_skips_trtllm_for_cutlass(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    kernel_by_backend = _mock_unquantized_backend_support(
        monkeypatch,
        {
            UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            UnquantizedMoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, experts_cls = select_unquantized_moe_backend(make_dummy_moe_config())

    assert backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS
    assert experts_cls is kernel_by_backend[UnquantizedMoeBackend.FLASHINFER_CUTLASS]


def test_gb10_explicit_fp8_trtllm_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_fp8_backend_support(monkeypatch, {Fp8MoeBackend.FLASHINFER_TRTLLM})
    config = make_dummy_moe_config()
    config.moe_backend = "flashinfer_trtllm"

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        select_fp8_moe_backend(
            config,
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )


def test_gb10_env_explicit_fp8_trtllm_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    monkeypatch.setattr(
        fp8_oracle,
        "get_flashinfer_moe_backend",
        lambda: FlashinferMoeBackend.TENSORRT_LLM,
    )
    _mock_fp8_backend_support(monkeypatch, {Fp8MoeBackend.FLASHINFER_TRTLLM})

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        select_fp8_moe_backend(
            make_dummy_moe_config(),
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )


def test_gb10_auto_fp8_moe_skips_trtllm_for_cutlass(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    kernel_by_backend = _mock_fp8_backend_support(
        monkeypatch,
        {
            Fp8MoeBackend.FLASHINFER_TRTLLM,
            Fp8MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, experts_cls = select_fp8_moe_backend(
        make_dummy_moe_config(),
        weight_key=kFp8Static128BlockSym,
        activation_key=kFp8Dynamic128Sym,
    )

    assert backend == Fp8MoeBackend.FLASHINFER_CUTLASS
    assert experts_cls is kernel_by_backend[Fp8MoeBackend.FLASHINFER_CUTLASS]


def test_gb10_explicit_fp8_marlin_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_fp8_backend_support(monkeypatch, {Fp8MoeBackend.MARLIN})
    config = make_dummy_moe_config()
    config.moe_backend = "marlin"

    with pytest.raises(ValueError, match="FP8 MoE fallback.*GB10/SM12x"):
        select_fp8_moe_backend(
            config,
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )


def test_gb10_auto_fp8_cpu_moe_fallback_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_fp8_backend_support(monkeypatch, {Fp8MoeBackend.CPU})

    with pytest.raises(NotImplementedError, match="FP8 MoE fallback.*not supported"):
        select_fp8_moe_backend(
            make_dummy_moe_config(),
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )


def test_gb10_env_explicit_fp8_marlin_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    monkeypatch.setenv("VLLM_TEST_FORCE_FP8_MARLIN", "1")
    _mock_fp8_backend_support(monkeypatch, {Fp8MoeBackend.MARLIN})

    with pytest.raises(ValueError, match="FP8 MoE fallback.*GB10/SM12x"):
        select_fp8_moe_backend(
            make_dummy_moe_config(),
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )


def test_gb10_auto_fp8_moe_reports_trtllm_and_marlin_rejection(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_fp8_backend_support(
        monkeypatch,
        {
            Fp8MoeBackend.FLASHINFER_TRTLLM,
            Fp8MoeBackend.MARLIN,
        },
    )

    with pytest.raises(NotImplementedError, match="FP8 MoE fallback.*not supported"):
        select_fp8_moe_backend(
            make_dummy_moe_config(),
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )


def test_gb10_explicit_mxfp8_trtllm_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_fp8_backend_support(monkeypatch, {Fp8MoeBackend.FLASHINFER_TRTLLM})
    config = make_dummy_moe_config()
    config.moe_backend = "flashinfer_trtllm"

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*GB10/SM12x"):
        select_mxfp8_moe_backend(config)


def test_gb10_explicit_mxfp8_marlin_moe_rejected(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_fp8_backend_support(monkeypatch, {Fp8MoeBackend.MARLIN})
    config = make_dummy_moe_config()
    config.moe_backend = "marlin"

    with pytest.raises(ValueError, match="MXFP8 MoE fallback.*GB10/SM12x"):
        select_mxfp8_moe_backend(config)


def test_gb10_auto_mxfp8_moe_reports_trtllm_rejection(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_fp8_backend_support(monkeypatch, {Fp8MoeBackend.FLASHINFER_TRTLLM})

    with pytest.raises(ValueError, match="TRTLLM Gen MoE.*not supported"):
        select_mxfp8_moe_backend(make_dummy_moe_config())


def test_gb10_auto_mxfp8_moe_reports_trtllm_and_marlin_rejection(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_fp8_backend_support(
        monkeypatch,
        {
            Fp8MoeBackend.FLASHINFER_TRTLLM,
            Fp8MoeBackend.MARLIN,
        },
    )

    with pytest.raises(ValueError, match="MXFP8 MoE fallback.*not supported"):
        select_mxfp8_moe_backend(make_dummy_moe_config())


def test_gb10_auto_wna16_moe_skips_trtllm_for_marlin(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    kernel_by_backend = _mock_wna16_backend_support(
        monkeypatch,
        {
            WNA16MoEBackend.FLASHINFER_TRTLLM,
            WNA16MoEBackend.MARLIN,
        },
    )

    backend, experts_cls = select_wna16_moe_backend(
        make_dummy_moe_config(),
        weight_key=kInt4Static32,
    )

    assert backend == WNA16MoEBackend.MARLIN
    assert experts_cls is kernel_by_backend[WNA16MoEBackend.MARLIN]


def test_gb10_auto_wna16_moe_reports_trtllm_rejection(monkeypatch):
    _mock_sm12x_platform(monkeypatch)
    _mock_wna16_backend_support(
        monkeypatch,
        {WNA16MoEBackend.FLASHINFER_TRTLLM},
    )

    with pytest.raises(NotImplementedError, match="TRTLLM Gen MoE.*not supported"):
        select_wna16_moe_backend(
            make_dummy_moe_config(),
            weight_key=kInt4Static32,
        )
