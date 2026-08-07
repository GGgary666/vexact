# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ModelOpt → vLLM-aligned NVFP4 kernel integration."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vexact.quantization import QATConfig
from vexact.quantization.quantize import _maybe_install_vllm_nvfp4_kernel

_HAS_MODELOPT = importlib.util.find_spec("modelopt") is not None
_HAS_VLLM = importlib.util.find_spec("vllm") is not None


def test_qat_config_vllm_nvfp4_kernel_default():
    cfg = QATConfig(enable=True, mode="w4a16")
    assert cfg.use_vllm_nvfp4_kernel is True


def test_qat_config_vllm_nvfp4_kernel_from_env(monkeypatch):
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a16")
    monkeypatch.setenv("VEXACT_QAT_VLLM_NVFP4_KERNEL", "0")
    cfg = QATConfig.from_env()
    assert cfg.use_vllm_nvfp4_kernel is False

    monkeypatch.setenv("VEXACT_QAT_VLLM_NVFP4_KERNEL", "1")
    cfg = QATConfig.from_env()
    assert cfg.use_vllm_nvfp4_kernel is True


def test_maybe_install_skips_when_disabled():
    cfg = QATConfig(enable=True, mode="w4a16", use_vllm_nvfp4_kernel=False)
    with patch(
        "vexact.quantization.nvfp4.install_vllm_aligned_nvfp4_kernel"
    ) as install:
        _maybe_install_vllm_nvfp4_kernel(cfg)
        install.assert_not_called()


def test_maybe_install_calls_patch_for_nvfp4():
    cfg = QATConfig(enable=True, mode="w4a4", allow_random_calib=True)
    with patch(
        "vexact.quantization.nvfp4.install_vllm_aligned_nvfp4_kernel",
        return_value=True,
    ) as install:
        _maybe_install_vllm_nvfp4_kernel(cfg)
        install.assert_called_once()


@pytest.mark.skipif(not (_HAS_MODELOPT and _HAS_VLLM), reason="modelopt+vllm required")
def test_install_and_uninstall_modelopt_patch():
    from vexact.quantization.nvfp4.modelopt_patch import (
        fp4_fake_quant_block_vllm,
        install_vllm_aligned_nvfp4_kernel,
        is_vllm_aligned_nvfp4_kernel_installed,
        uninstall_vllm_aligned_nvfp4_kernel,
    )
    import modelopt.torch.quantization.triton as mtq_triton

    if not hasattr(mtq_triton, "fp4_fake_quant_block"):
        pytest.skip("ModelOpt fp4_fake_quant_block unavailable")

    uninstall_vllm_aligned_nvfp4_kernel()
    original = mtq_triton.fp4_fake_quant_block

    assert install_vllm_aligned_nvfp4_kernel(force=True) is True
    assert is_vllm_aligned_nvfp4_kernel_installed()
    assert mtq_triton.fp4_fake_quant_block is fp4_fake_quant_block_vllm

    assert uninstall_vllm_aligned_nvfp4_kernel() is True
    assert mtq_triton.fp4_fake_quant_block is original
    assert not is_vllm_aligned_nvfp4_kernel_installed()


@pytest.mark.skipif(not _HAS_MODELOPT, reason="modelopt not installed")
def test_quantize_model_installs_patch_when_already_quantized():
    from vexact.quantization.quantize import quantize_model

    model = MagicMock()
    cfg = QATConfig(enable=True, mode="w4a16", use_vllm_nvfp4_kernel=True)
    with (
        patch("vexact.quantization.quantize.is_model_quantized", return_value=True),
        patch(
            "vexact.quantization.quantize._maybe_install_vllm_nvfp4_kernel"
        ) as maybe_install,
        patch(
            "vexact.quantization.quantize._require_modelopt",
            return_value=SimpleNamespace(),
        ),
    ):
        out = quantize_model(model, cfg)
        assert out is model
        maybe_install.assert_called_once_with(cfg)
