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

import importlib.util
from unittest.mock import patch

import pytest

from vexact.integrations.verl import fsdp_enable_qat
from vexact.quantization import QATConfig

_HAS_MODELOPT = importlib.util.find_spec("modelopt") is not None


def _get_quantizer_entry(quant_cfg, quantizer_name: str):
    if isinstance(quant_cfg, dict):
        return quant_cfg.get(quantizer_name)
    for entry in reversed(quant_cfg):
        if entry.get("quantizer_name") == quantizer_name:
            return entry
    return None


def _has_quantizer_entry(quant_cfg, quantizer_name: str) -> bool:
    return _get_quantizer_entry(quant_cfg, quantizer_name) is not None


def test_effective_skip_ref_quantization_defaults():
    assert QATConfig(enable=True, mode="w4a16").effective_skip_ref_quantization is True
    assert QATConfig(enable=True, mode="w4a4").effective_skip_ref_quantization is False
    assert QATConfig(enable=True, mode="w4a16", skip_ref_quantization=False).effective_skip_ref_quantization is False


def test_should_skip_ref_qat_only_in_ref_context():
    from vexact.integrations.verl import fsdp_enable_qat

    cfg = QATConfig(enable=True, mode="w4a16", skip_ref_quantization=True)
    with patch.object(fsdp_enable_qat, "_detect_ref_policy_build_context", return_value=False):
        assert fsdp_enable_qat._should_skip_ref_qat(cfg) is False
    with patch.object(fsdp_enable_qat, "_detect_ref_policy_build_context", return_value=True):
        assert fsdp_enable_qat._should_skip_ref_qat(cfg) is True


def test_should_skip_ref_qat_respects_disable_flag():
    from vexact.integrations.verl import fsdp_enable_qat

    cfg = QATConfig(enable=True, mode="w4a16", skip_ref_quantization=False)
    with patch.object(fsdp_enable_qat, "_detect_ref_policy_build_context", return_value=True):
        assert fsdp_enable_qat._should_skip_ref_qat(cfg) is False


def test_apply_qat_raises_on_failure():
    cfg = QATConfig(enable=True, mode="w4a16")
    with patch.object(fsdp_enable_qat, "quantize_model", side_effect=ValueError("quant failed")):
        with pytest.raises(RuntimeError, match="train/rollout quantization mismatch"):
            fsdp_enable_qat._apply_qat(object(), cfg, "unit-test")


@pytest.mark.skipif(not _HAS_MODELOPT, reason="requires nvidia-modelopt (qat extra)")
def test_w4a16_disables_input_quantizer_with_exact_wildcard():
    from vexact.quantization import resolve_quant_cfg
    from vexact.quantization.quantize import _W4A16_INPUT_QUANTIZER_PATTERN

    cfg = QATConfig(enable=True, mode="w4a16")
    quant_cfg = resolve_quant_cfg(cfg)["quant_cfg"]

    assert _get_quantizer_entry(quant_cfg, _W4A16_INPUT_QUANTIZER_PATTERN)["enable"] is False
    assert not _has_quantizer_entry(quant_cfg, "*input_quantizer*")
    assert _get_quantizer_entry(quant_cfg, "*weight_quantizer").get("enable") is not False


@pytest.mark.skipif(not _HAS_MODELOPT, reason="requires nvidia-modelopt (qat extra)")
def test_w4a4_keeps_input_quantizer_enabled():
    from vexact.quantization import resolve_quant_cfg
    from vexact.quantization.quantize import _W4A16_INPUT_QUANTIZER_PATTERN

    cfg = QATConfig(enable=True, mode="w4a4")
    quant_cfg = resolve_quant_cfg(cfg)["quant_cfg"]

    entry = _get_quantizer_entry(quant_cfg, _W4A16_INPUT_QUANTIZER_PATTERN)
    assert entry is None or entry.get("enable") is not False
    assert _has_quantizer_entry(quant_cfg, "*weight_quantizer")


@pytest.mark.skipif(not _HAS_MODELOPT, reason="requires nvidia-modelopt (qat extra)")
def test_w4a16_quant_cfg_override_skips_implicit_input_disable(caplog):
    from vexact.quantization import resolve_quant_cfg
    from vexact.quantization.quantize import _W4A16_INPUT_QUANTIZER_PATTERN

    cfg = QATConfig(enable=True, mode="w4a16", quant_cfg="NVFP4_DEFAULT_CFG")
    with caplog.at_level("WARNING"):
        quant_cfg = resolve_quant_cfg(cfg)["quant_cfg"]

    entry = _get_quantizer_entry(quant_cfg, _W4A16_INPUT_QUANTIZER_PATTERN)
    assert entry is None or entry.get("enable") is not False
    assert "implicit *input_quantizer disable is skipped" in caplog.text


@pytest.mark.skipif(not _HAS_MODELOPT, reason="requires nvidia-modelopt (qat extra)")
def test_w4a16_respects_ignore_patterns():
    from vexact.quantization import resolve_quant_cfg
    from vexact.quantization.quantize import _W4A16_INPUT_QUANTIZER_PATTERN

    cfg = QATConfig(enable=True, mode="w4a16", ignore_patterns=["lm_head"])
    quant_cfg = resolve_quant_cfg(cfg)["quant_cfg"]

    assert _get_quantizer_entry(quant_cfg, _W4A16_INPUT_QUANTIZER_PATTERN)["enable"] is False
    assert _get_quantizer_entry(quant_cfg, "*lm_head*")["enable"] is False
