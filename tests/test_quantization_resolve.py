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
import torch

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


def test_calibrate_defaults_by_mode():
    assert QATConfig(enable=True, mode="w4a4").calibrate is True
    assert QATConfig(enable=True, mode="w4a16").calibrate is False


def test_w4a4_rejects_calibrate_false():
    with pytest.raises(ValueError, match="calibrate=True"):
        QATConfig(enable=True, mode="w4a4", calibrate=False)


def test_random_calib_forward_loop_forces_eager_attn():
    from vexact.quantization.quantize import _random_calibration_forward_loop

    class _Cfg:
        def __init__(self):
            self._attn_implementation = "fa-invariant-cute"

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.seen = []

        def forward(self, input_ids=None, **kwargs):
            self.seen.append(self.config._attn_implementation)
            return input_ids

    model = _Model()
    loop = _random_calibration_forward_loop(QATConfig(enable=True, mode="w4a4"))
    loop(model)
    assert model.seen == ["eager"]
    assert model.config._attn_implementation == "fa-invariant-cute"


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


def test_maybe_enable_cpa_after_qat_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("VEXACT_CPA_ENABLE", raising=False)
    qat_cfg = QATConfig(enable=True, mode="w4a4")
    assert fsdp_enable_qat.maybe_enable_cpa_after_qat(True, qat_cfg) is False


def test_maybe_enable_cpa_after_qat_requires_qat_installed(monkeypatch):
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    qat_cfg = QATConfig(enable=True, mode="w4a4")
    with pytest.raises(RuntimeError, match="requires a successful W4A4 QAT"):
        fsdp_enable_qat.maybe_enable_cpa_after_qat(False, qat_cfg)


def test_maybe_enable_cpa_after_qat_rejects_w4a16(monkeypatch):
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    qat_cfg = QATConfig(enable=True, mode="w4a16")
    with pytest.raises(RuntimeError, match="only mode='w4a4'"):
        fsdp_enable_qat.maybe_enable_cpa_after_qat(True, qat_cfg)


def test_maybe_enable_cpa_after_qat_rejects_qaopd(monkeypatch):
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAOPD_ENABLE", "1")
    qat_cfg = QATConfig(enable=True, mode="w4a4")
    with pytest.raises(RuntimeError, match="conflicts with VEXACT_QAOPD_ENABLE"):
        fsdp_enable_qat.maybe_enable_cpa_after_qat(True, qat_cfg)


def test_maybe_enable_cpa_after_qat_installs_on_w4a4(monkeypatch):
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    monkeypatch.setenv("VEXACT_CPA_COEF", "0.002")
    monkeypatch.setenv("VEXACT_CPA_LOSS_TYPE", "abs_logprob")
    monkeypatch.delenv("VEXACT_QAOPD_ENABLE", raising=False)
    qat_cfg = QATConfig(enable=True, mode="w4a4")
    with patch("vexact.integrations.verl.fsdp_enable_cpa.enable_training_cpa", return_value=True) as enable:
        assert fsdp_enable_qat.maybe_enable_cpa_after_qat(True, qat_cfg) is True
        enable.assert_called_once()
        cpa_cfg = enable.call_args.args[0]
        assert cpa_cfg.enable is True
        assert cpa_cfg.coef == pytest.approx(0.002)
        assert cpa_cfg.loss_type == "abs_logprob"


def test_enable_training_qat_invokes_cpa_after_success(monkeypatch):
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a4")
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    with (
        patch.object(fsdp_enable_qat, "_patch_all_build_foundation_model", return_value=1),
        patch.object(fsdp_enable_qat, "_patch_all_weight_loaders", return_value=1),
        patch.object(fsdp_enable_qat, "maybe_enable_cpa_after_qat", return_value=True) as cpa,
    ):
        assert fsdp_enable_qat.enable_training_qat() is True
        cpa.assert_called_once()
        args, kwargs = cpa.call_args
        assert kwargs.get("qat_installed", args[0] if args else None) is True or args[0] is True


def test_enable_training_qat_cpa_fails_when_qat_patch_missing(monkeypatch):
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a4")
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    with (
        patch.object(fsdp_enable_qat, "_patch_all_build_foundation_model", return_value=0),
        patch.object(fsdp_enable_qat, "_patch_all_weight_loaders", return_value=0),
    ):
        with pytest.raises(RuntimeError, match="requires a successful W4A4 QAT"):
            fsdp_enable_qat.enable_training_qat()
