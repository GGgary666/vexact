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

import pytest
import torch
from torch import nn

from vexact.quantization.cpa import (
    CPAConfig,
    compute_cpa_loss,
    temporarily_disable_fake_quant_and_eval,
)


def test_cpa_config_defaults():
    cfg = CPAConfig.from_env()
    assert cfg.enable is False
    assert cfg.coef == 0.001
    assert cfg.loss_type == "low_var_kl"


def test_cpa_config_from_env(monkeypatch):
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    monkeypatch.setenv("VEXACT_CPA_COEF", "0.003")
    monkeypatch.setenv("VEXACT_CPA_LOSS_TYPE", "mse")
    cfg = CPAConfig.from_env()
    assert cfg.enable is True
    assert cfg.coef == pytest.approx(0.003)
    assert cfg.loss_type == "mse"


def test_cpa_config_rejects_non_positive_coef():
    with pytest.raises(ValueError, match="coef"):
        CPAConfig(enable=True, coef=0.0)
    with pytest.raises(ValueError, match="coef"):
        CPAConfig(enable=True, coef=-0.1)


def test_cpa_config_rejects_invalid_loss_type():
    with pytest.raises(ValueError, match="loss_type"):
        CPAConfig(enable=True, coef=0.001, loss_type="kl")


def test_cpa_config_from_env_rejects_non_positive_coef(monkeypatch):
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    monkeypatch.setenv("VEXACT_CPA_COEF", "0")
    with pytest.raises(ValueError, match="coef"):
        CPAConfig.from_env()


def test_cpa_config_from_env_rejects_invalid_loss_type(monkeypatch):
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    monkeypatch.setenv("VEXACT_CPA_LOSS_TYPE", "kl")
    with pytest.raises(ValueError, match="loss_type"):
        CPAConfig.from_env()


def test_compute_cpa_loss_low_var_kl_matches_formula():
    student = torch.tensor([[0.0, -1.0], [0.5, 0.25]], requires_grad=True)
    teacher = torch.tensor([[0.1, -0.5], [0.0, 0.5]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    loss, metrics = compute_cpa_loss(
        student_log_prob=student,
        teacher_log_prob=teacher,
        response_mask=mask,
        batch_num_tokens=3.0,
        dp_size=2,
        loss_type="low_var_kl",
    )

    delta = (teacher - student).clamp(-20.0, 20.0)
    token_obj = (delta.exp() - delta - 1.0).clamp(-10.0, 10.0)
    expected = (token_obj * mask).sum() / 3.0 * 2
    assert torch.allclose(loss, expected)
    assert metrics["actor/cpa_loss"] == pytest.approx(loss.detach().item())
    gap = ((student - teacher) * mask).sum() / (mask.sum() + torch.finfo(student.dtype).eps)
    assert metrics["actor/cpa_logprob_gap_mean"] == pytest.approx(gap.detach().item())


def test_compute_cpa_loss_mse_matches_formula():
    student = torch.tensor([[0.0, -1.0], [0.5, 0.25]], requires_grad=True)
    teacher = torch.tensor([[0.1, -0.5], [0.0, 0.5]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    loss, metrics = compute_cpa_loss(
        student_log_prob=student,
        teacher_log_prob=teacher,
        response_mask=mask,
        batch_num_tokens=3.0,
        dp_size=2,
        loss_type="mse",
    )

    gap = student - teacher
    expected = ((gap.square() * mask).sum() / 3.0) * 2
    assert torch.allclose(loss, expected)
    assert "actor/cpa_loss" in metrics


def test_compute_cpa_loss_abs_logprob_matches_formula():
    student = torch.tensor([[0.0, -1.0], [0.5, 0.25]], requires_grad=True)
    teacher = torch.tensor([[0.1, -0.5], [0.0, 0.5]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    loss, metrics = compute_cpa_loss(
        student_log_prob=student,
        teacher_log_prob=teacher,
        response_mask=mask,
        batch_num_tokens=3.0,
        dp_size=2,
        loss_type="abs_logprob",
    )

    gap = student - teacher
    expected = ((gap.abs() * mask).sum() / 3.0) * 2
    assert torch.allclose(loss, expected)
    assert "actor/cpa_loss" in metrics


def test_compute_cpa_loss_clamps_extreme_deltas():
    student = torch.tensor([[0.0]], requires_grad=True)
    teacher = torch.tensor([[100.0]])  # delta >> 20
    mask = torch.ones_like(student)
    loss, _ = compute_cpa_loss(
        student_log_prob=student,
        teacher_log_prob=teacher,
        response_mask=mask,
        batch_num_tokens=1.0,
        dp_size=1,
    )
    # After clamp delta=20 -> exp(20)-20-1 is huge, then clamp to 10
    assert torch.allclose(loss, torch.tensor(10.0))


def test_compute_cpa_loss_empty_mask_is_finite():
    student = torch.tensor([[0.0, 1.0]], requires_grad=True)
    teacher = torch.tensor([[0.5, 0.5]])
    mask = torch.zeros_like(student)
    loss, metrics = compute_cpa_loss(
        student_log_prob=student,
        teacher_log_prob=teacher,
        response_mask=mask,
        batch_num_tokens=1.0,
        dp_size=1,
    )
    assert torch.isfinite(loss)
    assert metrics["actor/cpa_logprob_gap_mean"] == 0.0


def test_compute_cpa_loss_student_only_gradient():
    student = torch.tensor([[0.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[0.2, -0.5]], requires_grad=True)
    mask = torch.ones_like(student)
    loss, _ = compute_cpa_loss(
        student_log_prob=student,
        teacher_log_prob=teacher,
        response_mask=mask,
        batch_num_tokens=2.0,
        dp_size=1,
    )
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None


class _FakeQuantizer(nn.Module):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self._enabled = enabled
        self.disable_calls = 0
        self.enable_calls = 0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def disable(self):
        self.disable_calls += 1
        self._enabled = False

    def enable(self):
        self.enable_calls += 1
        self._enabled = True


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_on = _FakeQuantizer(True)
        self.q_off = _FakeQuantizer(False)
        self.linear = nn.Linear(2, 2)


def test_temporarily_disable_fake_quant_and_eval_restores_state():
    model = _FakeModel()
    model.train()
    assert model.training
    assert model.q_on.is_enabled
    assert not model.q_off.is_enabled

    with temporarily_disable_fake_quant_and_eval(model, quantizer_type=_FakeQuantizer) as n_disabled:
        assert n_disabled == 1
        assert not model.training
        assert not model.q_on.is_enabled
        assert not model.q_off.is_enabled

    assert model.training
    assert model.q_on.is_enabled
    assert not model.q_off.is_enabled
    assert model.q_on.disable_calls == 1
    assert model.q_on.enable_calls == 1
    assert model.q_off.disable_calls == 0
    assert model.q_off.enable_calls == 0


def test_temporarily_disable_fake_quant_and_eval_restores_on_exception():
    model = _FakeModel()
    model.train()
    with pytest.raises(RuntimeError, match="boom"):
        with temporarily_disable_fake_quant_and_eval(model, quantizer_type=_FakeQuantizer):
            assert not model.q_on.is_enabled
            raise RuntimeError("boom")
    assert model.training
    assert model.q_on.is_enabled
    assert not model.q_off.is_enabled


def test_temporarily_disable_requires_enabled_quantizer_when_strict():
    model = _FakeModel()
    model.q_on.disable()
    with pytest.raises(RuntimeError, match="no enabled"):
        with temporarily_disable_fake_quant_and_eval(
            model, quantizer_type=_FakeQuantizer, require_enabled=True
        ):
            pass
