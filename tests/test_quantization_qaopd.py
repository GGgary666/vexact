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

"""Unit tests for NeMo-compatible QAOPD top-k mixed KL."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vexact.quantization.qaopd import (
    QAOPDConfig,
    compute_qaopd_token_kl,
    compute_qaopd_topk_loss,
    gather_student_topk_logits,
    renormalize_topk_log_probs,
)


def _manual_topk_renorm_kl(
    student_logits: torch.Tensor,
    teacher_logits_or_logprobs: torch.Tensor,
    teacher_ids: torch.Tensor,
    *,
    kl_type: str,
    mixed_kl_weight: float,
) -> torch.Tensor:
    """Reference implementation matching NeMo zero_outside_topk=False."""
    s_topk = student_logits.gather(-1, teacher_ids.long())
    s_logp = F.log_softmax(s_topk.float(), dim=-1)
    t_logp = F.log_softmax(teacher_logits_or_logprobs.float(), dim=-1)
    s_p, t_p = s_logp.exp(), t_logp.exp()
    fwd = (t_p * (t_logp - s_logp)).sum(-1)
    rev = (s_p * (s_logp - t_logp)).sum(-1)
    if kl_type == "forward":
        return fwd
    if kl_type == "reverse":
        return rev
    return mixed_kl_weight * fwd + (1.0 - mixed_kl_weight) * rev


def test_qaopd_config_defaults():
    cfg = QAOPDConfig.from_env()
    assert cfg.enable is False
    assert cfg.kl_type == "mixed"
    assert cfg.mixed_kl_weight == 0.5
    assert cfg.topk == 64


def test_qaopd_config_from_env(monkeypatch):
    monkeypatch.setenv("VEXACT_QAOPD_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAOPD_KL_TYPE", "forward")
    monkeypatch.setenv("VEXACT_QAOPD_MIXED_KL_WEIGHT", "0.3")
    monkeypatch.setenv("VEXACT_QAOPD_TOPK", "32")
    cfg = QAOPDConfig.from_env()
    assert cfg.enable is True
    assert cfg.kl_type == "forward"
    assert cfg.mixed_kl_weight == pytest.approx(0.3)
    assert cfg.topk == 32


def test_qaopd_config_rejects_invalid():
    with pytest.raises(ValueError, match="kl_type"):
        QAOPDConfig(kl_type="k3")
    with pytest.raises(ValueError, match="mixed_kl_weight"):
        QAOPDConfig(mixed_kl_weight=1.5)
    with pytest.raises(ValueError, match="topk"):
        QAOPDConfig(topk=0)


@pytest.mark.parametrize("kl_type", ["forward", "reverse", "mixed"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_compute_qaopd_topk_loss_matches_nemo_formula(kl_type, alpha):
    torch.manual_seed(0)
    bsz, seqlen, vocab, k = 2, 5, 32, 8
    student = torch.randn(bsz, seqlen, vocab, requires_grad=True)
    teacher_full = torch.randn(bsz, seqlen, vocab)
    teacher_vals, teacher_ids = torch.topk(teacher_full, k=k, dim=-1)
    # Teacher side as full-vocab logprobs restricted to top-k (VeRL path).
    teacher_logprobs = F.log_softmax(teacher_full, dim=-1).gather(-1, teacher_ids)

    out = compute_qaopd_topk_loss(
        student,
        teacher_logprobs,
        teacher_ids,
        kl_type=kl_type,
        mixed_kl_weight=alpha,
    )
    expected = _manual_topk_renorm_kl(
        student.detach(),
        teacher_logprobs,
        teacher_ids,
        kl_type=kl_type,
        mixed_kl_weight=alpha,
    )
    assert torch.allclose(out["distillation_losses"], expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out["student_mass"], torch.ones_like(out["student_mass"]), atol=1e-5)
    assert torch.allclose(out["teacher_mass"], torch.ones_like(out["teacher_mass"]), atol=1e-5)


def test_renormalize_topk_from_raw_logits_matches_logprobs():
    torch.manual_seed(1)
    logits = torch.randn(3, 4, 16)
    vals, ids = torch.topk(logits, k=5, dim=-1)
    from_raw = renormalize_topk_log_probs(vals)
    from_full_logp = renormalize_topk_log_probs(
        F.log_softmax(logits, dim=-1).gather(-1, ids)
    )
    assert torch.allclose(from_raw, from_full_logp, atol=1e-5, rtol=1e-5)


def test_gather_rejects_non_positive_topk():
    student = torch.randn(1, 2, 8)
    ids = torch.zeros(1, 2, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="topk must be positive"):
        gather_student_topk_logits(student, ids)


def test_student_only_gradient():
    torch.manual_seed(2)
    student = torch.randn(2, 3, 20, requires_grad=True)
    teacher = torch.randn(2, 3, 20)
    vals, ids = torch.topk(teacher, k=4, dim=-1)
    teacher_logprobs = F.log_softmax(teacher, dim=-1).gather(-1, ids)
    teacher_logprobs = teacher_logprobs.clone().requires_grad_(True)

    out = compute_qaopd_topk_loss(
        student, teacher_logprobs, ids, kl_type="mixed", mixed_kl_weight=0.5
    )
    out["distillation_losses"].sum().backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0
    assert teacher_logprobs.grad is None


def test_alpha_boundaries():
    torch.manual_seed(3)
    student = torch.randn(1, 4, 16)
    teacher = torch.randn(1, 4, 16)
    vals, ids = torch.topk(teacher, k=4, dim=-1)
    t_logp = F.log_softmax(teacher, dim=-1).gather(-1, ids)

    fwd = compute_qaopd_topk_loss(student, t_logp, ids, kl_type="mixed", mixed_kl_weight=1.0)
    rev = compute_qaopd_topk_loss(student, t_logp, ids, kl_type="mixed", mixed_kl_weight=0.0)
    pure_fwd = compute_qaopd_topk_loss(student, t_logp, ids, kl_type="forward")
    pure_rev = compute_qaopd_topk_loss(student, t_logp, ids, kl_type="reverse")
    assert torch.allclose(fwd["distillation_losses"], pure_fwd["distillation_losses"])
    assert torch.allclose(rev["distillation_losses"], pure_rev["distillation_losses"])


def test_extreme_logits_finite():
    student = torch.ones(1, 2, 10) * 100.0
    teacher = torch.ones(1, 2, 10) * -100.0
    vals, ids = torch.topk(teacher + torch.arange(10).float(), k=3, dim=-1)
    # Construct extreme but valid teacher logprobs on top-k.
    t_logp = F.log_softmax(torch.randn(1, 2, 3) * 50, dim=-1)
    out = compute_qaopd_topk_loss(student, t_logp, ids, kl_type="mixed")
    assert torch.isfinite(out["distillation_losses"]).all()


def test_compute_qaopd_token_kl_fp32_accumulation():
    s = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    t = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    s_logp = F.log_softmax(s.float(), dim=-1)
    t_logp = F.log_softmax(t.float(), dim=-1)
    per_token, comps = compute_qaopd_token_kl(s_logp, t_logp, kl_type="mixed", mixed_kl_weight=0.5)
    assert per_token.dtype == torch.float32
    assert comps["forward_kl"].dtype == torch.float32
