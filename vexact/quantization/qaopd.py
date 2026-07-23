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

"""Quantization-Aware On-Policy Distillation (QAOPD) loss helpers.

Implements the NeMo On-Policy QAD / DistillationLossFn semantics for the
``zero_outside_topk=False`` path used by the GSM8K recipes:

1. Gather student logits at teacher top-k indices.
2. Re-normalize both student and teacher distributions **within the top-k**.
3. Compute forward / reverse / mixed KL (default mixed, alpha=0.5).

This helper itself has no temperature, CE, or RL/task-reward terms; teacher
tensors must already be detached. Task/verifier rewards are orthogonal and,
when enabled, are combined outside this module by VeRL
``distillation_ppo_loss`` (``final = pg + coef * distill``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


QAOPD_KL_TYPES = ("forward", "reverse", "mixed")
QAOPD_LOSS_MODE = "qaopd_mixed_kl_topk"


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _normalize_kl_type(kl_type: str) -> str:
    normalized = str(kl_type).strip().lower()
    if normalized not in QAOPD_KL_TYPES:
        raise ValueError(
            f"QAOPDConfig.kl_type must be one of {list(QAOPD_KL_TYPES)}, got {kl_type!r}."
        )
    return normalized


@dataclass
class QAOPDConfig:
    """Runtime QAOPD configuration (training-side env vars + Hydra overrides).

    Recognized variables (prefix ``VEXACT_QAOPD_`` by default):
        {prefix}ENABLE           -> enable           (bool, default False)
        {prefix}KL_TYPE          -> kl_type          (str, default ``mixed``)
        {prefix}MIXED_KL_WEIGHT  -> mixed_kl_weight  (float, default 0.5)
        {prefix}TOPK             -> topk             (int, default 64)
    """

    enable: bool = False
    kl_type: str = "mixed"
    mixed_kl_weight: float = 0.5
    topk: int = 64

    def __post_init__(self):
        self.kl_type = _normalize_kl_type(self.kl_type)
        self.mixed_kl_weight = float(self.mixed_kl_weight)
        self.topk = int(self.topk)
        if not (0.0 <= self.mixed_kl_weight <= 1.0):
            raise ValueError(
                f"QAOPDConfig.mixed_kl_weight must be in [0, 1], got {self.mixed_kl_weight}."
            )
        if self.topk <= 0:
            raise ValueError(f"QAOPDConfig.topk must be > 0, got {self.topk}.")

    @classmethod
    def from_env(cls, prefix: str = "VEXACT_QAOPD_") -> QAOPDConfig:
        enable = _env_flag(f"{prefix}ENABLE", False)
        kl_raw = os.environ.get(f"{prefix}KL_TYPE")
        kl_type = (
            kl_raw.strip().lower()
            if kl_raw is not None and kl_raw.strip()
            else "mixed"
        )
        weight_raw = os.environ.get(f"{prefix}MIXED_KL_WEIGHT")
        mixed_kl_weight = (
            float(weight_raw) if weight_raw is not None and weight_raw.strip() else 0.5
        )
        topk_raw = os.environ.get(f"{prefix}TOPK")
        topk = int(topk_raw) if topk_raw is not None and topk_raw.strip() else 64
        return cls(
            enable=enable,
            kl_type=kl_type,
            mixed_kl_weight=mixed_kl_weight,
            topk=topk,
        )


def renormalize_topk_log_probs(log_probs_or_logits: torch.Tensor) -> torch.Tensor:
    """Re-normalize a top-k slice to a proper distribution over k.

    Accepts either raw logits or full-vocab log-probs restricted to the top-k
    indices. ``log_softmax`` over the last dim is correct in both cases.
    """
    return F.log_softmax(log_probs_or_logits.float(), dim=-1)


def gather_student_topk_logits(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather student logits at teacher top-k token ids.

    Args:
        student_logits: ``[..., V]``
        teacher_topk_ids: ``[..., K]`` (same leading dims as student_logits)

    Returns:
        ``[..., K]`` student logits at the teacher indices.
    """
    if teacher_topk_ids.shape[-1] <= 0:
        raise ValueError(
            f"topk must be positive, got {teacher_topk_ids.shape[-1]}. "
            "topk=0 is not supported."
        )
    if student_logits.shape[:-1] != teacher_topk_ids.shape[:-1]:
        raise ValueError(
            "student_logits and teacher_topk_ids leading dims must match: "
            f"{tuple(student_logits.shape[:-1])} vs {tuple(teacher_topk_ids.shape[:-1])}."
        )
    return student_logits.gather(dim=-1, index=teacher_topk_ids.long())


def compute_qaopd_token_kl(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    *,
    kl_type: str = "mixed",
    mixed_kl_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token KL between already top-k-renormalized log-probs.

    Args:
        student_topk_log_probs: ``[..., K]``, already ``log_softmax`` over K.
        teacher_topk_log_probs: ``[..., K]``, already ``log_softmax`` over K,
            detached / no-grad.
        kl_type: ``forward`` | ``reverse`` | ``mixed``.
        mixed_kl_weight: weight on forward KL when ``kl_type=mixed``.

    Returns:
        ``(per_token_kl, components)`` where ``per_token_kl`` is ``[...]`` and
        ``components`` contains ``forward_kl``, ``reverse_kl``, ``student_mass``,
        ``teacher_mass`` (masses are computed before renormalization is assumed
        complete, so they should be ~1).
    """
    kl_type = _normalize_kl_type(kl_type)
    if not (0.0 <= float(mixed_kl_weight) <= 1.0):
        raise ValueError(f"mixed_kl_weight must be in [0, 1], got {mixed_kl_weight}.")

    # FP32 for numerical stability (matches NeMo DistillationLossFn).
    s_logp = student_topk_log_probs.float()
    t_logp = teacher_topk_log_probs.float()
    if t_logp.requires_grad:
        t_logp = t_logp.detach()

    s_prob = s_logp.exp()
    t_prob = t_logp.exp()

    forward_kl = (t_prob * (t_logp - s_logp)).sum(dim=-1)
    reverse_kl = (s_prob * (s_logp - t_logp)).sum(dim=-1)

    if kl_type == "forward":
        per_token = forward_kl
    elif kl_type == "reverse":
        per_token = reverse_kl
    else:
        per_token = (
            float(mixed_kl_weight) * forward_kl
            + (1.0 - float(mixed_kl_weight)) * reverse_kl
        )

    components = {
        "forward_kl": forward_kl,
        "reverse_kl": reverse_kl,
        "student_mass": s_prob.sum(dim=-1),
        "teacher_mass": t_prob.sum(dim=-1),
    }
    return per_token, components


def compute_qaopd_topk_loss(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    *,
    kl_type: str = "mixed",
    mixed_kl_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """NeMo-compatible top-k QAOPD loss from student full-vocab logits.

    Teacher side accepts either full-vocab log-probs restricted to top-k or
    raw top-k logits; both are re-normalized within k via ``log_softmax``.

    Returns a dict with:
      - ``distillation_losses``: per-token KL ``[...,]``
      - ``student_mass`` / ``teacher_mass``: mass after re-normalization (~1)
      - ``forward_kl`` / ``reverse_kl``: component per-token KLs
    """
    student_topk_logits = gather_student_topk_logits(student_logits, teacher_topk_ids)
    student_topk_log_probs = renormalize_topk_log_probs(student_topk_logits)
    teacher_renorm = renormalize_topk_log_probs(teacher_topk_log_probs)

    per_token, components = compute_qaopd_token_kl(
        student_topk_log_probs,
        teacher_renorm,
        kl_type=kl_type,
        mixed_kl_weight=mixed_kl_weight,
    )
    return {
        "distillation_losses": per_token,
        "student_mass": components["student_mass"],
        "teacher_mass": components["teacher_mass"],
        "forward_kl": components["forward_kl"],
        "reverse_kl": components["reverse_kl"],
    }
