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

"""Cross-Precision Alignment (CPA) helpers for W4A4 QAT training.

CPA aligns the quantized (student) policy log-probs against a same-weight BF16
teacher forward with fake-quant disabled. Supported objectives match the
reference VERL self-distill losses: ``low_var_kl`` (default), ``mse``, and
``abs_logprob``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from torch import nn

CPA_LOSS_TYPES = ("low_var_kl", "mse", "abs_logprob")


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _normalize_loss_type(loss_type: str) -> str:
    normalized = str(loss_type).strip().lower()
    if normalized not in CPA_LOSS_TYPES:
        raise ValueError(
            "CPAConfig.loss_type must be one of "
            f"{list(CPA_LOSS_TYPES)}, got {loss_type!r}."
        )
    return normalized


@dataclass
class CPAConfig:
    """Runtime CPA configuration (training-side env vars only).

    Recognized variables (prefix ``VEXACT_CPA_`` by default):
        {prefix}ENABLE    -> enable    (bool, default False)
        {prefix}COEF      -> coef      (float, default 0.001; must be > 0 when enable)
        {prefix}LOSS_TYPE -> loss_type (str, default ``low_var_kl``;
                             one of ``low_var_kl`` / ``mse`` / ``abs_logprob``)
    """

    enable: bool = False
    coef: float = 0.001
    loss_type: str = "low_var_kl"

    def __post_init__(self):
        self.coef = float(self.coef)
        self.loss_type = _normalize_loss_type(self.loss_type)
        if self.enable and self.coef <= 0.0:
            raise ValueError(
                f"CPAConfig.coef must be > 0 when enable=True (got {self.coef})."
            )

    @classmethod
    def from_env(cls, prefix: str = "VEXACT_CPA_") -> CPAConfig:
        enable = _env_flag(f"{prefix}ENABLE", False)
        coef_raw = os.environ.get(f"{prefix}COEF")
        coef = float(coef_raw) if coef_raw is not None and coef_raw.strip() else 0.001
        loss_raw = os.environ.get(f"{prefix}LOSS_TYPE")
        loss_type = (
            loss_raw.strip().lower()
            if loss_raw is not None and loss_raw.strip()
            else "low_var_kl"
        )
        return cls(enable=enable, coef=coef, loss_type=loss_type)


def compute_cpa_token_objective(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    *,
    loss_type: str = "low_var_kl",
) -> torch.Tensor:
    """Per-token CPA objective between teacher and student log-probs.

    Aligns with tempref VERL ``self_distill_loss_type``:
      - ``low_var_kl``: ``exp(Δ)-Δ-1`` with ``Δ=clamp(teacher-student)``
      - ``mse``: ``(student-teacher)^2``
      - ``abs_logprob``: ``|student-teacher|``
    """
    loss_type = _normalize_loss_type(loss_type)
    if loss_type == "abs_logprob":
        return torch.abs(student_log_prob - teacher_log_prob)
    if loss_type == "mse":
        return torch.square(student_log_prob - teacher_log_prob)
    # Teacher is treated as the fixed reference: KL(teacher || student) approx.
    delta = (teacher_log_prob - student_log_prob).clamp(min=-20.0, max=20.0)
    token_obj = torch.exp(delta) - delta - 1.0
    return token_obj.clamp(min=-10.0, max=10.0)


def compute_cpa_loss(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    batch_num_tokens: float | int,
    dp_size: int = 1,
    loss_type: str = "low_var_kl",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Aggregate CPA objective with VERL FSDP global-token normalization.

    ``teacher_log_prob`` must already be detached / no-grad. Gradients flow only
    through ``student_log_prob``.
    """
    loss_type = _normalize_loss_type(loss_type)
    if teacher_log_prob.requires_grad:
        teacher_log_prob = teacher_log_prob.detach()

    mask = response_mask.to(dtype=student_log_prob.dtype)
    token_obj = compute_cpa_token_objective(
        student_log_prob, teacher_log_prob, loss_type=loss_type
    )
    denom = float(batch_num_tokens)
    if denom <= 0.0:
        denom = 1.0
    cpa_loss = (token_obj * mask).sum() / denom * float(dp_size)

    eps = torch.finfo(student_log_prob.dtype).eps
    weight_sum = mask.sum()
    if float(weight_sum.detach().item()) > 0.0:
        gap_mean = ((student_log_prob - teacher_log_prob) * mask).sum() / (weight_sum + eps)
        gap_value = float(gap_mean.detach().item())
    else:
        gap_value = 0.0

    metrics: dict[str, float] = {
        "actor/cpa_loss": float(cpa_loss.detach().item()),
        "actor/cpa_logprob_gap_mean": gap_value,
    }
    return cpa_loss, metrics


def _resolve_tensor_quantizer_type(quantizer_type: Optional[type[Any]]) -> type[Any]:
    if quantizer_type is not None:
        return quantizer_type
    try:
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
    except ImportError as exc:  # pragma: no cover - depends on qat extra
        raise RuntimeError(
            "[vexact-cpa] nvidia-modelopt is required to disable fake-quant for the "
            "CPA teacher forward. Install the vexact[qat] extra."
        ) from exc
    return TensorQuantizer


@contextmanager
def temporarily_disable_fake_quant_and_eval(
    model: nn.Module,
    *,
    quantizer_type: Optional[type[Any]] = None,
    require_enabled: bool = False,
) -> Iterator[int]:
    """Disable enabled modelopt TensorQuantizers and switch ``model`` to eval.

    Yields the number of quantizers that were disabled. Previously-disabled
    quantizers are left untouched. State is restored on both normal and
    exceptional exit.
    """
    qtype = _resolve_tensor_quantizer_type(quantizer_type)
    was_training = bool(model.training)
    disabled: list[Any] = []
    try:
        for module in model.modules():
            if not isinstance(module, qtype):
                continue
            if not bool(getattr(module, "is_enabled", False)):
                continue
            module.disable()
            disabled.append(module)
        if require_enabled and not disabled:
            raise RuntimeError(
                "[vexact-cpa] CPA teacher forward found no enabled fake-quant "
                "modules. Ensure W4A4 QAT is installed on the actor before "
                "enabling CPA."
            )
        model.eval()
        yield len(disabled)
    finally:
        for module in disabled:
            module.enable()
        model.train(was_training)
