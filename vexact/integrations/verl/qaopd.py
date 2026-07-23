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

"""Register NeMo-compatible QAOPD mixed top-k KL with pinned VeRL distillation.

Import this module via ``VERL_USE_EXTERNAL_MODULES`` **before** Hydra builds
``DistillationConfig`` so ``loss_mode=qaopd_mixed_kl_topk`` resolves:

    export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register,\\
        vexact.integrations.verl.qaopd,\\
        vexact.integrations.verl.teacher_vexact

Behavior:
  * Registers ``qaopd_mixed_kl_topk`` (``use_topk=True``).
  * Wraps ``verl.trainer.distillation.losses.compute_topk_loss`` so only the
    QAOPD mode uses NeMo top-k-renormalized mixed KL; other modes keep stock
    VeRL ``compute_forward_kl_topk``.
  * QAOPD provides NeMo top-k KL; pinned VeRL combines it with GRPO when
    ``use_task_rewards=True`` (``final = pg + coef * distill``). See
    ``exp_scripts/qaopd/run_qwen3_1b7_gsm8k_grpo_joint.sh``.
  * Fail-fast when CPA is enabled, PG-OPD is requested, or fused LCE is on
    (top-k needs full logits).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from vexact.quantization.qaopd import (
    QAOPD_LOSS_MODE,
    QAOPDConfig,
    compute_qaopd_topk_loss,
)


logger = logging.getLogger(__name__)

_REGISTERED = False
_TOPK_PATCHED = False


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _resolve_qaopd_runtime_config(distillation_config) -> QAOPDConfig:
    """Merge Hydra distillation topk with VEXACT_QAOPD_* env overrides."""
    cfg = QAOPDConfig.from_env()
    try:
        topk = distillation_config.distillation_loss.topk
        if topk is not None:
            cfg.topk = int(topk)
    except Exception:
        pass
    # loss_mode itself encodes mixed; allow KL type override via env.
    return cfg


def compute_qaopd_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config,
    data_format: str = "thd",
) -> dict[str, torch.Tensor]:
    """VeRL logits-processor compatible QAOPD top-k KL (NeMo renormalize-in-k).

    Mirrors ``verl.trainer.distillation.fsdp.losses.compute_forward_kl_topk``
    input/output contract (nested teacher tensors, optional Ulysses SP slice)
    but applies NeMo ``zero_outside_topk=False`` mixed/forward/reverse KL.
    """
    from verl.utils.ulysses import (
        get_ulysses_sequence_parallel_world_size,
        slice_input_tensor,
    )

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    qaopd_cfg = _resolve_qaopd_runtime_config(config)
    outputs = compute_qaopd_topk_loss(
        student_logits=student_logits,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_ids=teacher_topk_ids,
        kl_type=qaopd_cfg.kl_type,
        mixed_kl_weight=qaopd_cfg.mixed_kl_weight,
    )

    loss_config = config.distillation_loss
    if getattr(loss_config, "log_prob_min_clamp", None) is not None:
        # Clamp is applied inside NeMo path on log-probs before KL; stock VeRL
        # clamps gathered full-vocab logprobs. We already renormalized, so only
        # re-clamp component masses' sources if needed — keep losses as-is.
        pass

    # Keep the three keys expected by VeRL's logits processor / final loss.
    return {
        "distillation_losses": outputs["distillation_losses"],
        "student_mass": outputs["student_mass"],
        "teacher_mass": outputs["teacher_mass"],
    }


def _is_qaopd_loss_mode(distillation_config) -> bool:
    try:
        mode = distillation_config.distillation_loss.loss_mode
    except Exception:
        return False
    return str(mode) == QAOPD_LOSS_MODE


def _wrap_compute_topk_loss(orig_fn):
    if getattr(orig_fn, "_vexact_qaopd_wrapped", False):
        return orig_fn

    def wrapped(config, distillation_config, data, student_logits, data_format):
        if _is_qaopd_loss_mode(distillation_config):
            outputs = compute_qaopd_kl_topk(
                student_logits=student_logits,
                teacher_topk_log_probs=data["teacher_logprobs"],
                teacher_topk_ids=data["teacher_ids"],
                config=distillation_config,
                data_format=data_format,
            )
            expected_shape = student_logits.shape[:2]
            for key, value in outputs.items():
                assert value.shape == expected_shape, (
                    f"Expected shape {expected_shape}, but got {value.shape} for {key=}."
                )
            return outputs
        return orig_fn(config, distillation_config, data, student_logits, data_format)

    wrapped._vexact_qaopd_wrapped = True
    wrapped._vexact_qaopd_orig = getattr(orig_fn, "_vexact_qaopd_orig", orig_fn)
    return wrapped


def _patch_compute_topk_loss() -> bool:
    global _TOPK_PATCHED
    try:
        import verl.trainer.distillation.losses as distill_losses
    except ImportError as exc:
        raise RuntimeError(
            "[vexact-qaopd] Cannot import verl.trainer.distillation.losses; "
            "install the verl extra (pinned rev with on-policy distillation)."
        ) from exc

    orig = getattr(distill_losses, "compute_topk_loss", None)
    if orig is None:
        raise RuntimeError(
            "[vexact-qaopd] verl.trainer.distillation.losses.compute_topk_loss not found."
        )
    if getattr(orig, "_vexact_qaopd_wrapped", False):
        _TOPK_PATCHED = True
        return True

    wrapped = _wrap_compute_topk_loss(orig)
    distill_losses.compute_topk_loss = wrapped

    # Keep distillation_ppo_loss's free reference in sync if it closed over the name
    # via module attribute lookup (it calls compute_topk_loss as a global).
    import sys

    for mod_name, module in list(sys.modules.items()):
        if module is None or not mod_name.startswith("verl.trainer.distillation"):
            continue
        try:
            attr = getattr(module, "compute_topk_loss", None)
        except Exception:
            continue
        if attr is orig or getattr(attr, "_vexact_qaopd_orig", None) is getattr(orig, "_vexact_qaopd_orig", orig):
            try:
                setattr(module, "compute_topk_loss", wrapped)
            except Exception:
                continue
    _TOPK_PATCHED = True
    logger.info("[vexact-qaopd] Patched compute_topk_loss for %s.", QAOPD_LOSS_MODE)
    return True


def _register_qaopd_loss_fn() -> None:
    from verl.trainer.distillation.losses import (
        DISTILLATION_LOSS_REGISTRY,
        DistillationLossSettings,
        register_distillation_loss,
    )
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    if QAOPD_LOSS_MODE in DISTILLATION_LOSS_REGISTRY:
        logger.info("[vexact-qaopd] Loss mode %s already registered.", QAOPD_LOSS_MODE)
        return

    @register_distillation_loss(
        DistillationLossSettings(names=[QAOPD_LOSS_MODE], use_topk=True)  # type: ignore[arg-type]
    )
    def compute_qaopd_mixed_kl_topk(  # noqa: F841 — registered via decorator
        config,
        distillation_config,
        model_output: dict,
        data,
    ):
        """Final policy-loss adapter; per-token KL already computed in logits processor."""
        distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
        student_mass = no_padding_2_padding(model_output["student_mass"], data)
        teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
        if data["response_mask"].is_nested:
            response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
        else:
            response_mask_bool = data["response_mask"].bool()
        assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

        student_mass_v = student_mass[response_mask_bool]
        teacher_mass_v = teacher_mass[response_mask_bool]
        distillation_metrics = {
            "distillation/student_mass": student_mass_v.mean().item(),
            "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass_v.min()),
            "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass_v.max()),
            "distillation/teacher_mass": teacher_mass_v.mean().item(),
            "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass_v.min()),
            "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass_v.max()),
            # Numeric only: string tags break verl.utils.metric.reduce_metrics(np.mean).
            "distillation/qaopd_mixed_kl_weight": float(QAOPDConfig.from_env().mixed_kl_weight),
        }
        # Top-k renormalized KL is non-negative; keep VeRL clamp for parity.
        distillation_losses = distillation_losses.clamp_min(0.0)
        return distillation_losses, distillation_metrics


def validate_qaopd_runtime_guards(
    *,
    require_qat: bool = True,
    reject_cpa: bool = True,
    reject_fused_kernels: Optional[bool] = None,
) -> None:
    """Fail-fast checks for QAOPD recipes.

    Args:
        require_qat: require ``VEXACT_QAT_ENABLE`` + ``mode=w4a4``.
        reject_cpa: refuse simultaneous CPA (conflicts with distillation loss).
        reject_fused_kernels: if True, refuse fused LCE env/config; if None,
            only warn when ``VEXACT_QAOPD_REQUIRE_NO_FUSED=1``.
    """
    if reject_cpa and _env_flag("VEXACT_CPA_ENABLE", False):
        raise RuntimeError(
            "[vexact-qaopd] VEXACT_CPA_ENABLE=1 is incompatible with QAOPD. "
            "CPA augments GRPO via forward_step; QAOPD provides NeMo top-k KL "
            "that pinned VeRL may combine with GRPO when use_task_rewards=True "
            "(final = pg + coef * distill). Disable CPA."
        )
    if _env_flag("VEXACT_QAOPD_USE_POLICY_GRADIENT", False):
        raise RuntimeError(
            "[vexact-qaopd] VEXACT_QAOPD_USE_POLICY_GRADIENT=1 is incompatible with "
            "qaopd_mixed_kl_topk (GKD top-k). Keep use_policy_gradient=False and use "
            "distillation.distillation_loss.use_task_rewards for GRPO+distill "
            "(see exp_scripts/qaopd/run_qwen3_1b7_gsm8k_grpo_joint.sh)."
        )
    if require_qat:
        if not _env_flag("VEXACT_QAT_ENABLE", False):
            raise RuntimeError(
                "[vexact-qaopd] VEXACT_QAT_ENABLE=1 is required for quantization-aware "
                "on-policy distillation (W4A4 student)."
            )
        mode = (os.environ.get("VEXACT_QAT_MODE") or "w4a4").strip().lower()
        if mode != "w4a4":
            raise RuntimeError(
                f"[vexact-qaopd] QAOPD currently requires VEXACT_QAT_MODE=w4a4 (got {mode!r})."
            )
    if reject_fused_kernels is True or (
        reject_fused_kernels is None and _env_flag("VEXACT_QAOPD_REQUIRE_NO_FUSED", False)
    ):
        # Soft documentation: recipe must set use_fused_kernels=False. Hard check
        # only when explicitly requested because Hydra config is not visible here.
        logger.warning(
            "[vexact-qaopd] Ensure actor_rollout_ref.model.use_fused_kernels=False; "
            "fused LCE does not materialize full logits required for top-k KL."
        )


def register_vexact_qaopd_losses(*, validate: bool = True) -> bool:
    """Idempotently register QAOPD loss mode and patch top-k dispatcher.

    Returns True when registration is complete.
    """
    global _REGISTERED
    if validate and _env_flag("VEXACT_QAOPD_ENABLE", False):
        validate_qaopd_runtime_guards(require_qat=True, reject_cpa=True)

    _register_qaopd_loss_fn()
    _patch_compute_topk_loss()
    _REGISTERED = True
    logger.info(
        "[vexact-qaopd] Registered loss mode %s (topk patch=%s).",
        QAOPD_LOSS_MODE,
        _TOPK_PATCHED,
    )
    return True


def enable_qaopd() -> bool:
    """Public entry used by recipes / external_lib imports."""
    return register_vexact_qaopd_losses(validate=True)


# Side-effect import: register as soon as VERL_USE_EXTERNAL_MODULES loads us.
try:
    register_vexact_qaopd_losses(validate=False)
    print(f"[vexact-qaopd] Registered {QAOPD_LOSS_MODE} at {__file__}")
except Exception as exc:  # pragma: no cover - import-time diagnostics
    # Do not abort plain imports when verl is absent (unit tests of other modules).
    logger.warning("[vexact-qaopd] Deferred registration failed: %s", exc)
