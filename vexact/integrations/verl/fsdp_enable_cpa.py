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

"""Training-side CPA hook for the VeOmni FSDP actor.

VeXact does not own the training loop. When imported (typically via
``fsdp_enable_qat`` after QAT is installed), this module monkey-patches
``VeOmniEngineWithLMHead.forward_step`` so each actor train micro-batch adds a
cross-precision alignment (CPA) loss:

* student: existing W4A4 QAT forward (unchanged PPO path)
* teacher: same weights, ``eval`` + all modelopt fake-quant disabled, no-grad

Config is read from ``VEXACT_CPA_*`` env vars (see :class:`CPAConfig`).
"""

from __future__ import annotations

import functools
import logging
import sys
from typing import Any, Callable, Optional

import torch

from vexact.quantization.cpa import (
    CPAConfig,
    compute_cpa_loss,
    temporarily_disable_fake_quant_and_eval,
)


logger = logging.getLogger(__name__)


def _default_device_name() -> str:
    try:
        from verl.utils.device import get_device_name

        return get_device_name()
    except Exception:
        return "cuda" if torch.cuda.is_available() else "cpu"


def _default_to_response_log_probs(log_probs, data):
    from verl.workers.utils.padding import no_padding_2_padding

    return no_padding_2_padding(log_probs, data)


def _metric_value(value: float, metrics: dict) -> Any:
    """Wrap ``value`` as verl ``Metric`` only when surrounding metrics already use it.

    Unit tests and older verl paths store plain floats; wrapping unconditionally
    breaks ``np.mean``-style reduce_metrics and CPA unit assertions.
    """
    try:
        from verl.utils.metric import Metric
    except Exception:
        return float(value)

    pg = metrics.get("actor/pg_loss")
    if isinstance(pg, Metric):
        return Metric(aggregation=pg.aggregation, value=float(value))
    return float(value)


def _unwrap_non_tensor_value(val, default=None):
    """Unwrap verl/tensordict ``NonTensorData`` wrappers to a plain Python value."""
    if val is None:
        return default
    # Prefer verl helper when available.
    try:
        from verl.utils.tensordict_utils import unwrap_non_tensor_data

        val = unwrap_non_tensor_data(val)
    except Exception:
        data_attr = getattr(val, "data", None)
        # NonTensorData exposes `.data`; avoid mistaking torch tensors.
        if data_attr is not None and type(val).__name__ == "NonTensorData":
            val = data_attr
    return default if val is None else val


def _get_non_tensor(data, key: str, default=None):
    """Read non-tensor metadata from a TensorDict / mapping.

    Always unwraps ``NonTensorData`` so callers can safely do ``int(...)`` /
    ``float(...)`` without hitting tensordict's bool-conversion ban.
    """
    try:
        from verl.utils import tensordict_utils as tu

        return _unwrap_non_tensor_value(
            tu.get_non_tensor_data(data=data, key=key, default=default),
            default=default,
        )
    except Exception:
        pass

    if hasattr(data, "get"):
        try:
            return _unwrap_non_tensor_value(data.get(key, default), default=default)
        except Exception:
            pass
    return default


def _tensor_device(obj) -> Optional[torch.device]:
    if torch.is_tensor(obj):
        return obj.device
    values = getattr(obj, "values", None)
    if callable(values):
        try:
            vals = values()
            if torch.is_tensor(vals):
                return vals.device
        except Exception:
            pass
    return None


def _resolve_compute_device(loss: torch.Tensor, model_output: dict, device_name: str):
    """Resolve the device for CPA teacher inputs / loss alignment.

    Prefer the student log-prob device (where gradients live). Only fall back to
    verl's ``get_device_id()`` when student tensors are unavailable — otherwise a
    CUDA-capable host would pull CPU unit-test tensors onto ``cuda:0``.
    """
    student = (model_output or {}).get("log_probs")
    student_device = _tensor_device(student)
    if student_device is not None:
        return student_device
    loss_device = _tensor_device(loss)
    if loss_device is not None:
        return loss_device
    try:
        from verl.utils.device import get_device_id

        return get_device_id()
    except Exception:
        pass
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _align_cpa_tensors(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ensure CPA inputs share the student log-prob device/dtype layout."""
    device = student_log_prob.device
    if teacher_log_prob.device != device:
        teacher_log_prob = teacher_log_prob.to(device=device)
    if response_mask.device != device:
        response_mask = response_mask.to(device=device)
    return student_log_prob, teacher_log_prob, response_mask


def _move_micro_batch_to_device(micro_batch, device):
    """Move micro-batch tensors onto ``device``.

    verl's ``forward_step`` does ``micro_batch = micro_batch.to(get_device_id())``
    as a *local* rebinding. CPA runs after that returns, so the caller's
    ``micro_batch`` may still be on CPU while teacher logits are on GPU —
    causing ``nested_tensor_from_jagged`` to assert ``values.device == offsets.device``.
    """
    if hasattr(micro_batch, "to"):
        try:
            return micro_batch.to(device)
        except Exception:
            pass
    if isinstance(micro_batch, dict):
        moved = {}
        for key, value in micro_batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved
    return micro_batch


def _call_prepare_model_outputs(
    engine,
    *,
    raw_output,
    output_args,
    micro_batch,
    logits_processor_func=None,
):
    """Call ``prepare_model_outputs`` across verl API variants.

    Newer verl requires ``logits_processor_func`` (used only for distillation
    top-k paths). CPA teacher forward passes ``None``.
    """
    import inspect

    kwargs = {
        "output": raw_output,
        "output_args": output_args,
        "micro_batch": micro_batch,
    }
    try:
        params = inspect.signature(engine.prepare_model_outputs).parameters
    except (TypeError, ValueError):
        params = {}
    if "logits_processor_func" in params:
        kwargs["logits_processor_func"] = logits_processor_func
    return engine.prepare_model_outputs(**kwargs)


def _run_teacher_forward(
    engine,
    micro_batch,
    *,
    device_name: str,
    quantizer_type: Optional[type[Any]],
) -> Any:
    """BF16 teacher forward with fake-quant disabled; returns raw model_output."""
    model_inputs, output_args = engine.prepare_model_inputs(micro_batch=micro_batch)
    with torch.no_grad():
        with temporarily_disable_fake_quant_and_eval(
            engine.module,
            quantizer_type=quantizer_type,
            require_enabled=True,
        ):
            with torch.autocast(device_type=device_name, dtype=torch.bfloat16, enabled=(device_name != "cpu")):
                raw_output = engine.module(**model_inputs, use_cache=False)
            teacher_output = _call_prepare_model_outputs(
                engine,
                raw_output=raw_output,
                output_args=output_args,
                micro_batch=micro_batch,
                logits_processor_func=None,
            )
    return teacher_output


def augment_train_loss_with_cpa(
    engine,
    micro_batch,
    loss: torch.Tensor,
    output: dict,
    cpa_config: CPAConfig,
    *,
    to_response_log_probs: Optional[Callable] = None,
    device_name: Optional[str] = None,
    quantizer_type: Optional[type[Any]] = None,
) -> tuple[torch.Tensor, dict]:
    """Add CPA loss to an existing train ``forward_step`` result."""
    if to_response_log_probs is None:
        to_response_log_probs = _default_to_response_log_probs
    if device_name is None:
        device_name = _default_device_name()

    model_output = output.get("model_output") or {}
    student_raw = model_output.get("log_probs")
    if student_raw is None:
        raise RuntimeError(
            "[vexact-cpa] Student model_output is missing log_probs; cannot compute CPA loss."
        )

    # Match verl forward_step: teacher inputs/offsets must live on the compute device.
    compute_device = _resolve_compute_device(loss, model_output, device_name)
    micro_batch = _move_micro_batch_to_device(micro_batch, compute_device)

    student_log_prob = to_response_log_probs(student_raw, micro_batch)
    teacher_output = _run_teacher_forward(
        engine,
        micro_batch,
        device_name=device_name,
        quantizer_type=quantizer_type,
    )
    teacher_raw = teacher_output.get("log_probs")
    if teacher_raw is None:
        raise RuntimeError(
            "[vexact-cpa] Teacher forward did not produce log_probs; cannot compute CPA loss."
        )
    teacher_log_prob = to_response_log_probs(teacher_raw, micro_batch).detach()

    response_mask = micro_batch["response_mask"]
    student_log_prob, teacher_log_prob, response_mask = _align_cpa_tensors(
        student_log_prob, teacher_log_prob, response_mask
    )
    batch_num_tokens = _get_non_tensor(micro_batch, "batch_num_tokens", default=None)
    if batch_num_tokens is None:
        batch_num_tokens = float(response_mask.to(dtype=student_log_prob.dtype).sum().item())
    else:
        batch_num_tokens = float(batch_num_tokens)
    raw_dp_size = _get_non_tensor(micro_batch, "dp_size", default=1)
    dp_size = int(raw_dp_size) if raw_dp_size is not None else 1
    if dp_size <= 0:
        dp_size = 1

    cpa_loss, cpa_metrics = compute_cpa_loss(
        student_log_prob=student_log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        batch_num_tokens=batch_num_tokens,
        dp_size=dp_size,
        loss_type=cpa_config.loss_type,
    )
    total_loss = loss + float(cpa_config.coef) * cpa_loss

    metrics = output.setdefault("metrics", {})
    metrics["actor/cpa_loss"] = _metric_value(cpa_metrics["actor/cpa_loss"], metrics)
    metrics["actor/cpa_logprob_gap_mean"] = _metric_value(
        cpa_metrics["actor/cpa_logprob_gap_mean"], metrics
    )
    metrics["actor/cpa_coef"] = float(cpa_config.coef)
    output["loss"] = float(total_loss.detach().item())
    return total_loss, output


def wrap_forward_step(
    orig_fn: Callable,
    cpa_config: CPAConfig,
    *,
    to_response_log_probs: Optional[Callable] = None,
    device_name: Optional[str] = None,
    quantizer_type: Optional[type[Any]] = None,
) -> Callable:
    """Return a CPA-augmented ``forward_step`` (idempotent)."""
    if getattr(orig_fn, "_vexact_cpa_wrapped", False):
        return orig_fn

    @functools.wraps(orig_fn)
    def wrapped(self, micro_batch, loss_function, forward_only):
        loss, output = orig_fn(self, micro_batch, loss_function, forward_only)
        if loss_function is None or forward_only:
            return loss, output
        if not cpa_config.enable:
            return loss, output
        return augment_train_loss_with_cpa(
            self,
            micro_batch,
            loss,
            output,
            cpa_config,
            to_response_log_probs=to_response_log_probs,
            device_name=device_name,
            quantizer_type=quantizer_type,
        )

    wrapped._vexact_cpa_wrapped = True
    wrapped._vexact_cpa_orig = getattr(orig_fn, "_vexact_cpa_orig", orig_fn)
    return wrapped


def _patch_veomni_forward_step(cpa_config: CPAConfig) -> int:
    mod_name = "verl.workers.engine.veomni.transformer_impl"
    veomni_impl = sys.modules.get(mod_name)
    if veomni_impl is None:
        try:
            import verl.workers.engine.veomni.transformer_impl as veomni_impl
        except ImportError as exc:
            raise RuntimeError(
                "[vexact-cpa] VEXACT_CPA_ENABLE is set but "
                "'verl.workers.engine.veomni.transformer_impl' is not importable. "
                "CPA requires the VeOmni actor path."
            ) from exc

    cls = getattr(veomni_impl, "VeOmniEngineWithLMHead", None)
    if cls is None or not hasattr(cls, "forward_step"):
        raise RuntimeError(
            "[vexact-cpa] VeOmniEngineWithLMHead.forward_step not found; cannot install CPA hook."
        )

    if getattr(cls.forward_step, "_vexact_cpa_wrapped", False):
        logger.info("[vexact-cpa] CPA forward_step hook already installed.")
        return 1

    orig = getattr(cls.forward_step, "_vexact_cpa_orig", cls.forward_step)
    wrapped = wrap_forward_step(orig, cpa_config)
    cls.forward_step = wrapped
    # Keep any already-imported aliases in sync.
    for alias_name, module in list(sys.modules.items()):
        if module is None or not alias_name.startswith("verl"):
            continue
        try:
            candidate = getattr(module, "VeOmniEngineWithLMHead", None)
        except Exception:
            continue
        if candidate is cls:
            try:
                candidate.forward_step = wrapped
            except Exception:
                continue
    logger.info("[vexact-cpa] Patched VeOmniEngineWithLMHead.forward_step for CPA.")
    return 1


def enable_training_cpa(cpa_config: Optional[CPAConfig] = None) -> bool:
    """Install the CPA forward_step hook.

    Returns True when the hook is installed. Raises on hard configuration /
    import failures so CPA cannot silently degrade to a no-op.
    """
    if cpa_config is None:
        cpa_config = CPAConfig.from_env()
    if not cpa_config.enable:
        logger.info("[vexact-cpa] VEXACT_CPA_ENABLE not set; CPA disabled.")
        return False

    logger.info(f"[vexact-cpa] Training-side CPA enabled: {cpa_config}")
    patched = _patch_veomni_forward_step(cpa_config)
    if patched <= 0:
        raise RuntimeError("[vexact-cpa] Failed to patch VeOmniEngineWithLMHead.forward_step.")
    logger.info("[vexact-cpa] Training-side CPA hook installed.")
    return True
