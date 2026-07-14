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

"""Weight pre-folding for QAT rollout (NeMo-RL QARL-style refit).

Training-side fold is the only supported weight path when QAT is enabled:

1. The training-side ``ServerAdapter`` folds weights via the actor's
   ``weight_quantizer`` (after casting FSDP float32 masters to bf16) before
   sending them to rollout.
2. Rollout never runs live ``weight_quantizer``. For ``w4a16`` the model stays
   plain Linear; for ``w4a4`` only ``input_quantizer`` remains enabled.

``prepare_rollout_prefold`` remains as a helper to fold+disable weight
quantizers on a model that already has QuantModules (e.g. unit tests / legacy
rollout-side init). Production rollout init uses training-side fold and, for
w4a4, ``disable_weight_quantizers`` only.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import torch
from torch import nn

from .config import QATConfig


logger = logging.getLogger(__name__)

_VEXACT_WQ_MAP_ATTR = "_vexact_weight_quantizer_map"

# Module-level cache for the training-side weight_quantizer_map.
# Populated by ``fsdp_enable_qat.py`` after the training model is quantized.
# Read by ``ServerAdapter.update_weights()`` to fold weights before sending
# them to rollout.
_training_weight_quantizer_map: Optional[dict[str, Any]] = None


def set_training_weight_quantizer_map(wq_map: Optional[dict[str, Any]]) -> None:
    """Cache the training model's weight_quantizer map for training-side fold."""
    global _training_weight_quantizer_map
    _training_weight_quantizer_map = wq_map


def get_training_weight_quantizer_map() -> Optional[dict[str, Any]]:
    """Return the cached training-side weight_quantizer map, or ``None``."""
    return _training_weight_quantizer_map


def _materialize_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return a dense local tensor (no-op for non-DTensor inputs)."""
    if hasattr(weight, "full_tensor"):
        return weight.full_tensor()
    if hasattr(weight, "to_local"):
        return weight.to_local()
    return weight


def _weight_for_training_side_fold(weight: torch.Tensor) -> torch.Tensor:
    """Cast sender weights to bf16 so fold matches live QuantModule Parameters.

    FSDP ``full_tensor()`` often yields float32 masters while actor live WQ sees
    bf16 Parameters. Folding the fp32 master makes rollout receive a different
    QDQ source than ``old_log_prob``.
    """
    dense = _materialize_weight(weight)
    if dense.dtype == torch.bfloat16:
        return dense
    return dense.to(dtype=torch.bfloat16)


def fold_weights_generator(
    weights: Iterable[tuple[str, torch.Tensor]],
    wq_map: dict[str, Any],
    stats: Optional[dict[str, int]] = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Wrap a weights generator to fold each weight via its paired quantizer.

    Float32 sender weights are cast to bfloat16 before folding so the QDQ source
    matches live actor Parameters.

    If ``stats`` is provided, populate it with fold counters and skip the legacy
    INFO log (caller is expected to emit a step-level summary).
    """
    folded_count = 0
    total_count = 0
    unmatched = 0
    cast_to_bf16_count = 0

    for name, weight in weights:
        total_count += 1
        wq = wq_map.get(name)
        if wq is not None:
            fold_input = _weight_for_training_side_fold(weight)
            if fold_input.dtype != getattr(weight, "dtype", fold_input.dtype):
                cast_to_bf16_count += 1
            weight = fold_weight(fold_input, wq)
            folded_count += 1
        elif name.endswith(".weight") and "norm" not in name and "embed" not in name:
            unmatched += 1
        yield name, weight

    if stats is not None:
        stats["folded"] = folded_count
        stats["total"] = total_count
        stats["unmatched_linearish"] = unmatched
        stats["cast_fp32_to_bf16"] = cast_to_bf16_count
        return

    extra = ""
    if unmatched:
        extra += f", unmatched_linearish={unmatched}"
    if cast_to_bf16_count:
        extra += f", cast_fp32_to_bf16={cast_to_bf16_count}"
    logger.info(
        "[vexact-qat] Training-side fold: folded %d/%d weight(s) via weight_quantizer"
        "%s.",
        folded_count,
        total_count,
        extra,
    )


def get_weight_quantizer_map(model: nn.Module) -> Optional[dict[str, Any]]:
    """Return the cached weight-quantizer map attached during rollout prefold setup."""
    wq_map = getattr(model, _VEXACT_WQ_MAP_ATTR, None)
    if wq_map is not None:
        return wq_map
    inner = getattr(model, "model", None)
    if inner is not None:
        return getattr(inner, _VEXACT_WQ_MAP_ATTR, None)
    return None


def _require_modelopt_quant():
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModule
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    return QuantModule, TensorQuantizer


def build_weight_quantizer_map(model: nn.Module) -> dict[str, Any]:
    """Map ``.weight`` parameter names to their paired ``weight_quantizer`` modules.

    Only maps ENABLED quantizers. Disabled quantizers (e.g. from ignore_patterns
    like ``embed_tokens`` and ``lm_head``) are excluded from the map.
    """
    QuantModule, TensorQuantizer = _require_modelopt_quant()
    name_to_param = dict(model.named_parameters())
    param_id_to_name = {id(param): name for name, param in name_to_param.items()}
    mapping: dict[str, Any] = {}

    for module in model.modules():
        if not isinstance(module, QuantModule):
            continue
        for weight, wq in module.iter_weights_for_calibration():
            if not isinstance(wq, TensorQuantizer):
                continue
            if not wq.is_enabled:
                continue
            pname = param_id_to_name.get(id(weight))
            if pname is None:
                continue
            mapping[pname] = wq

    return mapping


def count_enabled_quantizers(model: nn.Module, *, name_suffix: str) -> int:
    """Count enabled modelopt ``TensorQuantizer`` modules whose names end with ``name_suffix``."""
    _, TensorQuantizer = _require_modelopt_quant()
    return sum(
        1
        for name, module in model.named_modules()
        if name.endswith(name_suffix) and isinstance(module, TensorQuantizer) and module.is_enabled
    )


def audit_rollout_quant_state(model: nn.Module, *, context: str) -> dict[str, int]:
    """Log how many weight/input quantizers remain enabled after rollout prefold."""
    stats = {
        "weight_quantizer_enabled": count_enabled_quantizers(model, name_suffix="weight_quantizer"),
        "input_quantizer_enabled": count_enabled_quantizers(model, name_suffix="input_quantizer"),
    }
    logger.info(
        "[vexact-qat] Rollout quant audit (%s): enabled weight_quantizer=%d, "
        "enabled input_quantizer=%d.",
        context,
        stats["weight_quantizer_enabled"],
        stats["input_quantizer_enabled"],
    )
    if stats["weight_quantizer_enabled"] > 0:
        logger.warning(
            "[vexact-qat] Rollout still has %d enabled weight_quantizer module(s) after "
            "prefold (%s). Inference will re-run live NVFP4 fake quant on every forward "
            "and can be orders of magnitude slower than folded weights.",
            stats["weight_quantizer_enabled"],
            context,
        )
    return stats


def fold_weight(weight: torch.Tensor, weight_quantizer: Any) -> torch.Tensor:
    """Apply a modelopt weight quantizer to produce a pre-folded weight tensor.

    Temporarily re-enables a disabled quantizer so actor refit can still fold raw
    FP weights even after rollout inference has turned weight quantizers off.

    Handles device mismatch: the TensorQuantizer may be on CPU (built from the
    pre-FSDP model) while the weight tensor is on GPU (from FSDP full_tensor()).
    """
    with torch.no_grad():
        weight_quantizer.to(weight.device)

        restore_disable = not weight_quantizer.is_enabled
        if restore_disable:
            weight_quantizer.enable()
        try:
            # Match modelopt's fold/export path: compute NVFP4 scales from fp32
            # weights, then cast the folded tensor back to the input dtype
            # (bf16 for training-side fold after live-dtype cast).
            folded = weight_quantizer(weight.float())
        finally:
            if restore_disable:
                weight_quantizer.disable()
        return folded.to(dtype=weight.dtype)


def fold_model_weights_in_place(model: nn.Module, weight_quantizer_map: dict[str, Any]) -> int:
    """Fold existing ``.weight`` parameters in place using ``weight_quantizer_map``."""
    if not weight_quantizer_map:
        return 0

    parameters = dict(model.named_parameters())
    folded = 0
    for name, wq in weight_quantizer_map.items():
        param = parameters.get(name)
        if param is None:
            continue
        param.data.copy_(fold_weight(param.data, wq))
        folded += 1
    return folded


def disable_weight_quantizers(model: nn.Module) -> int:
    """Disable all rollout ``weight_quantizer`` submodules (NeMo-RL vLLM rollout path)."""
    import modelopt.torch.quantization as mtq

    before = count_enabled_quantizers(model, name_suffix="weight_quantizer")
    if before == 0:
        return 0

    mtq.disable_quantizer(model, "*weight_quantizer")

    after = count_enabled_quantizers(model, name_suffix="weight_quantizer")
    if after > 0:
        _, TensorQuantizer = _require_modelopt_quant()
        for name, module in model.named_modules():
            if not name.endswith("weight_quantizer"):
                continue
            if isinstance(module, TensorQuantizer) and module.is_enabled:
                module.disable()
        after = count_enabled_quantizers(model, name_suffix="weight_quantizer")

    return before - after


def attach_weight_quantizer_map(model: nn.Module, weight_quantizer_map: dict[str, Any]) -> None:
    """Store the map on ``model`` (and the inner backbone when wrapped for PP)."""
    setattr(model, _VEXACT_WQ_MAP_ATTR, weight_quantizer_map)
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        setattr(inner, _VEXACT_WQ_MAP_ATTR, weight_quantizer_map)


def prepare_rollout_prefold(model: nn.Module) -> dict[str, Any]:
    """Fold current weights, disable rollout weight quantizers, cache the name→wq map."""
    weight_quantizer_map = build_weight_quantizer_map(model)
    enabled_before = count_enabled_quantizers(model, name_suffix="weight_quantizer")

    folded = fold_model_weights_in_place(model, weight_quantizer_map)
    disabled = disable_weight_quantizers(model)
    attach_weight_quantizer_map(model, weight_quantizer_map)
    stats = audit_rollout_quant_state(model, context="after prepare_rollout_prefold")
    logger.info(
        "[vexact-qat] Rollout prefold enabled: mapped %d weight(s), folded %d weight(s), "
        "disabled %d weight_quantizer(s) (enabled before fold=%d).",
        len(weight_quantizer_map),
        folded,
        disabled,
        enabled_before,
    )
    if stats["weight_quantizer_enabled"] > 0:
        raise RuntimeError(
            "[vexact-qat] Rollout prefold left "
            f"{stats['weight_quantizer_enabled']} enabled weight_quantizer(s). "
            "Inference would re-run live NVFP4 fake quant on every forward and can hang."
        )
    return weight_quantizer_map


def copy_weight_into_param(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_quantizer: Optional[Any] = None,
    *,
    prefold: bool = False,
) -> None:
    """Copy a synced or checkpoint weight into a parameter, optionally pre-folding."""
    target = loaded_weight
    if loaded_weight.device != param.device or loaded_weight.dtype != param.dtype:
        target = loaded_weight.to(device=param.device, dtype=param.dtype)
    if prefold and weight_quantizer is not None:
        target = fold_weight(target, weight_quantizer)
    param.data.copy_(target)


def load_weights_with_optional_prefold(
    model: nn.Module,
    weight_iterator: Iterable[tuple[str, torch.Tensor]],
    *,
    weight_quantizer_map: Optional[dict[str, Any]] = None,
    tied_weight_keys: Optional[list[str]] = None,
) -> None:
    """Default HF weight loader with optional NeMo-RL-style pre-folding on receive."""
    tied_weight_keys = tied_weight_keys or []
    parameters = dict(model.named_parameters())
    prefold = bool(weight_quantizer_map)
    embed_tokens_weight = None

    for full_name, loaded_weight in weight_iterator:
        if full_name == "model.embed_tokens.weight":
            embed_tokens_weight = loaded_weight

        if full_name not in parameters:
            continue

        wq = weight_quantizer_map.get(full_name) if weight_quantizer_map else None
        copy_weight_into_param(
            parameters[full_name],
            loaded_weight,
            wq,
            prefold=prefold and wq is not None,
        )

    for param_name in tied_weight_keys:
        if param_name not in parameters:
            continue
        if "model.embed_tokens.weight" in parameters:
            parameters[param_name].data = parameters["model.embed_tokens.weight"].data
        elif embed_tokens_weight is not None:
            copy_weight_into_param(parameters[param_name], embed_tokens_weight)


def maybe_prepare_rollout_prefold(model: nn.Module, qat_config: Optional[QATConfig]) -> None:
    """Fold+disable weight quantizers when QAT is enabled (legacy helper).

    Production w4a16 rollout skips QuantModule insertion entirely. Production
    w4a4 rollout calls ``disable_weight_quantizers`` after ``quantize_model``
    instead. This helper remains for tests and any path that still inserts
    full QuantModules then wants fold semantics.
    """
    if qat_config is None or not qat_config.enable:
        return
    prepare_rollout_prefold(model)
