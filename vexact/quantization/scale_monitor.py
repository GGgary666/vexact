# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or governing permissions and
# limitations under the License.

"""Monitor NVFP4 global_scale / amax after calibration and during QAT.

ModelOpt NVFP4 uses two-level scales. After max calibration the per-tensor
``amax`` (→ ``global_scale = amax / (6 * 448)``) is frozen via
``disable_calib()``; per-block scales remain dynamic. This module surfaces
that contract for logging and assertions (NeMo-RL ``get_quantizer_stats``
parity + global_scale summaries).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn


logger = logging.getLogger(__name__)

# NVFP4 E2M1 max * FP8 E4M3 max — ModelOpt / TRT-LLM global_scale denominator.
_NVFP4_GLOBAL_SCALE_DENOM = 6.0 * 448.0


def _require_tensor_quantizer():
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    return TensorQuantizer


def amax_to_global_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert per-tensor amax buffer to NVFP4 global_scale."""
    return amax.detach().float() / _NVFP4_GLOBAL_SCALE_DENOM


def get_quantizer_stats(model: nn.Module) -> dict[str, int]:
    """Return summary counts for TensorQuantizers (NeMo-RL parity)."""
    TensorQuantizer = _require_tensor_quantizer()
    total = 0
    enabled = 0
    with_amax = 0
    positive_amax = 0
    calib_enabled = 0
    for _, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        total += 1
        if not module.is_enabled:
            continue
        enabled += 1
        if getattr(module, "_if_calib", False):
            calib_enabled += 1
        amax = getattr(module, "amax", None)
        if amax is not None:
            with_amax += 1
            if bool((amax > 0).all().item()):
                positive_amax += 1
    return {
        "total": total,
        "enabled": enabled,
        "with_amax": with_amax,
        "positive_amax": positive_amax,
        "calib_enabled": calib_enabled,
    }


def collect_global_scale_stats(
    model: nn.Module,
    *,
    name_suffix: Optional[str] = None,
) -> dict[str, Any]:
    """Collect amax / global_scale stats for enabled quantizers.

    Args:
        model: quantized module tree.
        name_suffix: if set (e.g. ``input_quantizer`` / ``weight_quantizer``),
            only modules whose name ends with that suffix are included.
    """
    TensorQuantizer = _require_tensor_quantizer()
    amax_vals: list[float] = []
    scale_vals: list[float] = []
    samples: list[tuple[str, float, float]] = []

    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        if name_suffix is not None and not name.endswith(name_suffix):
            continue
        amax = getattr(module, "amax", None)
        if amax is None:
            continue
        # Per-tensor global amax may be a scalar or a reduced tensor.
        amax_f = float(amax.detach().float().amax().item())
        scale_f = amax_f / _NVFP4_GLOBAL_SCALE_DENOM
        amax_vals.append(amax_f)
        scale_vals.append(scale_f)
        if len(samples) < 8:
            samples.append((name, amax_f, scale_f))

    def _summary(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"count": 0.0}
        t = torch.tensor(vals, dtype=torch.float32)
        return {
            "count": float(t.numel()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "median": float(t.median().item()),
        }

    return {
        "amax": _summary(amax_vals),
        "global_scale": _summary(scale_vals),
        "samples": samples,
    }


def assert_calib_frozen(model: nn.Module) -> None:
    """Fail if any enabled non-dynamic quantizer still has calib collection on."""
    TensorQuantizer = _require_tensor_quantizer()
    still_calib: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        if getattr(module, "_dynamic", False):
            continue
        if getattr(module, "_if_calib", False):
            still_calib.append(name)
    if still_calib:
        raise RuntimeError(
            "[vexact-qat] Expected disable_calib() after max calibration, but "
            f"calib is still enabled on: {still_calib[:8]}"
            f"{'...' if len(still_calib) > 8 else ''}"
        )


def log_scale_monitor(
    model: nn.Module,
    *,
    prefix: str = "[vexact-qat]",
    step: Optional[int] = None,
) -> dict[str, Any]:
    """Log quantizer counts + input/weight global_scale summaries."""
    stats = get_quantizer_stats(model)
    iq = collect_global_scale_stats(model, name_suffix="input_quantizer")
    wq = collect_global_scale_stats(model, name_suffix="weight_quantizer")
    step_tag = f" step={step}" if step is not None else ""
    logger.info(
        "%s%s quantizer_stats total=%d enabled=%d with_amax=%d positive_amax=%d "
        "calib_enabled=%d",
        prefix,
        step_tag,
        stats["total"],
        stats["enabled"],
        stats["with_amax"],
        stats["positive_amax"],
        stats["calib_enabled"],
    )
    for label, block in (("input", iq), ("weight", wq)):
        amax = block["amax"]
        gs = block["global_scale"]
        if amax.get("count", 0) <= 0:
            logger.info("%s%s %s global_scale: (none)", prefix, step_tag, label)
            continue
        logger.info(
            "%s%s %s amax[min/mean/max]=%.6g/%.6g/%.6g "
            "global_scale[min/mean/max]=%.6g/%.6g/%.6g (n=%d)",
            prefix,
            step_tag,
            label,
            amax["min"],
            amax["mean"],
            amax["max"],
            gs["min"],
            gs["mean"],
            gs["max"],
            int(amax["count"]),
        )
        for name, amax_f, scale_f in block["samples"][:3]:
            logger.info(
                "%s%s   sample %s amax=%.6g global_scale=%.6g",
                prefix,
                step_tag,
                name,
                amax_f,
                scale_f,
            )
    return {"quantizer_stats": stats, "input": iq, "weight": wq}


def fingerprint_input_global_scales(model: nn.Module) -> dict[str, float]:
    """Map ``input_quantizer`` module name → scalar global_scale for drift checks."""
    TensorQuantizer = _require_tensor_quantizer()
    out: dict[str, float] = {}
    for name, module in model.named_modules():
        if not name.endswith("input_quantizer"):
            continue
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        amax = getattr(module, "amax", None)
        if amax is None:
            continue
        out[name] = float(amax.detach().float().amax().item()) / _NVFP4_GLOBAL_SCALE_DENOM
    return out
