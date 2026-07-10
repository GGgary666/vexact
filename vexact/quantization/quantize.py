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

"""modelopt-backed fake quantization used by both the training and rollout sides.

``modelopt`` is imported lazily inside the functions here so that importing
``vexact.quantization`` never fails in environments without the ``qat`` extra
installed. Only enabling QAT pulls modelopt in.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Optional

import torch
from torch import nn

from .config import QATConfig


logger = logging.getLogger(__name__)


ForwardLoop = Callable[[nn.Module], Any]

# Exact modelopt dict-key / list-entry pattern for per-layer input quantizers.
_W4A16_INPUT_QUANTIZER_PATTERN = "*input_quantizer"


def _require_modelopt():
    try:
        import modelopt.torch.quantization as mtq  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "QAT requires NVIDIA Model-Optimizer. Install the optional extra, "
            "e.g. `uv pip install -e '.[qat]'` (adds nvidia-modelopt)."
        ) from e
    return mtq


def resolve_quant_cfg(qat_config: QATConfig) -> dict[str, Any]:
    """Resolve a :class:`QATConfig` into a dict consumable by ``mtq.quantize``.

    Resolution order for the base config (mirrors NeMo-RL's resolver):

    1. Built-in modelopt config constant exposed on
       ``modelopt.torch.quantization`` (e.g. ``NVFP4_DEFAULT_CFG``).
    2. A modelopt PTQ recipe name or a path to a YAML recipe, resolved via
       ``modelopt.recipe.load_config``.

    ``ignore_patterns`` from the QATConfig are then appended as
    ``{"quantizer_name": "*<pattern>*", "enable": False}`` entries so the listed
    modules keep full precision. Entries are applied in order, later ones win,
    so these disables sit on top of the base config.

    modelopt has no built-in "NVFP4 weight-only, all layers" constant, so when
    ``mode == "w4a16"`` and the caller did not override ``quant_cfg``, an
    explicit ``*input_quantizer`` disable is merged on top of the
    ``NVFP4_DEFAULT_CFG`` base (same NVFP4 E2M1 weight quantizer as ``w4a4``,
    just with activations left unquantized). The disable uses the exact
    modelopt wildcard ``*input_quantizer`` (not a broader ``*input_quantizer*``
    substring) so it reliably overrides the dict-format ``NVFP4_DEFAULT_CFG``
    entry. This mirrors how NeMo-RL derives its weight-only NVFP4 recipe and
    keeps ``w4a16``'s numerics identical to ``w4a4`` on the weight side.
    """
    if qat_config.mode == "w4a16" and qat_config.quant_cfg:
        logger.warning(
            "[vexact-qat] mode=w4a16 with explicit quant_cfg=%r: the implicit "
            "*input_quantizer disable is skipped. Ensure the recipe is "
            "weight-only or train/rollout numerics may diverge.",
            qat_config.quant_cfg,
        )

    mtq = _require_modelopt()
    cfg_name = qat_config.resolved_cfg_name()

    base = getattr(mtq, cfg_name, None)
    if base is None:
        base = _load_recipe_cfg(cfg_name)

    # Deep-copy so we never mutate the shared modelopt module-level constant.
    resolved = copy.deepcopy(dict(base))

    disables = _ignore_pattern_entries(qat_config.ignore_patterns)
    if qat_config.mode == "w4a16" and not qat_config.quant_cfg:
        disables.append(_w4a16_input_quantizer_disable_entry())
    _merge_quant_cfg_disables(resolved, disables)

    return resolved


def _w4a16_input_quantizer_disable_entry() -> dict[str, Any]:
    return {"quantizer_name": _W4A16_INPUT_QUANTIZER_PATTERN, "enable": False}


def _merge_quant_cfg_disables(resolved: dict[str, Any], disables: list[dict[str, Any]]) -> None:
    """Append disable entries on top of a resolved modelopt config."""
    if not disables:
        return

    quant_cfg = resolved.get("quant_cfg")
    if isinstance(quant_cfg, list):
        resolved["quant_cfg"] = list(quant_cfg) + disables
    elif isinstance(quant_cfg, dict):
        # Dict format: later keys override earlier ones for the same wildcard.
        merged = dict(quant_cfg)
        for entry in disables:
            merged[entry["quantizer_name"]] = {"enable": False}
        resolved["quant_cfg"] = merged
    else:
        resolved["quant_cfg"] = disables


def _load_recipe_cfg(cfg_name: str) -> dict[str, Any]:
    try:
        from modelopt.recipe import load_config
    except ImportError as e:  # pragma: no cover
        raise ValueError(
            f"Unknown quant_cfg '{cfg_name}': not a built-in modelopt config "
            f"constant and modelopt.recipe is unavailable."
        ) from e

    try:
        loaded = load_config(cfg_name)
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(
            f"Unknown quant_cfg '{cfg_name}'. Must be a built-in modelopt config "
            f"name (e.g. 'NVFP4_DEFAULT_CFG'), a modelopt PTQ recipe name, or a "
            f"path to a YAML quantization recipe."
        ) from e

    quantize = loaded.get("quantize", loaded)
    if not isinstance(quantize, dict) or "quant_cfg" not in quantize:
        raise ValueError(
            f"Quantization recipe '{cfg_name}' must contain a 'quant_cfg' entry "
            f"(optionally nested under a top-level 'quantize:' section)."
        )
    return quantize


def _ignore_pattern_entries(patterns: Optional[list[str]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pattern in patterns or []:
        pattern = pattern.strip()
        if not pattern:
            continue
        # Support verl-style "re:..." by stripping the prefix; modelopt matches
        # with fnmatch wildcards, so we approximate a regex by a broad wildcard.
        if pattern.startswith("re:"):
            pattern = pattern[3:]
            logger.warning(
                "QAT ignore pattern '%s' uses regex syntax; modelopt uses fnmatch "
                "wildcards. Applying it as a wildcard match, which may differ. "
                "Prefer plain substrings.",
                pattern,
            )
        # Wrap bare substrings so they match anywhere in the quantizer module name.
        wildcard = pattern if ("*" in pattern or "?" in pattern) else f"*{pattern}*"
        entries.append({"quantizer_name": wildcard, "enable": False})
    return entries


def is_model_quantized(model: nn.Module) -> bool:
    """Return True if the model already has modelopt quantizers inserted."""
    try:
        from modelopt.torch.quantization.utils import is_quantized

        return bool(is_quantized(model))
    except ImportError:
        return False


def _needs_calibration(mtq_cfg: dict[str, Any]) -> bool:
    try:
        from modelopt.torch.quantization.config import need_calibration

        return bool(need_calibration(mtq_cfg))
    except Exception:  # pragma: no cover - be conservative
        return False


def _random_calibration_forward_loop(qat_config: QATConfig) -> ForwardLoop:
    """A minimal random-token calibration loop (matches NeMo-RL's 'random' path).

    This only populates quantizer statistics so mtq does not warn about unset
    amax. Prefer supplying a real forward loop over relying on this.
    """

    def forward_loop(model: nn.Module) -> None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover
            device = torch.device("cpu")
        seq_len = max(1, min(qat_config.calib_seq_len, 8))
        input_ids = torch.randint(0, 100, (1, seq_len), device=device)
        with torch.no_grad():
            model(input_ids=input_ids)

    return forward_loop


def quantize_model(
    model: nn.Module,
    qat_config: QATConfig,
    forward_loop: Optional[ForwardLoop] = None,
) -> nn.Module:
    """Insert modelopt fake quantizers into ``model`` in place.

    Args:
        model: the (pre-FSDP-wrap / pre-eval) model to quantize.
        qat_config: resolved QAT settings shared across train/infer sides.
        forward_loop: optional calibration callable ``fn(model) -> None``. When
            ``qat_config.calibrate`` is True and this is None, a lightweight
            random-token calibration loop is used as a fallback.

    Returns:
        The same model object, now quantized. Idempotent: if the model is
        already quantized this is a no-op.

    Notes:
        For NVFP4 (``w4a4``/``w4a16``) with ``calibrate=False`` we intentionally
        pass ``forward_loop=None`` so calibration is skipped and no static
        ``_amax`` buffer is created. modelopt then computes the (enabled)
        quantizer amax dynamically per forward, which keeps the training and
        rollout numerics consistent without any amax synchronization. For
        ``w4a16`` the activation (``input_quantizer``) side is simply disabled
        via :func:`resolve_quant_cfg`, so only the weight quantizer runs.
    """
    if not qat_config.enable:
        return model

    mtq = _require_modelopt()

    if is_model_quantized(model):
        logger.info("[vexact-qat] Model already quantized; skipping quantize_model().")
        return model

    mtq_cfg = resolve_quant_cfg(qat_config)
    cfg_name = qat_config.resolved_cfg_name()

    if qat_config.mode == "w4a16" and qat_config.calibrate:
        logger.warning(
            "[vexact-qat] calibrate=True with mode=w4a16 can materialize static "
            "quantizer state that is not synchronized to rollout. Prefer "
            "calibrate=False for weight-only NVFP4 QAT."
        )

    if qat_config.calibrate:
        if forward_loop is None:
            logger.warning(
                "[vexact-qat] calibrate=True but no forward_loop provided; using a "
                "random-token calibration fallback. Static amax from mismatched "
                "calibration data can break train/infer alignment."
            )
            forward_loop = _random_calibration_forward_loop(qat_config)
    else:
        if _needs_calibration(mtq_cfg):
            logger.info(
                "[vexact-qat] Skipping calibration for '%s' (calibrate=False): "
                "activation amax will be computed dynamically per forward, keeping "
                "train/infer aligned without amax sync.",
                cfg_name,
            )
        forward_loop = None

    logger.info(
        "[vexact-qat] Quantizing model: mode=%s cfg=%s ignore=%s calibrate=%s",
        qat_config.mode,
        cfg_name,
        qat_config.ignore_patterns,
        qat_config.calibrate,
    )

    model = mtq.quantize(model, mtq_cfg, forward_loop)

    try:
        mtq.print_quant_summary(model)
    except Exception:  # pragma: no cover - summary is best-effort
        logger.debug("[vexact-qat] print_quant_summary failed", exc_info=True)

    return model
