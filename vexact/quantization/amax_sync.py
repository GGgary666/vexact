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

"""Sync ``input_quantizer.amax`` from training actor to rollout (NeMo-RL style).

Used for ``w4a4`` train/rollout 0 mismatch: after init calibration, each
``update_weights`` refit must carry enabled ``input_quantizer`` amax buffers.
``w4a16`` has input quantizers disabled, so the map stays empty.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import torch
from torch import nn


logger = logging.getLogger(__name__)

_training_input_quantizer_map: Optional[dict[str, Any]] = None

# Buffer / state_dict key fragment used to detect amax payloads in weight sync.
_INPUT_QUANTIZER_TOKEN = "input_quantizer"
_AMAX_TOKEN = "amax"


def set_training_input_quantizer_map(iq_map: Optional[dict[str, Any]]) -> None:
    """Cache the training model's enabled input_quantizer map for amax refit."""
    global _training_input_quantizer_map
    _training_input_quantizer_map = iq_map


def get_training_input_quantizer_map() -> Optional[dict[str, Any]]:
    """Return the cached training-side input_quantizer map, or ``None``."""
    return _training_input_quantizer_map


def _require_tensor_quantizer():
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    return TensorQuantizer


def build_input_quantizer_map(model: nn.Module) -> dict[str, Any]:
    """Map module names ending in ``input_quantizer`` to enabled TensorQuantizers."""
    TensorQuantizer = _require_tensor_quantizer()
    mapping: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not name.endswith("input_quantizer"):
            continue
        if not isinstance(module, TensorQuantizer):
            continue
        if not module.is_enabled:
            continue
        mapping[name] = module
    return mapping


def input_amax_buffer_name(quantizer_module_name: str) -> str:
    """HF-style buffer name for a TensorQuantizer ``_amax`` under ``named_buffers``."""
    if quantizer_module_name.endswith("._amax"):
        return quantizer_module_name
    return f"{quantizer_module_name}._amax"


def is_input_amax_key(name: str) -> bool:
    """Return True if ``name`` looks like an input_quantizer amax buffer key."""
    return _INPUT_QUANTIZER_TOKEN in name and _AMAX_TOKEN in name.rsplit(".", 1)[-1]


def iter_input_amax_buffers(iq_map: dict[str, Any]) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(buffer_name, amax)`` for each enabled input quantizer.

    Raises:
        RuntimeError: if any enabled quantizer has ``amax is None`` (calibration
            did not materialize static amax — illegal for w4a4 0-mismatch path).
    """
    missing: list[str] = []
    for name, iq in iq_map.items():
        amax = getattr(iq, "amax", None)
        if amax is None:
            missing.append(name)
            continue
        yield input_amax_buffer_name(name), amax.detach().clone()
    if missing:
        raise RuntimeError(
            "[vexact-qat] Enabled input_quantizer(s) missing amax after calibration: "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''}. "
            "w4a4 requires calibrate=True so amax can be synced to rollout."
        )


def chain_weights_with_input_amax(
    weights: Iterable[tuple[str, torch.Tensor]],
    iq_map: dict[str, Any],
    stats: Optional[dict[str, int]] = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield folded/raw weights then all input_quantizer amax buffers.

    If ``stats`` is provided, populate ``attached`` and skip the legacy INFO log
    (caller is expected to emit a step-level summary).
    """
    yield from weights
    amax_count = 0
    for item in iter_input_amax_buffers(iq_map):
        amax_count += 1
        yield item
    if stats is not None:
        stats["attached"] = amax_count
        return
    logger.info(
        "[vexact-qat] Training-side amax sync: attached %d input_quantizer amax buffer(s).",
        amax_count,
    )


def assert_input_amax_materialized(model: nn.Module) -> int:
    """Ensure every enabled input_quantizer has a non-None amax; return count."""
    iq_map = build_input_quantizer_map(model)
    if not iq_map:
        return 0
    # Force validation via iterator consumption.
    return sum(1 for _ in iter_input_amax_buffers(iq_map))


def fill_input_amax_sentinel(model: nn.Module, value: float = -1.0) -> int:
    """Fill enabled input_quantizer amax with a sentinel (NeMo-RL rollout init).

    Rollout must not trust locally calibrated amax; actor refit overwrites these.
    """
    TensorQuantizer = _require_tensor_quantizer()
    filled = 0
    for name, module in model.named_modules():
        if not name.endswith("input_quantizer"):
            continue
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        amax = getattr(module, "amax", None)
        if amax is None:
            continue
        amax.fill_(value)
        filled += 1
    return filled


def _resolve_input_amax_buffer(
    buffers: dict[str, torch.Tensor], name: str
) -> Optional[torch.Tensor]:
    """Look up an amax buffer, tolerating optional ``module.`` prefix."""
    buf = buffers.get(name)
    if buf is not None:
        return buf
    if name.startswith("module."):
        return buffers.get(name[len("module.") :])
    return buffers.get(f"module.{name}")


def apply_input_amax_buffers(
    model: nn.Module,
    amax_items: Iterable[tuple[str, torch.Tensor]],
) -> int:
    """Load input_quantizer amax buffers with ``torch.max`` merge (TP-safe).

    Returns the number of buffers updated.
    """
    buffers = dict(model.named_buffers())
    updated = 0
    missing: list[str] = []
    for name, loaded in amax_items:
        if not is_input_amax_key(name):
            continue
        buf = _resolve_input_amax_buffer(buffers, name)
        if buf is None:
            missing.append(name)
            continue
        src = loaded.to(device=buf.device, dtype=buf.dtype)
        if src.shape != buf.shape:
            raise RuntimeError(
                f"[vexact-qat] input amax shape mismatch for '{name}': "
                f"buffer {tuple(buf.shape)} vs loaded {tuple(src.shape)}"
            )
        buf.copy_(torch.maximum(buf, src))
        updated += 1
    if missing:
        logger.warning(
            "[vexact-qat] %d input amax key(s) not found on rollout model "
            "(first=%s).",
            len(missing),
            missing[0],
        )
    # Per-bucket apply is noisy under bucketed weight transfer; step-level
    # summary is emitted by ServerAdapter.update_weights (RANK 0).
    if updated:
        logger.debug(
            "[vexact-qat] Applied %d input_quantizer amax buffer(s) (max-merge).",
            updated,
        )
    return updated


def load_input_amax_exact(
    model: nn.Module,
    amax_items: Iterable[tuple[str, torch.Tensor]],
    *,
    strict: bool = True,
) -> int:
    """Overwrite input_quantizer amax buffers with exact ``copy_`` (export path).

    Unlike :func:`apply_input_amax_buffers` (max-merge for online TP refit), this
    restores the training-final amax values for NVFP4 HF export.

    Args:
        model: already-quantized model with materialized amax buffers.
        amax_items: ``(buffer_name, amax_tensor)`` pairs from actor ckpt.
        strict: if True, require every enabled ``input_quantizer`` amax to be
            restored from ``amax_items`` (extra unmatched ckpt keys only warn).

    Returns:
        Number of buffers overwritten.
    """
    buffers = dict(model.named_buffers())
    updated = 0
    missing_on_model: list[str] = []
    applied_names: set[str] = set()

    for name, loaded in amax_items:
        if not is_input_amax_key(name):
            continue
        buf = _resolve_input_amax_buffer(buffers, name)
        if buf is None:
            missing_on_model.append(name)
            continue
        src = loaded.detach().to(device=buf.device, dtype=buf.dtype)
        if src.shape != buf.shape:
            raise RuntimeError(
                f"[vexact-qat] input amax shape mismatch for '{name}': "
                f"buffer {tuple(buf.shape)} vs loaded {tuple(src.shape)}"
            )
        buf.copy_(src)
        applied_names.add(name)
        if name.startswith("module."):
            applied_names.add(name[len("module.") :])
        else:
            applied_names.add(f"module.{name}")
        updated += 1

    if missing_on_model:
        logger.warning(
            "[vexact-qat] %d input amax key(s) from ckpt not found on model "
            "(first=%s).",
            len(missing_on_model),
            missing_on_model[0],
        )

    if strict:
        iq_map = build_input_quantizer_map(model)
        expected = {input_amax_buffer_name(n) for n in iq_map}
        unresolved = [
            exp
            for exp in expected
            if exp not in applied_names
            and not (exp.startswith("module.") and exp[len("module.") :] in applied_names)
            and f"module.{exp}" not in applied_names
        ]
        if unresolved:
            raise RuntimeError(
                "[vexact-qat] Export restore incomplete: enabled input_quantizer "
                f"amax missing from ckpt: {unresolved[:8]}"
                f"{'...' if len(unresolved) > 8 else ''}."
            )
        if not expected:
            raise RuntimeError(
                "[vexact-qat] load_input_amax_exact(strict=True) found no enabled "
                "input_quantizer on model."
            )

    logger.info(
        "[vexact-qat] Restored %d input_quantizer amax buffer(s) exactly from ckpt.",
        updated,
    )
    return updated


def split_weights_and_input_amax(
    items: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """Partition a weight iterator into parameter tensors vs input amax buffers."""
    weights: list[tuple[str, torch.Tensor]] = []
    amaxes: list[tuple[str, torch.Tensor]] = []
    for name, tensor in items:
        if is_input_amax_key(name):
            amaxes.append((name, tensor))
        else:
            weights.append((name, tensor))
    return weights, amaxes
