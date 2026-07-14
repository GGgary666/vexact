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

"""Export QAT QuantModule models to real NVFP4 HuggingFace checkpoints.

Online QAT uses fake-quant + BF16 fold for VeXact rollout. This module packs
weights into real NVFP4 tensors via ModelOpt ``export_hf_checkpoint`` for
deployment with vLLM ``--quantization modelopt_fp4``.

Example::

    python -m vllm.entrypoints.openai.api_server \\
        --model <export_dir> --quantization modelopt_fp4
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from torch import nn

from .amax_sync import (
    assert_input_amax_materialized,
    build_input_quantizer_map,
    is_input_amax_key,
    load_input_amax_exact,
)
from .config import QATConfig
from .quantize import quantize_model


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
ForwardLoop = Callable[[nn.Module], Any]
InputAmaxMapping = dict[str, torch.Tensor]

# VeOmni FSDP2 mesh names -> names accepted by verl.model_merger.FSDPModelMerger.
_VEOMNI_MESH_DIM_ALIAS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("dp_shard",): ("fsdp",),
    ("ddp", "dp_shard"): ("ddp", "fsdp"),
    ("dp", "dp_shard"): ("ddp", "fsdp"),
    ("dp_replicate", "dp_shard"): ("ddp", "fsdp"),
}


def normalize_mesh_dim_names(mesh_dim_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Map VeOmni mesh dim names onto verl FSDP merger's expected names."""
    names = tuple(mesh_dim_names)
    return _VEOMNI_MESH_DIM_ALIAS.get(names, names)


# ModelOpt QuantModule buffers that are not HuggingFace weight keys.
_MODELOPT_EXTRA_KEY_MARKERS: tuple[str, ...] = (
    "weight_quantizer",
    "input_quantizer",
    "output_quantizer",
    "._amax",
    ".amax",
)


def is_modelopt_extra_state_key(name: str) -> bool:
    """Return True for ModelOpt quantizer buffers (amax/scales), not HF weights."""
    return any(marker in name for marker in _MODELOPT_EXTRA_KEY_MARKERS)


def filter_hf_weight_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop ModelOpt quantizer extras; keep tensors loadable by a plain HF model."""
    kept: dict[str, Any] = {}
    dropped = 0
    for key, value in state_dict.items():
        if is_modelopt_extra_state_key(key):
            dropped += 1
            continue
        kept[key] = value
    if dropped:
        logger.info(
            "[vexact-qat] Dropped %d ModelOpt quantizer state key(s) before HF save "
            "(export re-applies QAT / packs NVFP4 separately).",
            dropped,
        )
    return kept


def merge_plain_tensor_shards(shards: list[torch.Tensor]) -> torch.Tensor:
    """Merge non-DTensor shards: scalars/replicated -> rank0; else cat on dim 0.

    Stock verl FSDP merger always ``torch.cat(..., dim=0)``, which crashes on
    0-dim QAT ``*_amax`` buffers and other replicated plain tensors.
    """
    if not shards:
        raise ValueError("merge_plain_tensor_shards received an empty shard list")
    first = shards[0]
    if first.ndim == 0:
        return first
    if all(t.shape == first.shape for t in shards):
        return first
    return torch.cat(shards, dim=0)


def extract_input_amax_from_state_dict(state_dict: dict[str, Any]) -> InputAmaxMapping:
    """Collect ``input_quantizer`` amax tensors from a (possibly sharded) state_dict."""
    out: InputAmaxMapping = {}
    for key, value in state_dict.items():
        if not is_input_amax_key(key):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        out[key] = value.detach().cpu()
    return out


def merge_input_amax_shards(
    per_rank_dicts: list[InputAmaxMapping],
) -> InputAmaxMapping:
    """Merge per-rank input amax dicts.

    Same-shape / scalar amax is treated as replicated: start from rank0, then
    ``torch.maximum`` across ranks when values differ (TP-safe, matches training
    refit). Differently shaped shards fall back to :func:`merge_plain_tensor_shards`.
    """
    if not per_rank_dicts:
        return {}

    keys: set[str] = set()
    for shard in per_rank_dicts:
        keys.update(shard.keys())

    merged: InputAmaxMapping = {}
    for key in sorted(keys):
        shards = [d[key] for d in per_rank_dicts if key in d]
        if not shards:
            continue
        first = shards[0]
        if first.ndim == 0 or all(t.shape == first.shape for t in shards):
            out = first.detach().clone()
            for other in shards[1:]:
                other_cpu = other.detach().to(dtype=out.dtype, device=out.device)
                if not torch.equal(out, other_cpu):
                    out = torch.maximum(out, other_cpu)
            merged[key] = out
        else:
            merged[key] = merge_plain_tensor_shards(
                [t.detach() for t in shards]
            ).cpu()
    return merged


def prepare_model_for_nvfp4_export(
    model: nn.Module,
    qat_config: QATConfig,
    *,
    forward_loop: Optional[ForwardLoop] = None,
    input_amax: Optional[InputAmaxMapping | list[tuple[str, torch.Tensor]]] = None,
) -> nn.Module:
    """Ensure ``model`` has QuantModules and (for w4a4) materialized input amax.

    Idempotent if the model is already quantized. For ``w4a4``, raises if
    enabled ``input_quantizer.amax`` is still missing after calibration.

    When ``input_amax`` is provided (from actor ckpt), QuantModules are still
    inserted via :func:`quantize_model` (short/random calib only materializes
    buffers); amax values are then overwritten exactly from the ckpt.
    """
    if not qat_config.enable:
        raise ValueError("prepare_model_for_nvfp4_export requires qat_config.enable=True")

    quantize_model(model, qat_config, forward_loop=forward_loop)

    if input_amax is not None:
        if isinstance(input_amax, dict):
            amax_items: list[tuple[str, torch.Tensor]] = list(input_amax.items())
        else:
            amax_items = list(input_amax)
        if not amax_items:
            raise RuntimeError(
                "prepare_model_for_nvfp4_export: input_amax was provided but empty"
            )
        restored = load_input_amax_exact(model, amax_items, strict=True)
        iq_map = build_input_quantizer_map(model)
        if restored < len(iq_map):
            raise RuntimeError(
                f"prepare_model_for_nvfp4_export: restored {restored} amax buffer(s) "
                f"but model has {len(iq_map)} enabled input_quantizer(s)."
            )
        logger.info(
            "[vexact-qat] Restored training input_quantizer.amax for export "
            "(%d buffer(s)).",
            restored,
        )

    if qat_config.mode == "w4a4":
        iq_map = build_input_quantizer_map(model)
        if not iq_map:
            raise RuntimeError(
                "prepare_model_for_nvfp4_export: mode=w4a4 but no enabled "
                "input_quantizer found; cannot export input_scale."
            )
        assert_input_amax_materialized(model)

    return model


# vLLM ``modelopt_fp4`` selects kernels from hf_quant_config.json quant_algo:
#   NVFP4       -> W4A4 (needs *.input_scale; quantizes activations)
#   W4A16_NVFP4 -> weight-only (no input_scale; bf16/fp16 activations)
# ModelOpt ``export_hf_checkpoint`` always emits quant_algo=NVFP4 for E2M1
# weights, even when input_quantizer is disabled — so w4a16 must rewrite.
_QUANT_ALGO_W4A4 = "NVFP4"
_QUANT_ALGO_W4A16 = "W4A16_NVFP4"


def export_nvfp4_hf_checkpoint(
    model: nn.Module,
    export_dir: PathLike,
    *,
    tokenizer: Any = None,
    dtype: Optional[torch.dtype] = None,
    require_input_scale: Optional[bool] = None,
    mode: Optional[str] = None,
) -> Path:
    """Pack a QuantModule model into a real NVFP4 HF directory.

    Args:
        model: already-quantized model (see :func:`prepare_model_for_nvfp4_export`).
        export_dir: output directory.
        tokenizer: optional tokenizer with ``save_pretrained``.
        dtype: optional dtype forwarded to ModelOpt export.
        require_input_scale: if True, require at least one ``*.input_scale`` in
            the exported state (w4a4). If False, assert none (w4a16). If None,
            infer from whether any enabled input_quantizer exists on ``model``.
        mode: ``w4a4`` / ``w4a16``. When ``w4a16``, rewrite ModelOpt's
            ``quant_algo`` from ``NVFP4`` to ``W4A16_NVFP4`` so vLLM uses the
            weight-only Marlin path instead of activation-quantizing NVFP4.

    Returns:
        Resolved ``export_dir`` path.
    """
    from modelopt.torch.export import export_hf_checkpoint

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    if require_input_scale is None:
        require_input_scale = bool(build_input_quantizer_map(model))
    if mode is None:
        mode = "w4a4" if require_input_scale else "w4a16"

    if require_input_scale:
        assert_input_amax_materialized(model)

    kwargs: dict[str, Any] = {"export_dir": str(export_path)}
    if dtype is not None:
        kwargs["dtype"] = dtype

    logger.info("[vexact-qat] Exporting NVFP4 HF checkpoint to %s (mode=%s)", export_path, mode)
    with torch.inference_mode():
        export_hf_checkpoint(model, **kwargs)

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(export_path))

    rewrite_nvfp4_quant_algo_for_mode(export_path, mode)
    validate_nvfp4_export_dir(
        export_path,
        require_input_scale=require_input_scale,
        mode=mode,
    )
    return export_path.resolve()


def rewrite_nvfp4_quant_algo_for_mode(export_dir: PathLike, mode: str) -> str:
    """Rewrite ModelOpt ``quant_algo`` so vLLM picks W4A4 vs W4A16 kernels.

    ModelOpt always writes ``NVFP4`` for E2M1 packed weights. For ``w4a16``
    (input_quantizer disabled, no ``input_scale``), vLLM requires
    ``W4A16_NVFP4``; leaving ``NVFP4`` makes it treat the ckpt as W4A4 and
    either fail or silently corrupt activations.
    """
    if mode not in ("w4a4", "w4a16"):
        raise ValueError(f"rewrite_nvfp4_quant_algo_for_mode: unknown mode {mode!r}")

    export_path = Path(export_dir)
    target = _QUANT_ALGO_W4A4 if mode == "w4a4" else _QUANT_ALGO_W4A16
    quant_cfg_path = export_path / "hf_quant_config.json"
    if not quant_cfg_path.is_file():
        raise FileNotFoundError(f"Missing hf_quant_config.json under {export_path}")

    with open(quant_cfg_path) as f:
        hf_quant_config = json.load(f)

    quant_section = hf_quant_config.get("quantization")
    if isinstance(quant_section, dict):
        old = quant_section.get("quant_algo")
        quant_section["quant_algo"] = target
        # Per-layer entries (MIXED_PRECISION / older dumps) may still say NVFP4.
        quantized_layers = quant_section.get("quantized_layers")
        if isinstance(quantized_layers, dict):
            for layer_cfg in quantized_layers.values():
                if isinstance(layer_cfg, dict) and layer_cfg.get("quant_algo") in (
                    _QUANT_ALGO_W4A4,
                    "nvfp4",
                ):
                    if mode == "w4a16":
                        layer_cfg["quant_algo"] = target
    else:
        old = hf_quant_config.get("quant_algo")
        hf_quant_config["quant_algo"] = target

    with open(quant_cfg_path, "w") as f:
        json.dump(hf_quant_config, f, indent=4)
        f.write("\n")

    config_path = export_path / "config.json"
    if config_path.is_file():
        with open(config_path) as f:
            config_data = json.load(f)
        qcfg = config_data.get("quantization_config")
        if isinstance(qcfg, dict):
            qcfg["quant_algo"] = target
            # Weight-only: drop fake activation scheme left by ModelOpt convert.
            if mode == "w4a16":
                groups = qcfg.get("config_groups")
                if isinstance(groups, dict):
                    for group in groups.values():
                        if isinstance(group, dict):
                            group.pop("input_activations", None)
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
                f.write("\n")

    if old != target:
        logger.info(
            "[vexact-qat] Rewrote quant_algo %r -> %r for mode=%s (vLLM modelopt_fp4).",
            old,
            target,
            mode,
        )
    return target


def validate_nvfp4_export_dir(
    export_dir: PathLike,
    *,
    require_input_scale: bool = False,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    """Validate exported HF dir has ModelOpt NVFP4 metadata (and optional input_scale)."""
    export_path = Path(export_dir)
    quant_cfg_path = export_path / "hf_quant_config.json"
    config_path = export_path / "config.json"

    if not quant_cfg_path.is_file():
        raise FileNotFoundError(
            f"Missing hf_quant_config.json under {export_path}. "
            "export_hf_checkpoint may have failed or model had no QuantModules."
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json under {export_path}")

    if mode is None:
        mode = "w4a4" if require_input_scale else "w4a16"
    expected_algo = _QUANT_ALGO_W4A4 if mode == "w4a4" else _QUANT_ALGO_W4A16

    with open(quant_cfg_path) as f:
        hf_quant_config = json.load(f)

    quant_algo = (
        hf_quant_config.get("quantization", {}).get("quant_algo")
        if isinstance(hf_quant_config.get("quantization"), dict)
        else hf_quant_config.get("quant_algo")
    )
    if quant_algo != expected_algo:
        raise RuntimeError(
            f"Expected quant_algo={expected_algo!r} in hf_quant_config.json for "
            f"mode={mode}, got {quant_algo!r}. For w4a16, ModelOpt emits 'NVFP4' "
            f"and must be rewritten to 'W4A16_NVFP4' for vLLM weight-only."
        )

    safetensors = list(export_path.glob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(f"No *.safetensors under {export_path}")

    input_scale_keys = _collect_safetensor_keys_ending(safetensors, "input_scale")
    if require_input_scale and not input_scale_keys:
        raise RuntimeError(
            f"w4a4 export expected *.input_scale tensors under {export_path}, found none. "
            "Ensure input_quantizer.amax was calibrated before export."
        )
    if not require_input_scale and input_scale_keys:
        raise RuntimeError(
            f"w4a16 / weight-only export must not contain *.input_scale "
            f"(found {len(input_scale_keys)}, first={input_scale_keys[0]}). "
            "Input quantizers were not disabled before export_hf_checkpoint."
        )

    summary = {
        "export_dir": str(export_path.resolve()),
        "quant_algo": quant_algo,
        "mode": mode,
        "safetensors": len(safetensors),
        "input_scale_keys": len(input_scale_keys),
    }
    logger.info("[vexact-qat] Validated NVFP4 export: %s", summary)
    return summary


def _collect_safetensor_keys_ending(paths: list[Path], suffix: str) -> list[str]:
    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover
        logger.warning("[vexact-qat] safetensors not installed; skipping key scan")
        return []

    keys: list[str] = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(suffix) or key.endswith(f".{suffix}"):
                    keys.append(key)
    return keys
