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

For NVFP4 (``w4a4`` / ``w4a16``), :func:`quantize_model` installs VeXact's
vLLM-aligned Triton fake-quant kernel in place of ModelOpt's
``fp4_fake_quant_block`` (see :mod:`vexact.quantization.nvfp4.modelopt_patch`)
so STE numerics match B200 ``scaled_fp4_quant``. Disable with
``QATConfig.use_vllm_nvfp4_kernel=False`` or ``VEXACT_QAT_VLLM_NVFP4_KERNEL=0``.
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


def _maybe_install_vllm_nvfp4_kernel(qat_config: QATConfig) -> None:
    """Swap ModelOpt NVFP4 Triton fake-quant for VeXact's vLLM-aligned kernel."""
    if not qat_config.use_vllm_nvfp4_kernel:
        return
    # Only NVFP4 modes (w4a4 / w4a16) use fp4_fake_quant_block.
    cfg_name = qat_config.resolved_cfg_name()
    if "NVFP4" not in cfg_name.upper() and qat_config.mode not in ("w4a4", "w4a16"):
        return
    try:
        from vexact.quantization.nvfp4 import install_vllm_aligned_nvfp4_kernel

        ok = install_vllm_aligned_nvfp4_kernel()
        if not ok:
            logger.warning(
                "[vexact-qat] use_vllm_nvfp4_kernel=True but patch install failed; "
                "falling back to stock ModelOpt NVFP4 numerics."
            )
    except Exception:  # pragma: no cover - best-effort
        logger.warning(
            "[vexact-qat] Failed to install vLLM-aligned NVFP4 kernel; "
            "using stock ModelOpt numerics.",
            exc_info=True,
        )


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


def _force_eager_attention(model: nn.Module) -> Optional[dict[str, Any]]:
    """Temporarily switch HF attention to eager so smoke calib needs no KV cache.

    VeXact rollout registers paged flash/flex attention backends that require
    ``set_kv_cache_context()``. Random-token QAT calibration runs before the
    inferencer installs that context, so we force ``eager`` for the duration of
    the smoke forward. Real amax for w4a4 rollout still comes from actor refit.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None
    prev = {
        "_attn_implementation": getattr(config, "_attn_implementation", None),
    }
    try:
        config._attn_implementation = "eager"
    except Exception:  # pragma: no cover
        return None
    return prev


def _restore_attention(model: nn.Module, prev: Optional[dict[str, Any]]) -> None:
    if not prev:
        return
    config = getattr(model, "config", None)
    if config is None:
        return
    impl = prev.get("_attn_implementation")
    if impl is not None:
        try:
            config._attn_implementation = impl
        except Exception:  # pragma: no cover
            pass


def _random_calibration_forward_loop(qat_config: QATConfig) -> ForwardLoop:
    """A minimal random-token calibration loop (matches NeMo-RL's 'random' path).

    Forces eager attention during the forward so VeXact paged-attn backends
    (which need KV cache context) are not invoked. This only materializes
    quantizer ``amax`` buffers; prefer a real forward loop for production
    training-side calibration. Rollout w4a4 still overwrites IQ amax from actor.
    """

    def forward_loop(model: nn.Module) -> None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover
            device = torch.device("cpu")
        seq_len = max(1, min(qat_config.calib_seq_len, 8))
        input_ids = torch.randint(0, 100, (1, seq_len), device=device)
        prev_attn = _force_eager_attention(model)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            _restore_attention(model, prev_attn)

    return forward_loop


_JSONL_TEXT_KEYS = (
    "text",
    "article",
    "content",
    "document",
    "prompt",
    "question",
    "input",
)


def _extract_jsonl_text(record: dict[str, Any]) -> Optional[str]:
    for key in _JSONL_TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Nested CNN/DailyMail style: {"article": "..."} already covered; also
    # accept a single-string payload under uncommon keys.
    for value in record.values():
        if isinstance(value, str) and len(value.strip()) >= 8:
            return value.strip()
    return None


def _load_jsonl_texts(path: str, max_samples: int) -> list[str]:
    import json
    from pathlib import Path

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"[vexact-qat] Calibration data file not found: {path}"
        )
    texts: list[str] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"[vexact-qat] Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"[vexact-qat] Expected a JSON object on line {line_no} of {path}, "
                    f"got {type(record).__name__}."
                )
            text = _extract_jsonl_text(record)
            if text is None:
                continue
            texts.append(text)
            if len(texts) >= max_samples:
                break
    if not texts:
        raise ValueError(
            f"[vexact-qat] No usable text fields found in {path}. "
            f"Expected one of {_JSONL_TEXT_KEYS}."
        )
    return texts


def _load_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "[vexact-qat] Dataset calibration requires transformers. "
            "Install the veomni/verl extras."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _try_modelopt_dataset_dataloader(qat_config: QATConfig, device: torch.device):
    """Best-effort ModelOpt dataset path (cnn_dailymail, etc.). Returns None on miss."""
    if not qat_config.calib_tokenizer:
        return None
    try:
        from modelopt.torch.utils.dataset_utils import (
            create_forward_loop,
            get_dataset_dataloader,
        )
    except ImportError:
        return None

    try:
        tokenizer = _load_tokenizer(qat_config.calib_tokenizer)
        dataloader = get_dataset_dataloader(
            dataset_name=qat_config.calib_data,
            tokenizer=tokenizer,
            batch_size=qat_config.calib_batch_size,
            num_samples=qat_config.calib_size,
            device=device,
            include_labels=False,
            max_sample_length=qat_config.calib_seq_len,
        )
        return create_forward_loop(dataloader=dataloader)
    except Exception as exc:
        logger.info(
            "[vexact-qat] ModelOpt dataset loader unavailable for %r (%s); "
            "falling back to JSONL/local path handling.",
            qat_config.calib_data,
            exc,
        )
        return None


def _jsonl_calibration_forward_loop(qat_config: QATConfig) -> ForwardLoop:
    """Build a calibration loop from a local JSONL text file."""
    if not qat_config.calib_tokenizer:
        raise ValueError(
            "[vexact-qat] calib_tokenizer is required when calib_data points to a "
            "JSONL file or custom dataset path."
        )
    texts = _load_jsonl_texts(qat_config.calib_data, qat_config.calib_size)
    tokenizer = _load_tokenizer(qat_config.calib_tokenizer)
    batch_size = max(1, int(qat_config.calib_batch_size))
    max_len = max(1, int(qat_config.calib_seq_len))

    def forward_loop(model: nn.Module) -> None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover
            device = torch.device("cpu")
        prev_attn = _force_eager_attention(model)
        try:
            with torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    batch_texts = texts[start : start + batch_size]
                    encoded = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded.get("attention_mask")
                    kwargs = {"input_ids": input_ids, "use_cache": False}
                    if attention_mask is not None:
                        kwargs["attention_mask"] = attention_mask.to(device)
                    model(**kwargs)
        finally:
            _restore_attention(model, prev_attn)

    return forward_loop


def build_calibration_forward_loop(qat_config: QATConfig) -> ForwardLoop:
    """Select random / JSONL / ModelOpt dataset calibration forward loop.

    Resolution order when ``calibrate=True`` and no caller-provided loop:

    1. ``calib_data`` empty / ``random`` -> random-token fallback
    2. Existing local file path -> JSONL text calibration
    3. ModelOpt named dataset (requires ``calib_tokenizer``)
    4. Otherwise fail-fast with a clear error
    """
    if not qat_config.uses_dataset_calibration:
        return _random_calibration_forward_loop(qat_config)

    from pathlib import Path

    data = qat_config.calib_data
    assert data is not None

    if Path(data).is_file():
        logger.info(
            "[vexact-qat] Using JSONL calibration data=%s size=%d seq_len=%d batch=%d",
            data,
            qat_config.calib_size,
            qat_config.calib_seq_len,
            qat_config.calib_batch_size,
        )
        return _jsonl_calibration_forward_loop(qat_config)

    # Named ModelOpt dataset (e.g. cnn_dailymail) — needs a device later; wrap.
    def _deferred(model: nn.Module) -> None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover
            device = torch.device("cpu")
        modelopt_loop = _try_modelopt_dataset_dataloader(qat_config, device)
        if modelopt_loop is None:
            raise FileNotFoundError(
                f"[vexact-qat] Calibration data {data!r} is neither an existing "
                f"file nor a loadable ModelOpt dataset. Provide a JSONL path via "
                f"VEXACT_QAT_CALIB_DATA / QATConfig.calib_data, or set "
                f"calib_data=random."
            )
        prev_attn = _force_eager_attention(model)
        try:
            with torch.no_grad():
                modelopt_loop(model)
        finally:
            _restore_attention(model, prev_attn)

    logger.info(
        "[vexact-qat] Using ModelOpt/named calibration dataset=%s size=%d",
        data,
        qat_config.calib_size,
    )
    return _deferred


def resolve_calibration_forward_loop(
    qat_config: QATConfig,
    forward_loop: Optional[ForwardLoop] = None,
) -> Optional[ForwardLoop]:
    """Return the forward loop to pass to ``mtq.quantize``, or None if disabled."""
    if not qat_config.effective_calibrate:
        return None
    if forward_loop is not None:
        return forward_loop
    if qat_config.uses_dataset_calibration:
        return build_calibration_forward_loop(qat_config)

    # Explicit ``calib_data=random`` or allow_random_calib opt-in (tests / smoke).
    explicit_random = (
        qat_config.calib_data is not None
        and str(qat_config.calib_data).strip().lower() == "random"
    )
    if explicit_random or qat_config.allow_random_calib:
        logger.warning(
            "[vexact-qat] Using random-token calibration fallback "
            "(calib_data=%r allow_random_calib=%s). Prefer CNN/DailyMail JSONL "
            "or ModelOpt named dataset (e.g. cnn_dailymail) for production w4a4.",
            qat_config.calib_data,
            qat_config.allow_random_calib,
        )
        return _random_calibration_forward_loop(qat_config)

    raise ValueError(
        "[vexact-qat] w4a4 calibration requires a real calib dataset "
        "(NeMo-RL parity). Set VEXACT_QAT_CALIB_DATA to a JSONL path "
        "(e.g. cnn_dailymail_calib.jsonl with {\"text\": ...}) or a ModelOpt "
        "named dataset (cnn_dailymail), plus VEXACT_QAT_CALIB_TOKENIZER. "
        "For tests/smoke only: VEXACT_QAT_CALIB_DATA=random or "
        "VEXACT_QAT_ALLOW_RANDOM_CALIB=1."
    )


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
            ``qat_config.effective_calibrate`` is True and this is None, a
            dataset loop (if ``calib_data`` is set) or random-token fallback
            is used.

    Returns:
        The same model object, now quantized. Idempotent: if the model is
        already quantized this is a no-op.

    Notes:
        ``w4a4`` always calibrates so ``input_quantizer.amax`` is materialized
        for train/rollout sync. ``w4a16`` typically skips calibration
        (``calibrate=False``); only the weight quantizer runs (activations
        disabled via :func:`resolve_quant_cfg`), and 0 mismatch comes from
        training-side weight fold.
    """
    if not qat_config.enable:
        return model

    mtq = _require_modelopt()

    if is_model_quantized(model):
        logger.info("[vexact-qat] Model already quantized; skipping quantize_model().")
        # Still (re)install the kernel patch so subsequent STE forwards use
        # vLLM-aligned numerics even if quantizers were inserted earlier.
        _maybe_install_vllm_nvfp4_kernel(qat_config)
        return model

    _maybe_install_vllm_nvfp4_kernel(qat_config)

    mtq_cfg = resolve_quant_cfg(qat_config)
    cfg_name = qat_config.resolved_cfg_name()
    do_calibrate = qat_config.effective_calibrate

    if qat_config.mode == "w4a4" and not do_calibrate:
        raise ValueError(
            "quantize_model: mode='w4a4' requires calibration "
            "(QATConfig.calibrate=True) for train/rollout 0 mismatch."
        )

    if do_calibrate:
        forward_loop = resolve_calibration_forward_loop(qat_config, forward_loop)
    else:
        if _needs_calibration(mtq_cfg):
            logger.info(
                "[vexact-qat] Skipping calibration for '%s' (mode=%s, "
                "calibrate=False): weight-only path; amax sync not required.",
                cfg_name,
                qat_config.mode,
            )
        forward_loop = None

    logger.info(
        "[vexact-qat] Quantizing model: mode=%s cfg=%s ignore=%s calibrate=%s "
        "calib_data=%s vllm_nvfp4_kernel=%s",
        qat_config.mode,
        cfg_name,
        qat_config.ignore_patterns,
        do_calibrate,
        qat_config.calib_data,
        qat_config.use_vllm_nvfp4_kernel,
    )

    model = mtq.quantize(model, mtq_cfg, forward_loop)

    try:
        mtq.print_quant_summary(model)
    except Exception:  # pragma: no cover - summary is best-effort
        logger.debug("[vexact-qat] print_quant_summary failed", exc_info=True)

    if do_calibrate:
        from vexact.quantization.scale_monitor import (
            assert_calib_frozen,
            get_quantizer_stats,
            log_scale_monitor,
        )

        # ModelOpt max calib ends with load_calib_amax + disable_calib so the
        # per-tensor amax (NVFP4 global_scale source) stays frozen.
        try:
            assert_calib_frozen(model)
        except RuntimeError:
            logger.warning(
                "[vexact-qat] calib_frozen check failed after mtq.quantize; "
                "global_scale may still update if calib stays enabled.",
                exc_info=True,
            )
        stats = get_quantizer_stats(model)
        if stats["enabled"] > 0 and stats["positive_amax"] < stats["with_amax"]:
            logger.warning(
                "[vexact-qat] Some enabled quantizers have non-positive amax "
                "(with_amax=%d positive_amax=%d).",
                stats["with_amax"],
                stats["positive_amax"],
            )
        log_scale_monitor(model, prefix="[vexact-qat] post-calib")

    return model
