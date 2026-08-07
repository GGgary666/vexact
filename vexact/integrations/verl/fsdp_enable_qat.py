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

"""Training-side QAT hook for the VeOmni FSDP actor/ref workers.

VeXact never owns the training loop -- it is only a rollout engine. The training
model is built by VeOmni (``veomni.models.build_foundation_model``) inside verl
workers, which VeXact can only touch via verl's
``actor_rollout_ref.model.external_lib`` hook (a module that is merely imported
by each worker process).

This module, when imported with ``VEXACT_QAT_ENABLE=1``, monkey-patches
VeOmni so every foundation model it builds gets modelopt fake quantizers
inserted. The QAT config is read from ``VEXACT_QAT_*`` environment variables
(see :meth:`QATConfig.from_env`) because the external_lib hook cannot see
verl's Hydra config; the launcher script exports the same values that the
rollout side receives via ``engine_kwargs.vexact.qat`` so both sides insert
identical quantizers.

Optional Cross-Precision Alignment (CPA): after a successful W4A4 QAT install,
set ``VEXACT_CPA_ENABLE=1`` (and optionally ``VEXACT_CPA_COEF``, default
``0.001``; ``VEXACT_CPA_LOSS_TYPE`` in ``{low_var_kl,mse,abs_logprob}``,
default ``low_var_kl``) to also patch ``VeOmniEngineWithLMHead.forward_step`` with a BF16
teacher alignment loss. No extra ``external_lib`` entry is required -- keep
mounting this module (or ``fsdp_enable_qat`` alone). CPA refuses to start unless
QAT is enabled with ``mode=w4a4``.

Usage (typically alongside the batch-invariant hook):

    actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant,vexact.integrations.verl.fsdp_enable_qat

Meta-device / FSDP2 subtlety:
    VeOmni's FSDP2 path (``model_engine=veomni`` with ``init_device=meta``)
    builds ``build_foundation_model`` on the ``meta`` device and only
    materializes real weights *afterwards*, inside
    ``parallelize_model_fsdp2`` -> ``load_model_weights`` /
    ``rank0_load_and_broadcast_weights`` (after FSDP2's ``fully_shard`` has
    already run). If we quantized the model immediately after
    ``build_foundation_model`` returns (as a naive "before FSDP wrap" hook
    would), modelopt's weight quantizer would calibrate its ``amax`` buffer
    against meta (data-less) weights, leaving that buffer on the ``meta``
    device. VeOmni's ``post_process_after_weight_loading`` later tries to
    ``copy_()`` that buffer into the materialized model and crashes with
    ``NotImplementedError: Cannot copy out of meta tensor; no data!`` --
    because the on-disk checkpoint has no ``*_amax`` key to overwrite it with
    real data first.

    To avoid this we only quantize immediately when ``build_foundation_model``
    already returned a model with real (non-meta) weights. Otherwise we mark
    the model as "quantization pending" and additionally patch VeOmni's
    meta-init weight loaders (``load_model_weights`` /
    ``rank0_load_and_broadcast_weights``) so quantization runs right after
    they materialize real weights -- mirroring exactly what the rollout side
    already does in ``vexact.inferencer.model_loader.ModelCreator`` (quantize
    *after* ``load_weights_from_weight_path``).

Notes:
    - ``w4a4`` must calibrate at init and cache ``input_quantizer`` amax for
      refit sync (train/rollout 0 mismatch). ``w4a16`` does not require
      calibration; 0 mismatch comes from training-side weight fold.
      Quantization must still happen after real weights are materialized
      (see meta-device subtlety above).
    - Any worker process that imports this module gets the QAT hook. verl's
      default schema has no ``actor_rollout_ref.ref.model.external_lib``; ref
      deep-copies ``actor_rollout_ref.model`` (including ``external_lib``).
      When ``skip_ref_quantization`` is enabled (default for ``w4a16``), QAT is
      skipped only when the build call stack indicates a reference-policy init.
      VeOmni hybrid workers often build the actor exactly once (ref shares weights
      or is initialized elsewhere), so a naive "skip the first build" would leave
      the actor unquantized.
    - Quantization failures raise :class:`RuntimeError` rather than silently
      falling back to an unquantized model, which would desync train/rollout.
"""

import functools
import inspect
import logging
import os
import sys
from typing import Callable

from vexact.quantization import QATConfig, quantize_model


logger = logging.getLogger(__name__)

# Stack markers that indicate a reference-policy model build (not the actor).
_REF_BUILD_FUNC_MARKERS = (
    "init_ref",
    "build_ref",
    "ref_policy",
    "ref_module",
    "init_reference",
    "init_reference_policy",
    "setup_ref",
    "_init_ref",
)


# By default, pull in the batch-invariant hook so mounting just ``fsdp_enable_qat``
# enables both (verl's ``import_external_libs`` does not split comma-separated
# strings). Set ``VEXACT_BATCH_INVARIANT=0`` to skip — ablation that keeps QAT
# but disables training-side batch-invariant ATen replacements. Guarded so a
# missing dependency does not abort QAT setup.
_batch_invariant_env = os.environ.get("VEXACT_BATCH_INVARIANT", "1").strip().lower()
if _batch_invariant_env in ("0", "false", "no", "off"):
    logger.info(
        "[vexact-qat] VEXACT_BATCH_INVARIANT=%s — skipping fsdp_enable_invariant.",
        os.environ.get("VEXACT_BATCH_INVARIANT"),
    )
else:
    try:
        from vexact.integrations.verl import fsdp_enable_invariant as _fsdp_enable_invariant  # noqa: F401
    except Exception:  # pragma: no cover - depends on training env
        logger.warning(
            "[vexact-qat] Could not import fsdp_enable_invariant; batch-invariant mode "
            "may be disabled. Mount it explicitly if needed.",
            exc_info=True,
        )


def _model_is_meta(model) -> bool:
    """Return True if ``model``'s parameters still live on the meta device."""
    try:
        return next(model.parameters()).device.type == "meta"
    except StopIteration:  # no parameters (unlikely for a foundation model)
        return False


def _detect_ref_policy_build_context() -> bool:
    """Return True when the current call stack looks like a ref-policy model init."""
    for frame_info in inspect.stack()[2:35]:
        func = frame_info.function.lower()
        if any(marker in func for marker in _REF_BUILD_FUNC_MARKERS):
            return True
        try:
            frame = frame_info.frame
            for key in ("role", "worker_role", "policy_role"):
                val = frame.f_locals.get(key)
                if val is not None:
                    val_str = str(val).lower()
                    if "ref" in val_str and "rollout" not in val_str:
                        return True
            self_obj = frame.f_locals.get("self")
            if self_obj is not None:
                cls_name = type(self_obj).__name__.lower()
                if (
                    "ref" in cls_name
                    and "rollout" not in cls_name
                    and "refresh" not in cls_name
                ):
                    return True
        except Exception:
            continue
    return False


def _should_skip_ref_qat(qat_config: QATConfig) -> bool:
    """Return True when this build should stay full precision (reference policy)."""
    if not qat_config.effective_skip_ref_quantization:
        return False
    if not _detect_ref_policy_build_context():
        return False
    logger.info(
        "[vexact-qat] Skipping QAT for reference policy build "
        "(detected from call stack; set VEXACT_QAT_SKIP_REF=0 to quantize ref too)."
    )
    return True


def _maybe_apply_qat(model, qat_config: QATConfig, context: str):
    if _should_skip_ref_qat(qat_config):
        return
    _apply_qat(model, qat_config, context=context)


def _apply_qat(model, qat_config: QATConfig, context: str):
    logger.info(
        f"[vexact-qat] Applying QAT fake quantization to VeOmni training "
        f"model ({context}; mode={qat_config.mode}, cfg={qat_config.resolved_cfg_name()})."
    )
    try:
        quantize_model(model, qat_config)
    except Exception as exc:
        logger.exception(
            f"[vexact-qat] Failed to quantize VeOmni training model ({context})."
        )
        raise RuntimeError(
            f"[vexact-qat] Failed to quantize VeOmni training model ({context}). "
            "Aborting to avoid a train/rollout quantization mismatch."
        ) from exc

    # Build and cache the training-side weight_quantizer_map so that
    # ServerAdapter.update_weights() can fold weights before sending them to
    # rollout. Rollout never runs live weight_quantizer (w4a16: plain Linear;
    # w4a4: input_quantizer only). For w4a4 also cache input_quantizer map and
    # assert amax was materialized by calibration.
    from vexact.quantization.amax_sync import (
        assert_input_amax_materialized,
        build_input_quantizer_map,
        set_training_input_quantizer_map,
    )
    from vexact.quantization.fold import (
        build_weight_quantizer_map,
        set_training_weight_quantizer_map,
    )

    wq_map = build_weight_quantizer_map(model)
    if wq_map:
        set_training_weight_quantizer_map(wq_map)
        logger.info(
            "[vexact-qat] Cached training-side weight_quantizer_map "
            "(%d quantized weight(s)) for training-side fold.",
            len(wq_map),
        )
    else:
        logger.warning(
            "[vexact-qat] Training model was quantized but no weight_quantizer "
            "map was built. Training-side fold will be disabled."
        )

    iq_map = build_input_quantizer_map(model)
    if iq_map:
        try:
            n_amax = assert_input_amax_materialized(model)
        except RuntimeError as exc:
            raise RuntimeError(
                f"[vexact-qat] w4a4 calibration did not materialize input amax "
                f"({context}). Aborting to avoid train/rollout mismatch."
            ) from exc
        set_training_input_quantizer_map(iq_map)
        logger.info(
            "[vexact-qat] Cached training-side input_quantizer_map "
            "(%d quantizer(s), %d amax buffer(s)) for refit sync.",
            len(iq_map),
            n_amax,
        )
        try:
            from vexact.quantization.scale_monitor import (
                fingerprint_input_global_scales,
                log_scale_monitor,
            )

            log_scale_monitor(model, prefix=f"[vexact-qat] train-ready ({context})")
            # Stash fingerprint for later drift checks on update_weights.
            model._vexact_input_global_scale_fp = fingerprint_input_global_scales(model)
        except Exception:  # pragma: no cover - monitoring must not break QAT
            logger.debug(
                "[vexact-qat] scale monitor after train quantize failed",
                exc_info=True,
            )
    else:
        set_training_input_quantizer_map(None)
        if qat_config.mode == "w4a4":
            logger.warning(
                "[vexact-qat] mode=w4a4 but no enabled input_quantizer found after "
                "quantize; activation amax sync will be skipped."
            )


def _wrap_build_foundation_model(orig_fn, qat_config: QATConfig):
    """Wrap VeOmni's build_foundation_model to quantize its output.

    If the returned model is still on the meta device (FSDP2's meta-init
    path), quantization is deferred to ``_wrap_weight_loader`` below, which
    runs once real weights are materialized.
    """

    if getattr(orig_fn, "_vexact_qat_wrapped", False):
        return orig_fn

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        model = orig_fn(*args, **kwargs)
        if _model_is_meta(model):
            if _should_skip_ref_qat(qat_config):
                return model
            model._vexact_qat_pending = True
            logger.info(
                "[vexact-qat] Model built on the meta device (FSDP2 meta-init); "
                "deferring QAT quantization until real weights are loaded."
            )
        else:
            _maybe_apply_qat(model, qat_config, context="right after construction")
        return model

    wrapped._vexact_qat_wrapped = True
    wrapped._vexact_qat_orig = orig_fn
    return wrapped


def _wrap_weight_loader(orig_fn, qat_config: QATConfig, fn_name: str):
    """Wrap a VeOmni meta-init weight loader (``load_model_weights`` /
    ``rank0_load_and_broadcast_weights``) to quantize the model right after
    it materializes real weights, if quantization was deferred by
    ``_wrap_build_foundation_model``.
    """

    if getattr(orig_fn, "_vexact_qat_wrapped", False):
        return orig_fn

    @functools.wraps(orig_fn)
    def wrapped(model, *args, **kwargs):
        result = orig_fn(model, *args, **kwargs)
        if getattr(model, "_vexact_qat_pending", False):
            model._vexact_qat_pending = False
            _maybe_apply_qat(model, qat_config, context=f"after {fn_name} materialized real weights")
        return result

    wrapped._vexact_qat_wrapped = True
    wrapped._vexact_qat_orig = orig_fn
    return wrapped


def _patch_everywhere(attr_name: str, orig_fn: Callable, wrapped: Callable) -> int:
    """Patch ``attr_name`` in VeOmni (and VeXact) namespaces that still bind ``orig_fn``.

    Only scan ``veomni.*`` / ``vexact.*`` modules. A full ``sys.modules`` scan also
    hits HuggingFace ``transformers`` lazy aliases (e.g. image-processing modules
    that expose ``build_foundation_model``), which slows every worker import by
    minutes and is unrelated to training-side QAT.
    """
    patched = 0
    for mod_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if not (mod_name.startswith("veomni") or mod_name.startswith("vexact")):
            continue
        # Some modules (e.g. torch._classes) fabricate arbitrary attributes via
        # __getattr__, so guard every attribute access defensively.
        try:
            attr = getattr(module, attr_name, None)
            if attr is None:
                continue
            # Match either the original function or an earlier wrapper of it.
            is_match = attr is orig_fn or getattr(attr, "_vexact_qat_orig", None) is orig_fn
        except Exception:
            continue
        if is_match:
            try:
                setattr(module, attr_name, wrapped)
            except Exception:
                continue
            patched += 1
            logger.info(f"[vexact-qat] Patched {attr_name} in module '{mod_name}'.")

    return patched


def _patch_all_build_foundation_model(qat_config: QATConfig) -> int:
    """Patch build_foundation_model at its definition, re-export, and callers."""
    try:
        import veomni.models.auto as veomni_auto
    except ImportError:
        logger.error(
            "[vexact-qat] VEXACT_QAT_ENABLE is set but 'veomni' is not importable. "
            "The training-side QAT hook cannot patch build_foundation_model. "
            "Ensure the training worker runs in the veomni environment."
        )
        return 0

    orig_fn = getattr(veomni_auto, "build_foundation_model", None)
    if orig_fn is None:
        logger.error("[vexact-qat] veomni.models.auto has no build_foundation_model to patch.")
        return 0

    # Unwrap if a previous import already wrapped it (idempotent re-import).
    orig_fn = getattr(orig_fn, "_vexact_qat_orig", orig_fn)
    wrapped = _wrap_build_foundation_model(orig_fn, qat_config)
    return _patch_everywhere("build_foundation_model", orig_fn, wrapped)


def _patch_all_weight_loaders(qat_config: QATConfig) -> int:
    """Patch VeOmni's meta-init weight loaders so deferred QAT quantization
    (see module docstring) runs once real weights are materialized.
    """
    try:
        import veomni.models.module_utils as veomni_module_utils
    except ImportError:
        logger.error(
            "[vexact-qat] VEXACT_QAT_ENABLE is set but 'veomni' is not importable. "
            "The training-side QAT hook cannot patch VeOmni's weight loaders. "
            "Ensure the training worker runs in the veomni environment."
        )
        return 0

    patched = 0
    for fn_name in ("load_model_weights", "rank0_load_and_broadcast_weights"):
        orig_fn = getattr(veomni_module_utils, fn_name, None)
        if orig_fn is None:
            logger.warning(f"[vexact-qat] veomni.models.module_utils has no '{fn_name}' to patch.")
            continue

        # Unwrap if a previous import already wrapped it (idempotent re-import).
        orig_fn = getattr(orig_fn, "_vexact_qat_orig", orig_fn)
        wrapped = _wrap_weight_loader(orig_fn, qat_config, fn_name)
        patched += _patch_everywhere(fn_name, orig_fn, wrapped)

    return patched


def maybe_enable_cpa_after_qat(qat_installed: bool, qat_config: QATConfig) -> bool:
    """Install CPA after QAT when ``VEXACT_CPA_ENABLE=1``.

    Fail-fast rules:
      * CPA enable requires a successful QAT hook install.
      * CPA enable requires ``qat_config.enable`` and ``mode == "w4a4"``.
      * CPA is incompatible with QAOPD (``VEXACT_QAOPD_ENABLE=1``).
    """
    from vexact.integrations.verl import fsdp_enable_cpa
    from vexact.quantization.cpa import CPAConfig

    cpa_config = CPAConfig.from_env()
    if not cpa_config.enable:
        return False

    if os.environ.get("VEXACT_QAOPD_ENABLE", "").strip().lower() in ("1", "true", "yes", "on"):
        raise RuntimeError(
            "[vexact-cpa] VEXACT_CPA_ENABLE=1 conflicts with VEXACT_QAOPD_ENABLE=1. "
            "QAOPD uses VeRL on-policy distillation as the sole loss; disable CPA."
        )

    if not qat_installed or not qat_config.enable:
        raise RuntimeError(
            "[vexact-cpa] VEXACT_CPA_ENABLE=1 requires a successful W4A4 QAT hook "
            "(VEXACT_QAT_ENABLE=1). QAT was not installed."
        )
    if qat_config.mode != "w4a4":
        raise RuntimeError(
            f"[vexact-cpa] CPA currently supports only mode='w4a4' "
            f"(got mode={qat_config.mode!r})."
        )
    return fsdp_enable_cpa.enable_training_cpa(cpa_config)


def enable_training_qat() -> bool:
    """Enable the training-side QAT hook based on ``VEXACT_QAT_*`` env vars.

    Returns True if quantization was enabled and the patch was installed.
    When ``VEXACT_CPA_ENABLE=1``, also installs the CPA forward_step hook after
    a successful W4A4 QAT install (fail-fast otherwise).
    """
    qat_config = QATConfig.from_env()
    if not qat_config.enable:
        logger.info("[vexact-qat] VEXACT_QAT_ENABLE not set; training-side QAT disabled.")
        # CPA without QAT is a hard error (do not silently train without alignment).
        maybe_enable_cpa_after_qat(qat_installed=False, qat_config=qat_config)
        return False

    logger.info(f"[vexact-qat] Training-side QAT enabled: {qat_config}")
    patched = _patch_all_build_foundation_model(qat_config)
    patched += _patch_all_weight_loaders(qat_config)
    if patched == 0:
        logger.error(
            "[vexact-qat] Training-side QAT enabled but no VeOmni hooks were patched "
            "(build_foundation_model / load_model_weights / rank0_load_and_broadcast_weights)."
        )
        maybe_enable_cpa_after_qat(qat_installed=False, qat_config=qat_config)
        return False
    logger.info(f"[vexact-qat] Training-side QAT hook installed ({patched} namespace(s)).")
    maybe_enable_cpa_after_qat(qat_installed=True, qat_config=qat_config)
    return True


enable_training_qat()
print("[vexact-qat] fsdp_enable_qat imported.")
