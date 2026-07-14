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

"""Quantization-Aware Training (QAT) configuration for VeXact.

The design mirrors verl's ``actor.qat`` config surface (``enable`` / ``mode`` /
``ignore_patterns`` / ...) but the actual fake quantization is delegated to
NVIDIA Model-Optimizer (``modelopt``), following NeMo-RL's approach.

``mode`` is a short, framework-friendly alias that maps to a concrete modelopt
quantization config constant (see ``MODE_TO_MODELOPT_CFG``):

- ``w4a4``  -> ``NVFP4_DEFAULT_CFG`` as-is                        (NVFP4 E2M1 weight + activation)
- ``w4a16`` -> ``NVFP4_DEFAULT_CFG`` with ``*input_quantizer`` forced off (NVFP4 E2M1 weight-only)

modelopt does not ship a built-in "NVFP4 weight-only, all layers" constant
(only MoE/MLP-restricted variants like ``NVFP4_MLP_WEIGHT_ONLY_CFG`` exist), so
``w4a16`` is built from ``NVFP4_DEFAULT_CFG`` by merging an explicit
``*input_quantizer`` disable in :func:`vexact.quantization.quantize.resolve_quant_cfg`,
unless ``quant_cfg`` is set explicitly. This keeps ``w4a16`` on the same NVFP4 E2M1
numerics as ``w4a4`` (so folding/export stays uniform) while only quantizing weights.

Calibration / amax sync (train–rollout 0 mismatch):

- ``w4a4``: **MUST** calibrate at init so ``input_quantizer.amax`` is materialized,
  then **MUST** sync that amax to rollout on every ``update_weights``. Setting
  ``calibrate=False`` is illegal (dynamic activation scales diverge across sides).
- ``w4a16``: no activation quantizer; init calibration is **not** required. 0
  mismatch comes from training-side ``weight_quantizer`` fold into BF16.

``quant_cfg`` may be set to override the mode mapping with any modelopt config
constant name (e.g. ``"INT4_BLOCKWISE_WEIGHT_ONLY_CFG"``) or a modelopt PTQ
recipe / YAML path (resolved by :func:`vexact.quantization.quantize.resolve_quant_cfg`).
When set, the implicit ``w4a16`` activation-disable is skipped since the
override is trusted to already encode its own weight/activation split.

Rollout weights always use training-side fold (NeMo-RL QARL style): the actor
keeps live STE weight fake-quant, folds ``fp32→bf16`` before sync, and rollout
never runs live ``weight_quantizer``. For ``w4a16`` rollout stays plain Linear;
for ``w4a4`` rollout only keeps ``input_quantizer`` (activation) modules.
``prefold_weights`` is retained for config compatibility but is always treated
as enabled when QAT is on.

Offline deployment export (real NVFP4 HF for vLLM ``modelopt_fp4``) is separate
from online fold: use :mod:`vexact.quantization.export` / ``scripts/export_qat_to_hf.py``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)

# Alias -> modelopt config constant name. Keep in sync with the modelopt version
# pinned by the ``qat`` extra. Both modes share the same NVFP4 base config;
# ``w4a16`` additionally forces off ``*input_quantizer`` in resolve_quant_cfg()
# to turn it into NVFP4 weight-only (see module docstring for rationale).
MODE_TO_MODELOPT_CFG: dict[str, str] = {
    "w4a4": "NVFP4_DEFAULT_CFG",
    "w4a16": "NVFP4_DEFAULT_CFG",
}

VALID_MODES = tuple(MODE_TO_MODELOPT_CFG.keys())

# Extra module exclusions appended on top of modelopt's built-in defaults (which
# already skip lm_head / MoE routers / mamba conv, etc.). Patterns are matched by
# modelopt against quantizer module names using fnmatch wildcards, so a bare
# substring like ``lm_head`` should be wrapped as ``*lm_head*`` (done in
# :func:`resolve_quant_cfg`).
DEFAULT_IGNORE_PATTERNS: list[str] = ["lm_head", "embed_tokens"]


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_flag_explicit(name: str) -> Optional[bool]:
    """Return True/False if ``name`` is set, else ``None``."""
    val = os.environ.get(name)
    if val is None:
        return None
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class QATConfig:
    """Fake-quantization (QAT) configuration shared by the training and rollout sides.

    Both sides construct the *same* ``QATConfig`` so the inserted modelopt
    quantizers are identical, which is what keeps the training and inference
    numerics aligned.

    For ``w4a4``, calibration is mandatory and ``input_quantizer.amax`` must be
    synced to rollout. For ``w4a16``, calibration is optional (default off).
    """

    enable: bool = False
    mode: str = "w4a4"
    # Optional explicit modelopt config name / recipe / YAML path. When set it
    # overrides the ``mode`` -> config mapping.
    quant_cfg: Optional[str] = None
    ignore_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))
    # ``None`` resolves to mode default: True for w4a4, False for w4a16.
    # w4a4 + False raises (train/rollout activation mismatch).
    calibrate: Optional[bool] = None
    # Optional calibration knobs, only used when calibrate=True and a real
    # forward loop is not otherwise supplied.
    calib_size: int = 512
    calib_seq_len: int = 512
    # Retained for config/Hydra compatibility. When QAT is enabled, rollout
    # weights always come from training-side fold; live weight quantizers on
    # rollout are not supported. ``effective_prefold_weights`` is always True.
    prefold_weights: Optional[bool] = None
    # When True, skip QAT only when ``build_foundation_model`` is invoked from a
    # reference-policy init (detected via call stack). verl hybrid workers share
    # ``actor_rollout_ref.model.external_lib`` (no separate ``ref.model.external_lib``).
    # Default True for ``w4a16`` so ref stays full precision while actor is quantized.
    skip_ref_quantization: Optional[bool] = None

    def __post_init__(self):
        if self.mode not in VALID_MODES and self.quant_cfg is None:
            raise ValueError(
                f"QATConfig.mode must be one of {VALID_MODES} (got '{self.mode}'), "
                f"or set quant_cfg explicitly."
            )
        if self.ignore_patterns is None:
            self.ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
        if self.calibrate is None:
            # w4a4 must calibrate for activation amax; w4a16 does not need it.
            self.calibrate = self.mode == "w4a4"
        if self.enable and self.mode == "w4a4" and not self.calibrate:
            raise ValueError(
                "QATConfig: mode='w4a4' requires calibrate=True for train/rollout "
                "0 mismatch (static input_quantizer.amax + refit sync). "
                "calibrate=False is not allowed for w4a4."
            )
        if self.prefold_weights is False:
            # Live rollout weight quantizers were removed; fold is mandatory.
            logger.warning(
                "[vexact-qat] prefold_weights=False is ignored; training-side "
                "weight fold is mandatory (rollout never runs live weight_quantizer)."
            )

    @property
    def effective_calibrate(self) -> bool:
        """Whether calibration runs at quantize time (always True for enabled w4a4)."""
        return bool(self.calibrate)

    @property
    def effective_prefold_weights(self) -> bool:
        """Training-side weight fold is always on when QAT is enabled."""
        return True

    @property
    def effective_skip_ref_quantization(self) -> bool:
        """Whether reference-policy builds should skip QAT (actor is always quantized)."""
        if self.skip_ref_quantization is not None:
            return self.skip_ref_quantization
        return self.mode == "w4a16"

    def resolved_cfg_name(self) -> str:
        """Return the modelopt config name/recipe to use.

        ``quant_cfg`` takes precedence over the ``mode`` alias.
        """
        if self.quant_cfg:
            return self.quant_cfg
        return MODE_TO_MODELOPT_CFG[self.mode]

    @classmethod
    def from_env(cls, prefix: str = "VEXACT_QAT_") -> "QATConfig":
        """Build a QATConfig from ``VEXACT_QAT_*`` environment variables.

        Used by the training-side external_lib hook, which is only *imported*
        by verl/VeOmni workers and therefore cannot see the Hydra config. The
        launcher script exports these vars so the training and rollout sides
        agree on the same quantization config.

        Recognized variables (all optional except ENABLE):
            {prefix}ENABLE            -> enable            (bool)
            {prefix}MODE              -> mode              (str)
            {prefix}CFG               -> quant_cfg         (str)
            {prefix}IGNORE_PATTERNS   -> ignore_patterns   (comma-separated)
            {prefix}CALIBRATE         -> calibrate         (bool)
            {prefix}CALIB_SIZE        -> calib_size        (int)
            {prefix}CALIB_SEQ_LEN     -> calib_seq_len     (int)
            {prefix}PREFOLD_WEIGHTS   -> prefold_weights   (bool)
            {prefix}SKIP_REF        -> skip_ref_quantization (bool)
        """
        kwargs: dict = {"enable": _env_flag(f"{prefix}ENABLE", False)}

        mode = os.environ.get(f"{prefix}MODE")
        if mode:
            kwargs["mode"] = mode.strip()

        quant_cfg = os.environ.get(f"{prefix}CFG")
        if quant_cfg and quant_cfg.strip():
            kwargs["quant_cfg"] = quant_cfg.strip()

        ignore = os.environ.get(f"{prefix}IGNORE_PATTERNS")
        if ignore is not None and ignore.strip():
            kwargs["ignore_patterns"] = [p.strip() for p in ignore.split(",") if p.strip()]

        calibrate = _env_flag_explicit(f"{prefix}CALIBRATE")
        if calibrate is not None:
            kwargs["calibrate"] = calibrate

        calib_size = os.environ.get(f"{prefix}CALIB_SIZE")
        if calib_size and calib_size.strip():
            kwargs["calib_size"] = int(calib_size)

        calib_seq_len = os.environ.get(f"{prefix}CALIB_SEQ_LEN")
        if calib_seq_len and calib_seq_len.strip():
            kwargs["calib_seq_len"] = int(calib_seq_len)

        if os.environ.get(f"{prefix}PREFOLD_WEIGHTS") is not None:
            kwargs["prefold_weights"] = _env_flag(f"{prefix}PREFOLD_WEIGHTS", False)

        if os.environ.get(f"{prefix}SKIP_REF") is not None:
            kwargs["skip_ref_quantization"] = _env_flag(f"{prefix}SKIP_REF", False)

        return cls(**kwargs)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["QATConfig"]:
        """Build a QATConfig from a plain dict (e.g. engine_kwargs.vexact.qat).

        Returns ``None`` if ``data`` is empty/None or ``enable`` is falsy, so
        callers can uniformly treat "no quantization" as ``None``.
        """
        if not data:
            return None
        # Tolerate OmegaConf DictConfig by shallow-copying known keys only.
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        kwargs = {k: v for k, v in dict(data).items() if k in known}
        cfg = cls(**kwargs)
        if not cfg.enable:
            return None
        return cfg
