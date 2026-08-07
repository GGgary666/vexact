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

"""Replace ModelOpt NVFP4 Triton fake-quant with vLLM-aligned VeXact kernel.

ModelOpt QAT still owns:
  - ``TensorQuantizer`` insertion / amax buffers / STE autograd
  - config resolution (``NVFP4_DEFAULT_CFG``)

Only the forward numerics of ``fp4_fake_quant_block`` are swapped so train/rollout
fake-quant matches vLLM B200 ``scaled_fp4_quant`` bit-exactly.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .constants import BLOCK_SIZE
from .reference import nvfp4_fake_quant_eager, nvfp4_global_scale
from .triton_kernel import nvfp4_fake_quant_triton_fused, triton_is_available

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL: Any = None


def _normalize_amax(global_amax: torch.Tensor) -> torch.Tensor:
    amax = global_amax
    if amax.numel() != 1:
        amax = amax.amax()
    return amax.to(dtype=torch.float32)


def fp4_fake_quant_block_vllm(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    tile_rows: int = 16,
    tile_cols: int = 64,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Drop-in replacement for ModelOpt ``fp4_fake_quant_block``.

    Signature matches ModelOpt so existing call sites / STE wrappers keep working.
    Extra tile kwargs are accepted for API compatibility but ignored (VeXact uses
    its B200-tuned defaults).
    """
    del tile_rows, tile_cols, num_warps, num_stages

    if block_size != BLOCK_SIZE:
        if _ORIGINAL is not None:
            return _ORIGINAL(x, global_amax, block_size=block_size)
        raise ValueError(f"vLLM-aligned NVFP4 only supports block_size={BLOCK_SIZE}")

    if x.shape[-1] % BLOCK_SIZE != 0:
        # Unusual shapes: keep ModelOpt behaviour rather than failing QAT.
        if _ORIGINAL is not None:
            logger.debug(
                "[vexact-nvfp4] last dim %d not divisible by %d; "
                "falling back to ModelOpt kernel",
                x.shape[-1],
                BLOCK_SIZE,
            )
            return _ORIGINAL(x, global_amax, block_size=block_size)
        raise ValueError(
            f"last dim must be divisible by {BLOCK_SIZE} for NVFP4, got {x.shape[-1]}"
        )

    amax = _normalize_amax(global_amax)
    gs = nvfp4_global_scale(amax)

    # Call fused/eager directly (no @torch.inference_mode wrapper) so ModelOpt's
    # DynamicBlockQuantizationFunction STE path stays intact.
    if x.is_cuda and triton_is_available():
        return nvfp4_fake_quant_triton_fused(x, gs, block_size=block_size)
    return nvfp4_fake_quant_eager(x, global_amax=amax)


def install_vllm_aligned_nvfp4_kernel(*, force: bool = False) -> bool:
    """Monkeypatch ModelOpt ``fp4_fake_quant_block`` → VeXact vLLM-aligned kernel.

    Returns True if the patch is active after the call.
    """
    global _INSTALLED, _ORIGINAL

    if _INSTALLED and not force:
        return True

    try:
        import modelopt.torch.quantization.triton as mtq_triton
    except ImportError:
        logger.warning(
            "[vexact-nvfp4] modelopt triton package unavailable; "
            "cannot install vLLM-aligned NVFP4 kernel."
        )
        return False

    if not getattr(mtq_triton, "IS_AVAILABLE", False) and not hasattr(
        mtq_triton, "fp4_fake_quant_block"
    ):
        logger.warning(
            "[vexact-nvfp4] ModelOpt Triton kernels not available on this device; "
            "skipping vLLM-aligned NVFP4 patch."
        )
        return False

    original = getattr(mtq_triton, "fp4_fake_quant_block", None)
    if original is None:
        logger.warning(
            "[vexact-nvfp4] ModelOpt has no fp4_fake_quant_block; patch skipped."
        )
        return False

    if original is fp4_fake_quant_block_vllm and not force:
        _INSTALLED = True
        return True

    _ORIGINAL = original
    mtq_triton.fp4_fake_quant_block = fp4_fake_quant_block_vllm

    # Hopper submodule may hold its own binding used by some imports.
    try:
        import modelopt.torch.quantization.triton.fp4_kernel_hopper as hopper

        if hasattr(hopper, "fp4_fake_quant_block"):
            hopper.fp4_fake_quant_block = fp4_fake_quant_block_vllm
    except ImportError:
        pass

    _INSTALLED = True
    logger.info(
        "[vexact-nvfp4] Installed vLLM-aligned NVFP4 fake-quant kernel "
        "(replaces ModelOpt fp4_fake_quant_block)."
    )
    return True


def uninstall_vllm_aligned_nvfp4_kernel() -> bool:
    """Restore the original ModelOpt ``fp4_fake_quant_block`` (tests / debug)."""
    global _INSTALLED, _ORIGINAL

    if not _INSTALLED or _ORIGINAL is None:
        return False

    try:
        import modelopt.torch.quantization.triton as mtq_triton

        mtq_triton.fp4_fake_quant_block = _ORIGINAL
    except ImportError:
        return False

    try:
        import modelopt.torch.quantization.triton.fp4_kernel_hopper as hopper

        if hasattr(hopper, "fp4_fake_quant_block"):
            hopper.fp4_fake_quant_block = _ORIGINAL
    except ImportError:
        pass

    _INSTALLED = False
    return True


def is_vllm_aligned_nvfp4_kernel_installed() -> bool:
    return bool(_INSTALLED)
