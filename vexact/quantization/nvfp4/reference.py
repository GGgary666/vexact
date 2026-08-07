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

"""Ground-truth NVFP4 fake quant via vLLM ``scaled_fp4_quant`` + dequant."""

from __future__ import annotations

import torch

from .constants import BLOCK_SIZE, E2M1_MAX, FP8_E4M3_MAX
from .e2m1 import dequantize_nvfp4_packed, round_to_e2m1
from .scales import apply_block_scales_vllm, compute_block_scales_vllm


_GS_NUMER = float(FP8_E4M3_MAX * E2M1_MAX)


def nvfp4_global_amax(x: torch.Tensor) -> torch.Tensor:
    # ``amax`` over abs is a single reduction kernel (faster than abs().max() chain).
    return torch.amax(x.abs()).to(dtype=torch.float32)


def nvfp4_global_scale(amax: torch.Tensor) -> torch.Tensor:
    # Scalar / tensor keeps device/dtype without host ``torch.tensor`` sync.
    return _GS_NUMER / amax.to(dtype=torch.float32)


@torch.inference_mode()
def nvfp4_fake_quant_vllm_qdq(x: torch.Tensor, global_amax: torch.Tensor | None = None) -> torch.Tensor:
    """Fake quant = vLLM real quant + dequant (exact B200 inference numerics)."""
    from vllm import _custom_ops as vllm_ops

    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1]).contiguous()
    if global_amax is None:
        global_amax = nvfp4_global_amax(flat)
    gs = nvfp4_global_scale(global_amax)
    packed, block_scale = vllm_ops.scaled_fp4_quant(flat, gs, is_sf_swizzled_layout=True)
    out = dequantize_nvfp4_packed(packed, block_scale, gs, flat.dtype)
    return out.reshape(orig_shape)


@torch.inference_mode()
def nvfp4_fake_quant_eager(x: torch.Tensor, global_amax: torch.Tensor | None = None) -> torch.Tensor:
    """Python mirror of vLLM C++ recipe (matches QDQ on tested B200 shapes)."""
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1]).contiguous()
    if global_amax is None:
        global_amax = nvfp4_global_amax(flat)
    gs = nvfp4_global_scale(global_amax)
    sf = compute_block_scales_vllm(flat, gs)
    out = apply_block_scales_vllm(flat, sf, gs)
    return out.reshape(orig_shape)
