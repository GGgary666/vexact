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

"""Block-scale helpers (vLLM recipe, torch FP8 cast = C++ on B200)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .constants import BLOCK_SIZE, E2M1_MAX


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        asm="rcp.approx.ftz.f32 $0, $1;",
        constraints="=f, f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _output_scale_kernel(
    sf_ptr,
    out_ptr,
    n_elements,
    global_scale_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    sf = tl.load(sf_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gs = tl.load(global_scale_ptr).to(tl.float32)
    inv_gs = _rcp_approx_ftz(gs)
    oscale = tl.where(sf != 0.0, _rcp_approx_ftz(sf * inv_gs), 0.0)
    tl.store(out_ptr + offs, oscale, mask=mask)


def compute_output_scale_vllm(
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Per-block output scale matching vLLM ``cvt_warp_fp16_to_fp4`` recipe."""
    sf = block_scales.contiguous().to(torch.float32)
    gs = global_scale.to(device=sf.device, dtype=torch.float32).reshape(())
    out = torch.empty_like(sf)
    n = sf.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    with torch.cuda.device(sf.device):
        _output_scale_kernel[grid](
            sf,
            out,
            n,
            gs,
            BLOCK=block,
        )
    return out.reshape(block_scales.shape)


def compute_block_scales_vllm(
    x2d: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Return per-block SF in fp32 with shape ``(M, K_blocks)``.

    Match vLLM C++: reduce ``amax`` in the input dtype (bf16/fp16) before fp32
    scale math / FP8 cast.
    """
    m, n = x2d.shape
    blk = x2d.reshape(m, n // BLOCK_SIZE, BLOCK_SIZE)
    vmax = blk.abs().amax(dim=-1).to(torch.float32)
    gs = global_scale.to(device=x2d.device, dtype=torch.float32).reshape(())
    return (gs * vmax / E2M1_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def apply_block_scales_vllm(
    x2d: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply vLLM fake-quant round-trip in high precision (no pack/unpack)."""
    from .e2m1 import round_to_e2m1

    m, n = x2d.shape
    gs = global_scale.to(device=x2d.device, dtype=torch.float32).reshape(())
    blk_hp = x2d.reshape(m, n // BLOCK_SIZE, BLOCK_SIZE)
    blk = blk_hp.to(torch.float32)
    oscale = compute_output_scale_vllm(block_scales, gs).unsqueeze(-1)
    normed = (blk * oscale).clamp(-E2M1_MAX, E2M1_MAX)
    q = round_to_e2m1(normed)
    sf = block_scales.unsqueeze(-1)
    out = q * (sf / gs)
    return out.reshape(m, n).to(dtype=x2d.dtype)
