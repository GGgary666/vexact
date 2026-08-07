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

"""Triton NVFP4 fake quant aligned with vLLM B200 ``scaled_fp4_quant``.

B200 / CUDA 13.0 / Triton 3.6 notes
-----------------------------------
* Fake-quant is memory-light but launch/occupancy sensitive on typical QAT
  shapes (tens of K–few M elements). HBM3e (~8 TB/s) is not the limiter until
  shapes reach ~10M+ elements.
* Keep vLLM numerics: bf16 amax, ``rcp.approx.ftz`` output_scale, dequant
  ``e2m1 * SF / global_scale``.
* Autotune tile/warps/stages for SM100 (148 SMs). Prefer software E2M1
  buckets over pack/unpack (same bit-exact result, lower latency).
* Fast path skips ``boundary_check`` when tiles divide M/N exactly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .constants import BLOCK_SIZE
from .scales import compute_block_scales_vllm, compute_output_scale_vllm

_TORCH_TO_TL = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


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
def _round_e2m1(abs_scaled):
    # Bucket thresholds match vLLM test_nvfp4_quant.cast_to_fp4 / C++ RTNE.
    return tl.where(
        abs_scaled <= 0.25,
        0.0,
        tl.where(
            abs_scaled < 0.75,
            0.5,
            tl.where(
                abs_scaled <= 1.25,
                1.0,
                tl.where(
                    abs_scaled < 1.75,
                    1.5,
                    tl.where(
                        abs_scaled <= 2.5,
                        2.0,
                        tl.where(
                            abs_scaled < 3.5,
                            3.0,
                            tl.where(abs_scaled <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )


def _fused_configs():
    """SM100-oriented configs; NUM_FP4_BLOCKS = TILE_N // 16."""
    cfgs = []
    for tile_m, tile_n, warps, stages in (
        (8, 64, 2, 2),
        (16, 64, 2, 2),
        (16, 64, 4, 2),
        (16, 128, 4, 2),
        (16, 128, 4, 3),
        (32, 64, 4, 2),
        (32, 128, 4, 2),
        (32, 128, 4, 3),
        (32, 128, 8, 2),
        (32, 256, 4, 2),
        (32, 256, 8, 2),
        (32, 256, 8, 3),
        (64, 64, 4, 2),
        (64, 128, 4, 2),
        (64, 128, 8, 2),
        (64, 256, 8, 2),
        (64, 256, 8, 3),
        (128, 128, 8, 2),
        (128, 256, 8, 2),
        (128, 256, 8, 3),
    ):
        cfgs.append(
            triton.Config(
                {
                    "TILE_M": tile_m,
                    "TILE_N": tile_n,
                    "NUM_FP4_BLOCKS": tile_n // BLOCK_SIZE,
                },
                num_warps=warps,
                num_stages=stages,
            )
        )
    return cfgs


def _prune_fused_configs(configs, named_args, **_kwargs):
    """Prefer tiles that divide M/N so the aligned (no boundary) path is safe."""
    m, n = int(named_args["M"]), int(named_args["N"])
    aligned_ok = [
        c
        for c in configs
        if m % c.kwargs["TILE_M"] == 0 and n % c.kwargs["TILE_N"] == 0
    ]
    return aligned_ok if aligned_ok else configs


@triton.jit
def _fused_kernel_body(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    """Fused amax + FP8 SF + rcp oscale + E2M1 QDQ (software buckets)."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    inv_gs_exact = 1.0 / global_scale
    inv_gs_approx = _rcp_approx_ftz(global_scale)

    if ALIGNED:
        tile = tl.load(x_block_ptr)
    else:
        tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")

    tile_abs_hp = tl.abs(tile)
    abs_rs = tl.reshape(tile_abs_hp, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    vmax = tl.max(abs_rs, axis=2).to(tl.float32)
    vmax = tl.reshape(vmax, (TILE_M, NUM_FP4_BLOCKS, 1))

    sf = (global_scale * vmax * (1.0 / 6.0)).to(tl.float8e4nv).to(tl.float32)
    oscale = tl.where(sf != 0.0, _rcp_approx_ftz(sf * inv_gs_approx), 0.0)

    abs_f32 = abs_rs.to(tl.float32)
    q_val = _round_e2m1(abs_f32 * oscale)
    out_rs = tl.where(sf != 0.0, q_val * sf * inv_gs_exact, 0.0)

    tile_rs = tl.reshape(tile.to(tl.float32), (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    out_rs = tl.where(tile_rs >= 0, out_rs, -out_rs)
    out = tl.reshape(out_rs, (TILE_M, TILE_N)).to(OUT_DTYPE)

    if ALIGNED:
        tl.store(y_block_ptr, out)
    else:
        tl.store(y_block_ptr, out, boundary_check=(0, 1))


@triton.autotune(
    configs=_fused_configs(),
    key=["M", "N", "ALIGNED"],
    prune_configs_by={"early_config_prune": _prune_fused_configs},
)
@triton.jit
def _nvfp4_fake_quant_fused_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    _fused_kernel_body(
        x_ptr,
        y_ptr,
        M,
        N,
        global_scale_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        BLOCK_SIZE,
        TILE_M,
        TILE_N,
        NUM_FP4_BLOCKS,
        OUT_DTYPE,
        ALIGNED,
    )


@triton.jit
def _nvfp4_fake_quant_fused_kernel_manual(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    _fused_kernel_body(
        x_ptr,
        y_ptr,
        M,
        N,
        global_scale_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        BLOCK_SIZE,
        TILE_M,
        TILE_N,
        NUM_FP4_BLOCKS,
        OUT_DTYPE,
        ALIGNED,
    )


def _block1d_configs():
    cfgs = []
    for tile_blocks, warps, stages in (
        (4, 2, 2),
        (8, 2, 2),
        (8, 4, 2),
        (16, 4, 2),
        (16, 4, 3),
        (32, 4, 2),
        (32, 8, 2),
        (64, 4, 2),
        (64, 8, 2),
        (64, 8, 3),
        (128, 8, 2),
        (128, 8, 3),
    ):
        cfgs.append(
            triton.Config(
                {"TILE_BLOCKS": tile_blocks},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return cfgs


@triton.autotune(
    configs=_block1d_configs(),
    key=["NUM_BLOCKS"],
)
@triton.jit
def _nvfp4_fake_quant_block1d_kernel(
    x_ptr,
    y_ptr,
    NUM_BLOCKS,
    global_scale_ptr,
    BLOCK_SIZE: tl.constexpr,
    TILE_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """1D launch over FP4 blocks — good occupancy on SM100 for large N."""
    pid = tl.program_id(0)
    block_start = pid * TILE_BLOCKS
    block_offs = block_start + tl.arange(0, TILE_BLOCKS)
    block_mask = block_offs < NUM_BLOCKS

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    inv_gs_exact = 1.0 / global_scale
    inv_gs_approx = _rcp_approx_ftz(global_scale)

    # Flattened layout: block b owns elems [b*16, (b+1)*16).
    elem_offs = block_offs[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    elem_mask = block_mask[:, None]

    x = tl.load(x_ptr + elem_offs, mask=elem_mask, other=0.0)
    x_abs_hp = tl.abs(x)
    vmax = tl.max(x_abs_hp, axis=1).to(tl.float32)

    sf = (global_scale * vmax * (1.0 / 6.0)).to(tl.float8e4nv).to(tl.float32)
    oscale = tl.where(sf != 0.0, _rcp_approx_ftz(sf * inv_gs_approx), 0.0)

    abs_f32 = x_abs_hp.to(tl.float32)
    q_val = _round_e2m1(abs_f32 * oscale[:, None])
    out = tl.where(sf[:, None] != 0.0, q_val * sf[:, None] * inv_gs_exact, 0.0)
    out = tl.where(x.to(tl.float32) >= 0, out, -out)

    tl.store(y_ptr + elem_offs, out.to(OUT_DTYPE), mask=elem_mask)


@triton.jit
def _nvfp4_fake_quant_vllm_kernel(
    x_ptr,
    y_ptr,
    oscale_ptr,
    sf_ptr,
    M,
    N,
    K_BLOCKS,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_osm,
    stride_osn,
    stride_sfm,
    stride_sfn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Legacy staged kernel (precomputed SF/oscale). Kept for A/B benches."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N
    col_block_start = col_start // BLOCK_SIZE

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    oscale_block_ptr = tl.make_block_ptr(
        base=oscale_ptr,
        shape=(M, K_BLOCKS),
        strides=(stride_osm, stride_osn),
        offsets=(row_start, col_block_start),
        block_shape=(TILE_M, NUM_FP4_BLOCKS),
        order=(1, 0),
    )
    sf_block_ptr = tl.make_block_ptr(
        base=sf_ptr,
        shape=(M, K_BLOCKS),
        strides=(stride_sfm, stride_sfn),
        offsets=(row_start, col_block_start),
        block_shape=(TILE_M, NUM_FP4_BLOCKS),
        order=(1, 0),
    )

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    tile_f32 = tile.to(tl.float32)
    oscale_tile = tl.load(oscale_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    sf_tile = tl.load(sf_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    tile_rs = tl.reshape(tile_f32, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    oscale_rs = tl.reshape(oscale_tile, (TILE_M, NUM_FP4_BLOCKS, 1))
    sf_rs = tl.reshape(sf_tile, (TILE_M, NUM_FP4_BLOCKS, 1))
    x_abs = tl.abs(tile_rs)

    abs_scaled = x_abs * oscale_rs
    q_val = _round_e2m1(abs_scaled)
    sf_pos = sf_rs > 0.0
    x_rescaled = tl.where(sf_pos, q_val * sf_rs / global_scale, 0.0)
    x_rescaled = tl.where(tile_rs >= 0, x_rescaled, -x_rescaled)

    tl.store(y_block_ptr, tl.reshape(x_rescaled, (TILE_M, TILE_N)).to(OUT_DTYPE), boundary_check=(0, 1))


def _dtype_to_tl(dtype: torch.dtype):
    if dtype not in _TORCH_TO_TL:
        raise ValueError(f"Unsupported dtype for nvfp4 fake quant: {dtype}")
    return _TORCH_TO_TL[dtype]


def nvfp4_fake_quant_triton_fused(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
    tile_rows: int | None = None,
    tile_cols: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    autotune: bool = False,
    use_hw_cvt: bool | None = None,
) -> torch.Tensor:
    """Fused fake quant (preferred fast path on B200 / CUDA 13 / Triton 3.6).

    Defaults (from B200 sweep): ``TILE_M=32, TILE_N=128, warps=4, stages=2``.
    These beat ModelOpt Triton while staying bit-exact vs vLLM QDQ.

    ``use_hw_cvt`` is accepted for API compatibility but ignored: software
    buckets are bit-exact and faster than pack/unpack on B200.
    """
    del use_hw_cvt  # kept for call-site compatibility
    if not x.is_cuda:
        raise ValueError("nvfp4_fake_quant_triton_fused expects a CUDA tensor.")
    if block_size != BLOCK_SIZE:
        raise ValueError(f"Only block_size={BLOCK_SIZE} is supported.")

    x_shape = x.shape
    x2d = x.reshape(-1, x_shape[-1]).contiguous()
    m, n = x2d.shape
    if n % block_size != 0:
        raise ValueError(f"last dim must be divisible by {block_size}, got {n}")

    gs = global_scale.to(device=x.device, dtype=torch.float32).reshape(())
    y = torch.empty_like(x2d)

    if autotune and tile_rows is None and tile_cols is None:
        aligned = any(
            m % c.kwargs["TILE_M"] == 0 and n % c.kwargs["TILE_N"] == 0 for c in _fused_configs()
        )
        grid = lambda meta: (triton.cdiv(m, meta["TILE_M"]), triton.cdiv(n, meta["TILE_N"]))
        with torch.cuda.device(x.device):
            _nvfp4_fake_quant_fused_kernel[grid](
                x2d,
                y,
                m,
                n,
                gs,
                x2d.stride(0),
                x2d.stride(1),
                y.stride(0),
                y.stride(1),
                BLOCK_SIZE=block_size,
                OUT_DTYPE=_dtype_to_tl(x2d.dtype),
                ALIGNED=aligned,
            )
        return y.view(x_shape)

    # B200-tuned defaults (see scripts/bench_nvfp4_fake_quant_vllm.py --sweep).
    tile_rows = 32 if tile_rows is None else tile_rows
    tile_cols = 128 if tile_cols is None else tile_cols
    num_warps = 4 if num_warps is None else num_warps
    num_stages = 2 if num_stages is None else num_stages

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    aligned = (m % tile_rows == 0) and (n % tile_cols_aligned == 0)
    launch_kwargs: dict = {
        "BLOCK_SIZE": block_size,
        "TILE_M": tile_rows,
        "TILE_N": tile_cols_aligned,
        "NUM_FP4_BLOCKS": tile_cols_aligned // block_size,
        "OUT_DTYPE": _dtype_to_tl(x2d.dtype),
        "ALIGNED": aligned,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    grid = (triton.cdiv(m, tile_rows), triton.cdiv(n, tile_cols_aligned))
    with torch.cuda.device(x.device):
        _nvfp4_fake_quant_fused_kernel_manual[grid](
            x2d,
            y,
            m,
            n,
            gs,
            x2d.stride(0),
            x2d.stride(1),
            y.stride(0),
            y.stride(1),
            **launch_kwargs,
        )
    return y.view(x_shape)


def nvfp4_fake_quant_triton_block1d(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """1D autotuned block kernel (alternative launch strategy)."""
    if not x.is_cuda:
        raise ValueError("nvfp4_fake_quant_triton_block1d expects a CUDA tensor.")
    if block_size != BLOCK_SIZE:
        raise ValueError(f"Only block_size={BLOCK_SIZE} is supported.")

    x_shape = x.shape
    x2d = x.reshape(-1, x_shape[-1]).contiguous()
    m, n = x2d.shape
    if n % block_size != 0:
        raise ValueError(f"last dim must be divisible by {block_size}, got {n}")

    gs = global_scale.to(device=x.device, dtype=torch.float32).reshape(())
    y = torch.empty_like(x2d)
    num_blocks = m * (n // block_size)
    grid = lambda meta: (triton.cdiv(num_blocks, meta["TILE_BLOCKS"]),)
    with torch.cuda.device(x.device):
        _nvfp4_fake_quant_block1d_kernel[grid](
            x2d,
            y,
            num_blocks,
            gs,
            BLOCK_SIZE=block_size,
            OUT_DTYPE=_dtype_to_tl(x2d.dtype),
        )
    return y.view(x_shape)


def nvfp4_fake_quant_triton(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
    tile_rows: int = 16,
    tile_cols: int = 64,
    num_warps: int | None = None,
    num_stages: int | None = None,
    block_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Staged fake quant: torch FP8 block scales + Triton E2M1 rounding."""
    if not x.is_cuda:
        raise ValueError("nvfp4_fake_quant_triton expects a CUDA tensor.")

    x_shape = x.shape
    x2d = x.reshape(-1, x_shape[-1]).contiguous()
    m, n = x2d.shape
    if n % block_size != 0:
        raise ValueError(f"last dim must be divisible by {block_size}, got {n}")

    gs = global_scale.to(device=x.device, dtype=torch.float32).reshape(())
    sf = block_scales if block_scales is not None else compute_block_scales_vllm(x2d, gs)
    sf = sf.contiguous()
    oscale = compute_output_scale_vllm(sf, gs)

    y = torch.empty_like(x2d)
    k_blocks = n // block_size
    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    launch_kwargs: dict = {
        "BLOCK_SIZE": block_size,
        "TILE_M": tile_rows,
        "TILE_N": tile_cols_aligned,
        "NUM_FP4_BLOCKS": num_fp4_blocks,
        "OUT_DTYPE": _dtype_to_tl(x2d.dtype),
    }
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    grid = lambda *_: (triton.cdiv(m, tile_rows), triton.cdiv(n, tile_cols_aligned))
    with torch.cuda.device(x.device):
        _nvfp4_fake_quant_vllm_kernel[grid](
            x2d,
            y,
            oscale,
            sf,
            m,
            n,
            k_blocks,
            gs,
            x2d.stride(0),
            x2d.stride(1),
            y.stride(0),
            y.stride(1),
            oscale.stride(0),
            oscale.stride(1),
            sf.stride(0),
            sf.stride(1),
            **launch_kwargs,
        )
    return y.view(x_shape)


def triton_is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.get_device_capability()[0] >= 8
