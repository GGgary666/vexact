#!/usr/bin/env python3
"""Diagnose ModelOpt NVFP4 fake-quant vs vLLM real quant+dequant mismatch (bf16).

Run on B200:
  source /workspace/gg/venvs/vexact_0706/bin/activate
  python scripts/diagnose_nvfp4_fake_vs_real.py
"""

from __future__ import annotations

import torch
from vllm import _custom_ops as vllm_ops
from vllm.scalar_type import scalar_types
from modelopt.torch.quantization.triton import fp4_fake_quant_block

BLOCK = 16
E2M1_MAX = float(scalar_types.float4_e2m1f.max())
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def global_amax(x: torch.Tensor) -> torch.Tensor:
    return x.abs().max().to(torch.float32)


def global_scale(amax: torch.Tensor) -> torch.Tensor:
    return torch.tensor(FP8_MAX * E2M1_MAX, device=amax.device, dtype=torch.float32) / amax


def recover_swizzled_scales(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    scale_n = n // BLOCK
    rounded_m = (m + 127) // 128 * 128
    rounded_n = (scale_n + 3) // 4 * 4
    tmp = scale.view(1, rounded_m // 128, rounded_n // 4, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    return tmp.reshape(rounded_m, rounded_n).to(torch.float32)[:m, :scale_n]


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    m, pn = packed.shape
    flat = packed.flatten()
    low, high = flat & 0x0F, (flat >> 4) & 0x0F
    idx = torch.stack((low, high), dim=1).flatten().to(torch.long)
    signs = (idx & 0x08).bool()
    mag = idx & 0x07
    lut = E2M1_LUT.to(packed.device)
    return (lut[mag] * torch.where(signs, -1.0, 1.0)).reshape(m, pn * 2)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x = x.abs()
    out = torch.zeros_like(x)
    out[(x > 0.25) & (x < 0.75)] = 0.5
    out[(x >= 0.75) & (x <= 1.25)] = 1.0
    out[(x > 1.25) & (x < 1.75)] = 1.5
    out[(x >= 1.75) & (x <= 2.5)] = 2.0
    out[(x > 2.5) & (x < 3.5)] = 3.0
    out[(x >= 3.5) & (x <= 5.0)] = 4.0
    out[x > 5.0] = 6.0
    return out * sign


def dequant_cpp(packed: torch.Tensor, sf: torch.Tensor, gs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    m, pn = packed.shape
    n = pn * 2
    vals = unpack_fp4(packed).reshape(m, n // BLOCK, BLOCK)
    sf_lin = recover_swizzled_scales(sf.view(torch.float8_e4m3fn), m, n)
    out = (vals * (sf_lin / gs).unsqueeze(-1)).reshape(m, n)
    return out.to(dtype)


def cpp_qdq(x: torch.Tensor) -> torch.Tensor:
    gs = global_scale(global_amax(x))
    packed, sf = vllm_ops.scaled_fp4_quant(x.reshape(-1, x.shape[-1]), gs, is_sf_swizzled_layout=True)
    return dequant_cpp(packed, sf, gs, x.dtype).reshape(x.shape)


def modelopt_fake(x: torch.Tensor, *, tile_cols: int | None = None) -> torch.Tensor:
    amax = global_amax(x)
    kwargs = {"block_size": BLOCK}
    if tile_cols is not None:
        kwargs["tile_cols"] = tile_cols
        kwargs["tile_rows"] = x.reshape(-1, x.shape[-1]).shape[0]
    return fp4_fake_quant_block(x, amax, **kwargs)


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    d = (a.float() - b.float()).abs()
    print(
        f"{name:40s} max={d.max().item():.6e} mean={d.mean().item():.6e} "
        f"rel_l2={(d.norm() / b.float().norm().clamp_min(1e-12)).item():.6e} "
        f"nz={(d > 0).float().mean().item():.4f}"
    )


def compare_scales(x: torch.Tensor) -> None:
    gs = global_scale(global_amax(x))
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    m, n = flat.shape
    _, sf_cpp = vllm_ops.scaled_fp4_quant(flat, gs, is_sf_swizzled_layout=True)
    sf_cpp_lin = recover_swizzled_scales(sf_cpp.view(torch.float8_e4m3fn), m, n)

    blk = flat.reshape(m, n // BLOCK, BLOCK).float()
    vmax = blk.abs().amax(dim=-1)
    # vLLM stores fp8(gs * vmax / 6)
    sf_ref = (gs * vmax / E2M1_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    # ModelOpt uses fp8(vmax / (6*gs)) * gs
    sf_mo = (vmax / (E2M1_MAX * gs)).to(torch.float8_e4m3fn).to(torch.float32) * gs

    d_ref_cpp = (sf_ref - sf_cpp_lin).abs()
    d_mo_ref = (sf_mo - sf_ref).abs()
    d_mo_cpp = (sf_mo - sf_cpp_lin).abs()
    print("--- block scale tensors ---")
    print(f"  ref vs cpp SF   max={d_ref_cpp.max().item():.6e} mean={d_ref_cpp.mean().item():.6e}")
    print(f"  modelopt vs ref max={d_mo_ref.max().item():.6e} mean={d_mo_ref.mean().item():.6e}")
    print(f"  modelopt vs cpp max={d_mo_cpp.max().item():.6e} mean={d_mo_cpp.mean().item():.6e}")
    if d_mo_ref.max().item() > 0:
        idx = d_mo_ref.argmax()
        r, c = idx // sf_ref.shape[1], idx % sf_ref.shape[1]
        print(
            f"  worst SF cell ({r},{c}): vmax={vmax[r,c].item():.6g} "
            f"ref={sf_ref[r,c].item():.6g} mo={sf_mo[r,c].item():.6g} cpp={sf_cpp_lin[r,c].item():.6g}"
        )


def compare_packed(x: torch.Tensor, gs: torch.Tensor) -> None:
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    m, n = flat.shape
    packed_cpp, sf_cpp = vllm_ops.scaled_fp4_quant(flat, gs, is_sf_swizzled_layout=True)
    fake = modelopt_fake(x).reshape(flat.shape).float()
    ref = vllm_formula_fake(flat, gs).float()
    cpp = dequant_cpp(packed_cpp, sf_cpp, gs, flat.dtype).float()
    d_ref_mo = (ref - fake).abs()
    d_cpp_ref = (cpp - ref).abs()
    d_cpp_mo = (cpp - fake).abs()
    print("--- denormalized fake-quant outputs ---")
    print(f"  vllm_formula vs modelopt max={d_ref_mo.max().item():.6e} nz={(d_ref_mo>0).float().mean().item():.4f}")
    print(f"  cpp_qdq vs vllm_formula max={d_cpp_ref.max().item():.6e}")
    print(f"  cpp_qdq vs modelopt max={d_cpp_mo.max().item():.6e}")


def tile_boundary_stats(x: torch.Tensor, fake: torch.Tensor, real: torch.Tensor) -> None:
    diff = (fake.float() - real.float()).abs()
    m, n = x.reshape(-1, x.shape[-1]).shape
    cols = torch.arange(n, device=x.device)
    # modelopt default tile width along N is 64
    in_tile_edge = (cols % 64) == 0
    mask = in_tile_edge.unsqueeze(0).expand(m, -1)
    edge = diff.reshape(m, n)[mask]
    interior = diff.reshape(m, n)[~mask]
    print("--- tile boundary (col%64==0) vs interior ---")
    print(f"  edge max={edge.max().item():.6e} mean={edge.mean().item():.6e}")
    print(f"  interior max={interior.max().item():.6e} mean={interior.mean().item():.6e}")


@torch.inference_mode()
def vllm_formula_fake(x: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """Fake quant using the same SF + E2M1 recipe as vLLM C++ (Python mirror)."""
    m, n = x.shape
    blk = x.reshape(m, n // BLOCK, BLOCK).to(torch.float32)
    vmax = blk.abs().amax(dim=-1, keepdim=True)
    sf = (gs * vmax / E2M1_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    oscale = torch.where(sf != 0, gs / sf, torch.zeros_like(sf))
    normed = (blk * oscale).clamp(-E2M1_MAX, E2M1_MAX)
    q = cast_to_fp4(normed.reshape(m, n))
    return (q.reshape(m, n // BLOCK, BLOCK) * (sf / gs)).reshape(m, n).to(x.dtype)


def diagnose_shape(shape: tuple[int, int], seed: int = 0, device: torch.device | None = None) -> None:
    if device is None:
        device = torch.device("cuda")
    print(f"\n{'='*72}\nshape={shape} bf16 seed={seed}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(*shape, dtype=torch.bfloat16, generator=gen) * 0.5).to(device)

    gs = global_scale(global_amax(x))
    flat = x.reshape(-1, x.shape[-1])
    ref = vllm_formula_fake(flat, gs).reshape(shape)
    mo_def = modelopt_fake(x)
    try:
        mo_wide = modelopt_fake(x, tile_cols=64)  # triton requires power-of-2 tile
    except Exception as exc:  # pragma: no cover
        mo_wide = mo_def
        print(f"  (skip wide tile: {exc})")
    cpp = cpp_qdq(x)

    report("vllm-formula fake vs cpp_qdq", ref, cpp)
    report("modelopt(default tile) vs cpp_qdq", mo_def, cpp)
    report("modelopt(64-col tile) vs cpp_qdq", mo_wide, cpp)
    report("modelopt(default) vs vllm-formula", mo_def, ref)

    # fp32 output path: avoid final bf16 cast difference
    mo_fp32 = modelopt_fake(x.float()).to(torch.bfloat16)
    report("modelopt fp32-in vs cpp_qdq", mo_fp32, cpp)

    compare_scales(x)
    compare_packed(x, gs)
    tile_boundary_stats(x, mo_def, cpp)

    diff = (mo_def.float() - cpp.float()).abs().reshape(-1, shape[1])
    worst = diff.max(dim=1).values.argmax().item()
    col_worst = diff[worst].argmax().item()
    print(
        f"worst row={worst} col={col_worst}: x={x.reshape(-1, shape[1])[worst,col_worst].item():.6g} "
        f"mo={mo_def.reshape(-1, shape[1])[worst,col_worst].item():.6g} "
        f"cpp={cpp.reshape(-1, shape[1])[worst,col_worst].item():.6g}"
    )


def main() -> None:
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name()} cap={torch.cuda.get_device_capability()}")
    for shape in [(1, 64), (32, 4096), (128, 7168), (64, 7168)]:
        diagnose_shape(shape, device=device)


if __name__ == "__main__":
    main()
