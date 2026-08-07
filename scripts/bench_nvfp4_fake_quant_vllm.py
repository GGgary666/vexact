#!/usr/bin/env python3
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

"""Benchmark vexact NVFP4 fake-quant backends on B200.

Usage:
  python scripts/bench_nvfp4_fake_quant_vllm.py
  python scripts/bench_nvfp4_fake_quant_vllm.py --shape 32,4096 --iters 200
  python scripts/bench_nvfp4_fake_quant_vllm.py --sweep
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from vexact.quantization.nvfp4 import nvfp4_fake_quant
from vexact.quantization.nvfp4.reference import nvfp4_fake_quant_vllm_qdq
from vexact.quantization.nvfp4.triton_kernel import (
    nvfp4_fake_quant_triton,
    nvfp4_fake_quant_triton_block1d,
    nvfp4_fake_quant_triton_fused,
    triton_is_available,
)
from vexact.quantization.nvfp4.reference import nvfp4_global_amax, nvfp4_global_scale
from vexact.quantization.nvfp4.scales import compute_block_scales_vllm


@dataclass
class BenchRow:
    name: str
    ms_mean: float
    ms_p50: float
    tokens_per_s: float
    gbps: float


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, x: torch.Tensor, warmup: int, iters: int) -> BenchRow:
    for _ in range(warmup):
        fn(x)
    _sync()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    m, n = x.shape
    elems = m * n
    mean_ms = statistics.mean(times)
    p50 = statistics.median(times)
    tps = elems / (mean_ms / 1000.0)
    # bf16 read + bf16 write
    gbps = (elems * 4) / (mean_ms / 1000.0) / 1e9
    return BenchRow(name="", ms_mean=mean_ms, ms_p50=p50, tokens_per_s=tps, gbps=gbps)


def _try_modelopt(x: torch.Tensor):
    try:
        from modelopt.torch.quantization.triton import fp4_fake_quant_block
    except ImportError:
        return None
    amax = nvfp4_global_amax(x)

    def _run(t: torch.Tensor) -> torch.Tensor:
        return fp4_fake_quant_block(t, amax)

    return _run


def _run_shape(m: int, n: int, dtype: torch.dtype, warmup: int, iters: int) -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn(m, n, dtype=dtype, generator=gen) * 0.5).to(device)

    print(f"\n=== shape=({m},{n}) elems={m*n} dtype={dtype} ===")
    print(f"device={torch.cuda.get_device_name()} cap={torch.cuda.get_device_capability()}")

    ref = nvfp4_fake_quant_vllm_qdq(x)
    for name, fn in (
        ("triton_fused", lambda t: nvfp4_fake_quant(t, backend="triton")),
        ("triton_block1d", lambda t: nvfp4_fake_quant(t, backend="triton_block1d")),
    ):
        if not triton_is_available():
            continue
        out = fn(x)
        d = (out.float() - ref.float()).abs().max().item()
        print(f"{name} vs vllm_qdq max_diff={d}")

    runners: list[tuple[str, object]] = [
        ("vllm_qdq", lambda t: nvfp4_fake_quant_vllm_qdq(t)),
        ("eager", lambda t: nvfp4_fake_quant(t, backend="eager")),
    ]
    amax = nvfp4_global_amax(x)
    gs = nvfp4_global_scale(amax)
    sf = compute_block_scales_vllm(x, gs)
    if triton_is_available():
        runners.append(("triton_fused", lambda t: nvfp4_fake_quant_triton_fused(t, gs)))
        runners.append(("triton_block1d", lambda t: nvfp4_fake_quant_triton_block1d(t, gs)))
        runners.append(
            ("api_e2e", lambda t: nvfp4_fake_quant(t, backend="triton"))  # includes amax
        )
        runners.append(
            (
                "api_with_amax",
                lambda t: nvfp4_fake_quant(t, global_amax=amax, backend="triton"),
            )
        )
        runners.append(("triton_staged", lambda t: nvfp4_fake_quant_triton(t, gs)))
        runners.append(
            ("triton_cached_sf", lambda t: nvfp4_fake_quant_triton(t, gs, block_scales=sf))
        )
    mo = _try_modelopt(x)
    if mo is not None:
        runners.append(("modelopt_triton", mo))

    # Bandwidth ceiling: pure memcpy
    y = torch.empty_like(x)

    def _memcpy(t: torch.Tensor) -> torch.Tensor:
        y.copy_(t)
        return y

    runners.append(("memcpy_bf16", _memcpy))

    print("--- benchmark ---")
    rows: list[BenchRow] = []
    for name, fn in runners:
        row = _bench(fn, x, warmup, iters)
        row.name = name
        rows.append(row)
        print(
            f"{name:18s} mean={row.ms_mean:7.3f}ms p50={row.ms_p50:7.3f}ms "
            f" thr={row.tokens_per_s/1e6:8.2f} Melem/s  bw={row.gbps:7.1f} GB/s"
        )

    best = min(rows, key=lambda r: r.ms_mean)
    base = next(r for r in rows if r.name == "vllm_qdq")
    print(f"fastest: {best.name} ({base.ms_mean / best.ms_mean:.2f}x vs vllm_qdq)")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="32,4096", help="M,N")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a shape sweep covering small/QAT/large regimes",
    )
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.sweep:
        shapes = [
            (16, 128),
            (32, 4096),
            (64, 7168),
            (128, 7168),
            (512, 4096),
            (2048, 4096),
            (4096, 4096),
            (8192, 7168),
        ]
        for m, n in shapes:
            _run_shape(m, n, dtype, args.warmup, args.iters)
        return

    m, n = (int(v) for v in args.shape.split(","))
    _run_shape(m, n, dtype, args.warmup, args.iters)


if __name__ == "__main__":
    main()
