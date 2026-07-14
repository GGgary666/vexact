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

"""Benchmark VeXact-rollout-style QDQ overhead (input_quantizer fake-quant).

Mirrors the production w4a4 rollout path:
  quantize_model(w4a4) -> disable_weight_quantizers -> keep input_quantizer only

Uses HF ``eager`` attention so we isolate Linear/QDQ cost without VeXact KV-cache
paged attention. This is intentionally NOT a full VeXact engine benchmark; it
answers: how much slower is a forward when NVFP4 input fake-quant is on.

Example (on the training server)::

    source /workspace/gg/venvs/vexact_0706/bin/activate
    cd /workspace/gg/projects/vexact_repo/vexact

    python scripts/bench_rollout_qdq_overhead.py \\
        --model-path /xpfs/fp4/models/Qwen3-1.7B-Base \\
        --seq-len 512 --batch-size 1 --warmup 10 --iters 50

    # Also time decode-shaped (q_len=1) forwards:
    python scripts/bench_rollout_qdq_overhead.py \\
        --model-path /xpfs/fp4/models/Qwen3-1.7B-Base \\
        --seq-len 1 --batch-size 32 --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM


@dataclass
class BenchResult:
    name: str
    ms_mean: float
    ms_std: float
    ms_p50: float
    ms_p90: float
    tokens_per_s: float


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[k]


@torch.inference_mode()
def _time_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    """Return per-iter latency in milliseconds (CUDA-event timed when possible)."""
    use_cuda = input_ids.is_cuda
    for _ in range(warmup):
        model(input_ids=input_ids)
    _sync()

    times_ms: list[float] = []
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            starter.record()
            model(input_ids=input_ids)
            ender.record()
            ender.synchronize()
            times_ms.append(float(starter.elapsed_time(ender)))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            model(input_ids=input_ids)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def _summarize(name: str, times_ms: list[float], n_tokens: int) -> BenchResult:
    mean = statistics.fmean(times_ms)
    std = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    return BenchResult(
        name=name,
        ms_mean=mean,
        ms_std=std,
        ms_p50=_percentile(times_ms, 50),
        ms_p90=_percentile(times_ms, 90),
        tokens_per_s=(n_tokens / (mean / 1000.0)) if mean > 0 else float("nan"),
    )


def _count_enabled(model: nn.Module, suffix: str) -> int:
    try:
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
    except ImportError:
        return 0
    return sum(
        1
        for n, m in model.named_modules()
        if n.endswith(suffix) and isinstance(m, TensorQuantizer) and m.is_enabled
    )


def _prepare_rollout_w4a4(model: nn.Module, *, calibrate: bool = True) -> dict[str, int]:
    """Match VeXact rollout: insert NVFP4 QAT, disable WQ, keep IQ."""
    from vexact.quantization import QATConfig, quantize_model
    from vexact.quantization.fold import (
        build_weight_quantizer_map,
        disable_weight_quantizers,
        fold_model_weights_in_place,
    )

    qat = QATConfig(enable=True, mode="w4a4", calibrate=calibrate)
    quantize_model(model, qat)
    # Fold weights once (production sends folded BF16); then disable live WQ.
    wq_map = build_weight_quantizer_map(model)
    folded = fold_model_weights_in_place(model, wq_map)
    disabled = disable_weight_quantizers(model)
    return {
        "folded_weights": folded,
        "disabled_wq": disabled,
        "wq_enabled": _count_enabled(model, "weight_quantizer"),
        "iq_enabled": _count_enabled(model, "input_quantizer"),
    }


def _disable_all_input_quantizers(model: nn.Module) -> int:
    import modelopt.torch.quantization as mtq

    before = _count_enabled(model, "input_quantizer")
    if before == 0:
        return 0
    mtq.disable_quantizer(model, "*input_quantizer")
    return before - _count_enabled(model, "input_quantizer")


def _time_input_quantizers_only(
    model: nn.Module,
    *,
    batch: int,
    seq_len: int,
    hidden: int,
    warmup: int,
    iters: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[BenchResult]:
    """Microbench: only run enabled input_quantizer modules on a random activation."""
    try:
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
    except ImportError:
        return None

    iqs = [
        m
        for n, m in model.named_modules()
        if n.endswith("input_quantizer") and isinstance(m, TensorQuantizer) and m.is_enabled
    ]
    if not iqs:
        return None

    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)

    @torch.inference_mode()
    def run_all() -> None:
        for iq in iqs:
            iq(x)

    for _ in range(warmup):
        run_all()
    _sync()

    times_ms: list[float] = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            starter.record()
            run_all()
            ender.record()
            ender.synchronize()
            times_ms.append(float(starter.elapsed_time(ender)))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            run_all()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    r = _summarize(f"IQ-only x{len(iqs)} (synthetic act)", times_ms, batch * seq_len)
    return r


def _print_result(r: BenchResult, baseline_ms: Optional[float] = None) -> None:
    slowdown = ""
    if baseline_ms is not None and baseline_ms > 0:
        slowdown = f"  slowdown={r.ms_mean / baseline_ms:.2f}x"
    print(
        f"  {r.name:40s}  "
        f"mean={r.ms_mean:8.2f}ms  std={r.ms_std:6.2f}  "
        f"p50={r.ms_p50:8.2f}  p90={r.ms_p90:8.2f}  "
        f"tok/s={r.tokens_per_s:8.1f}{slowdown}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/xpfs/fp4/models/Qwen3-1.7B-Base",
        help="HF model path (same as training)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument(
        "--skip-iq-microbench",
        action="store_true",
        help="Skip synthetic activation-only IQ microbench",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    n_tokens = args.batch_size * args.seq_len

    print("=" * 72)
    print("VeXact-rollout QDQ overhead bench")
    print(f"  model={args.model_path}")
    print(f"  batch={args.batch_size} seq_len={args.seq_len} tokens/fwd={n_tokens}")
    print(f"  warmup={args.warmup} iters={args.iters} dtype={args.dtype}")
    print("  attn=eager (isolate Linear/QDQ; not full VeXact paged attn)")
    print("=" * 72)

    print("\n[1/4] Load BF16 model (eager attn)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    vocab = int(model.config.vocab_size)
    input_ids = torch.randint(0, min(vocab, 32000), (args.batch_size, args.seq_len), device=device)

    print("\n[2/4] Bench A: BF16 baseline (no QuantModule)...")
    t_bf16 = _time_forward(model, input_ids, warmup=args.warmup, iters=args.iters)
    r_bf16 = _summarize("A) BF16 eager", t_bf16, n_tokens)
    _print_result(r_bf16)

    print("\n[3/4] Insert w4a4 QuantModules + fold W + disable WQ (rollout path)...")
    stats = _prepare_rollout_w4a4(model, calibrate=True)
    print(
        f"  folded={stats['folded_weights']} disabled_wq={stats['disabled_wq']} "
        f"wq_enabled={stats['wq_enabled']} iq_enabled={stats['iq_enabled']}"
    )
    if stats["wq_enabled"] != 0:
        raise RuntimeError("weight_quantizer still enabled; rollout path broken")
    if stats["iq_enabled"] <= 0:
        raise RuntimeError("no enabled input_quantizer; cannot measure IQ overhead")

    print("\n[4/4] Bench B: rollout w4a4 (IQ on, WQ off, folded weights)...")
    t_iq = _time_forward(model, input_ids, warmup=args.warmup, iters=args.iters)
    r_iq = _summarize("B) rollout w4a4 (IQ only)", t_iq, n_tokens)
    _print_result(r_iq, baseline_ms=r_bf16.ms_mean)

    print("\n[extra] Bench C: disable IQ too (folded BF16 + QuantModule shells)...")
    n_dis = _disable_all_input_quantizers(model)
    print(f"  disabled_iq={n_dis} remaining_iq={_count_enabled(model, 'input_quantizer')}")
    t_off = _time_forward(model, input_ids, warmup=args.warmup, iters=args.iters)
    r_off = _summarize("C) IQ disabled (control)", t_off, n_tokens)
    _print_result(r_off, baseline_ms=r_bf16.ms_mean)

    # Re-enable path is not needed; microbench uses a fresh list from before disable.
    # Rebuild IQ-enabled model for microbench if requested.
    if not args.skip_iq_microbench:
        print("\n[micro] Re-quantize for IQ-only synthetic activation bench...")
        # Reload clean model for a clean IQ-enabled state.
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        model.eval()
        stats = _prepare_rollout_w4a4(model, calibrate=True)
        hidden = int(model.config.hidden_size)
        r_micro = _time_input_quantizers_only(
            model,
            batch=args.batch_size,
            seq_len=args.seq_len,
            hidden=hidden,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            dtype=dtype,
        )
        if r_micro is not None:
            print(
                f"  iq_enabled={stats['iq_enabled']} hidden={hidden} "
                f"(same act fed to every IQ; upper-bound style cost)"
            )
            _print_result(r_micro)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    _print_result(r_bf16)
    _print_result(r_iq, baseline_ms=r_bf16.ms_mean)
    _print_result(r_off, baseline_ms=r_bf16.ms_mean)
    overhead_ms = r_iq.ms_mean - r_bf16.ms_mean
    overhead_pct = 100.0 * overhead_ms / r_bf16.ms_mean if r_bf16.ms_mean > 0 else float("nan")
    print(
        f"\n  IQ overhead vs BF16: {overhead_ms:+.2f} ms/fwd "
        f"({overhead_pct:+.1f}%), slowdown={r_iq.ms_mean / r_bf16.ms_mean:.2f}x"
    )
    print(
        "  Note: full VeXact gen also pays paged-attn + no-CUDA-graph; "
        "this bench isolates Linear/QDQ only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
