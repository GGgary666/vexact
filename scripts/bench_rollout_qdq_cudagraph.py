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

"""Benchmark eager vs CUDA-graph replay for VeXact-rollout-style QDQ.

Companion to ``bench_rollout_qdq_overhead.py``. Asks:

  1) How much does CUDA-graph replay help BF16?
  2) Can we capture a CUDA graph with w4a4 input_quantizer (伪量化) enabled?
  3) If capture works, how much IQ overhead remains under graph replay?

CUDA-graph capture is run in a **spawned subprocess**. A failed capture can
poison the CUDA context (``Offset increment outside graph capture``); isolating
it keeps the parent process able to finish IQ eager / summary.

Example::

    source /workspace/gg/venvs/vexact_0706/bin/activate
    cd /workspace/gg/projects/vexact_repo/vexact

    python scripts/bench_rollout_qdq_cudagraph.py \\
        --model-path /xpfs/fp4/models/Qwen3-1.7B-Base \\
        --seq-len 1 --batch-size 32 --warmup 20 --iters 100

    python scripts/bench_rollout_qdq_cudagraph.py \\
        --model-path /xpfs/fp4/models/Qwen3-1.7B-Base \\
        --seq-len 512 --batch-size 1 --warmup 10 --iters 50
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import traceback
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
    ok: bool
    detail: str = ""


@dataclass
class StaticInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

    def clone_static(self) -> "StaticInputs":
        return StaticInputs(
            input_ids=self.input_ids.clone(),
            attention_mask=self.attention_mask.clone(),
            position_ids=self.position_ids.clone(),
        )

    def copy_from(self, other: "StaticInputs") -> None:
        self.input_ids.copy_(other.input_ids)
        self.attention_mask.copy_(other.attention_mask)
        self.position_ids.copy_(other.position_ids)

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
            "use_cache": False,
        }


def _sync() -> None:
    torch.cuda.synchronize()


def _pct(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[k]


def _summarize(name: str, times_ms: list[float]) -> BenchResult:
    return BenchResult(
        name=name,
        ms_mean=statistics.fmean(times_ms),
        ms_std=statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        ok=True,
    )


def _short_err(exc: BaseException, limit: int = 240) -> str:
    msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return msg if len(msg) <= limit else msg[: limit - 3] + "..."


def _print_result(
    r: BenchResult,
    times: Optional[list[float]] = None,
    baseline: Optional[float] = None,
) -> None:
    if not r.ok:
        print(f"  {r.name:42s}  FAIL: {r.detail}")
        return
    p50 = _pct(times, 50) if times else r.ms_mean
    p90 = _pct(times, 90) if times else r.ms_mean
    extra = ""
    if baseline is not None and baseline > 0:
        extra = f"  vs_eager={r.ms_mean / baseline:.2f}x  delta={(r.ms_mean - baseline):+.2f}ms"
    print(
        f"  {r.name:42s}  mean={r.ms_mean:8.2f}ms  std={r.ms_std:6.2f}  "
        f"p50={p50:8.2f}  p90={p90:8.2f}{extra}"
    )


def _make_inputs(batch: int, seq_len: int, vocab: int, device: torch.device) -> StaticInputs:
    input_ids = torch.randint(0, min(vocab, 32000), (batch, seq_len), device=device)
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    return StaticInputs(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)


def _forward(model: nn.Module, inputs: StaticInputs):
    return model(**inputs.as_kwargs())


def _load_model(path: str, dtype: torch.dtype, device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def _count_enabled(model: nn.Module, suffix: str) -> int:
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    return sum(
        1
        for n, m in model.named_modules()
        if n.endswith(suffix) and isinstance(m, TensorQuantizer) and m.is_enabled
    )


def _prepare_rollout_w4a4(model: nn.Module) -> dict[str, int]:
    from vexact.quantization import QATConfig, quantize_model
    from vexact.quantization.fold import (
        build_weight_quantizer_map,
        disable_weight_quantizers,
        fold_model_weights_in_place,
    )

    quantize_model(model, QATConfig(enable=True, mode="w4a4", calibrate=True))
    wq_map = build_weight_quantizer_map(model)
    folded = fold_model_weights_in_place(model, wq_map)
    disabled = disable_weight_quantizers(model)
    return {
        "folded": folded,
        "disabled_wq": disabled,
        "wq_enabled": _count_enabled(model, "weight_quantizer"),
        "iq_enabled": _count_enabled(model, "input_quantizer"),
    }


@torch.inference_mode()
def _time_eager(model: nn.Module, inputs: StaticInputs, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        _forward(model, inputs)
    _sync()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    out: list[float] = []
    for _ in range(iters):
        starter.record()
        _forward(model, inputs)
        ender.record()
        ender.synchronize()
        out.append(float(starter.elapsed_time(ender)))
    return out


@torch.inference_mode()
def _try_cudagraph_inplace(
    model: nn.Module,
    inputs: StaticInputs,
    *,
    warmup: int,
    iters: int,
) -> tuple[Optional[list[float]], str]:
    static = inputs.clone_static()
    graph: Optional[torch.cuda.CUDAGraph] = None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(max(3, min(warmup, 5))):
                _forward(model, static)
        torch.cuda.current_stream().wait_stream(s)
        _sync()
        for _ in range(max(warmup, 1)):
            _forward(model, static)
        _sync()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _ = _forward(model, static)
        _sync()
    except Exception as exc:  # noqa: BLE001
        return None, _short_err(exc)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    try:
        for _ in range(iters):
            static.copy_from(inputs)
            starter.record()
            assert graph is not None
            graph.replay()
            ender.record()
            ender.synchronize()
            times.append(float(starter.elapsed_time(ender)))
    except Exception as exc:  # noqa: BLE001
        return None, f"replay failed: {_short_err(exc)}"
    return times, "ok"


def _cudagraph_subprocess_main(payload: dict[str, Any], result_path: str) -> None:
    """Child entry: load model, optional w4a4, try CUDA graph, write JSON result."""
    out: dict[str, Any] = {"ok": False, "times_ms": None, "detail": "", "traceback": ""}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required in subprocess")
        device = torch.device("cuda")
        dtype = getattr(torch, payload["dtype"])
        model = _load_model(payload["model_path"], dtype, device)
        vocab = int(model.config.vocab_size)
        inputs = _make_inputs(payload["batch_size"], payload["seq_len"], vocab, device)

        mode = payload["mode"]
        if mode == "w4a4":
            stats = _prepare_rollout_w4a4(model)
            if stats["iq_enabled"] <= 0 or stats["wq_enabled"] != 0:
                raise RuntimeError(f"unexpected quantizer state: {stats}")
            inputs = _make_inputs(payload["batch_size"], payload["seq_len"], vocab, device)

        times, status = _try_cudagraph_inplace(
            model,
            inputs,
            warmup=int(payload["warmup"]),
            iters=int(payload["iters"]),
        )
        if times is None:
            out["detail"] = status
        else:
            out["ok"] = True
            out["times_ms"] = times
            out["detail"] = status
    except Exception as exc:  # noqa: BLE001
        out["detail"] = _short_err(exc)
        out["traceback"] = traceback.format_exc()

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f)


def _run_cudagraph_subprocess(payload: dict[str, Any], *, label: str) -> tuple[Optional[list[float]], str]:
    """Run CUDA-graph attempt in a fresh process so failures cannot poison parent CUDA."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="vexact_cg_") as td:
        result_path = str(Path(td) / "result.json")
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_cudagraph_subprocess_main,
            args=(payload, result_path),
            name=f"cudagraph-{label}",
        )
        print(f"  (running {label} CUDA-graph attempt in subprocess pid pending...)")
        proc.start()
        proc.join(timeout=float(payload.get("timeout_s", 1800)))
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=30)
            return None, "subprocess timed out / killed"
        if proc.exitcode not in (0, None) and not Path(result_path).exists():
            return None, f"subprocess exited with code {proc.exitcode}"
        if not Path(result_path).exists():
            return None, f"subprocess produced no result (exitcode={proc.exitcode})"
        data = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if data.get("ok") and data.get("times_ms"):
            return list(map(float, data["times_ms"])), "ok"
        detail = data.get("detail") or "unknown failure"
        tb = data.get("traceback") or ""
        if tb:
            # Keep parent log readable; first line of detail is enough unless empty.
            pass
        return None, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/xpfs/fp4/models/Qwen3-1.7B-Base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--graph-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    n_tok = args.batch_size * args.seq_len

    print("=" * 72)
    print("Eager vs CUDA-graph bench (rollout-style QDQ)")
    print(f"  model={args.model_path}")
    print(f"  batch={args.batch_size} seq_len={args.seq_len} tokens/fwd={n_tok}")
    print(f"  warmup={args.warmup} iters={args.iters} dtype={args.dtype} attn=eager")
    print("  forward kwargs: use_cache=False + static attention_mask/position_ids")
    print("  CUDA-graph attempts: isolated spawn subprocess (poison-safe)")
    print("=" * 72)

    graph_payload_base = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "iters": args.iters,
        "timeout_s": args.graph_timeout_s,
    }

    # ---- BF16 eager (parent) ----
    print("\n[1] BF16 model...")
    model = _load_model(args.model_path, dtype, device)
    vocab = int(model.config.vocab_size)
    inputs = _make_inputs(args.batch_size, args.seq_len, vocab, device)

    print("\n[2] BF16 eager...")
    t_bf16_e = _time_eager(model, inputs, warmup=args.warmup, iters=args.iters)
    r_bf16_e = _summarize("BF16 eager", t_bf16_e)
    _print_result(r_bf16_e, t_bf16_e)

    # Free parent GPU memory before subprocess graph attempt.
    del model
    del inputs
    torch.cuda.empty_cache()

    print("\n[3] BF16 CUDA graph capture + replay (subprocess)...")
    t_bf16_g, st_bf16 = _run_cudagraph_subprocess(
        {**graph_payload_base, "mode": "bf16"},
        label="BF16",
    )
    if t_bf16_g is None:
        r_bf16_g = BenchResult("BF16 cudagraph", 0.0, 0.0, ok=False, detail=st_bf16)
        _print_result(r_bf16_g)
    else:
        r_bf16_g = _summarize("BF16 cudagraph replay", t_bf16_g)
        _print_result(r_bf16_g, t_bf16_g, baseline=r_bf16_e.ms_mean)

    # ---- w4a4 IQ eager (parent, fresh model) ----
    print("\n[4] Insert w4a4 rollout path (fold W, disable WQ, keep IQ)...")
    model = _load_model(args.model_path, dtype, device)
    stats = _prepare_rollout_w4a4(model)
    print(
        f"  folded={stats['folded']} disabled_wq={stats['disabled_wq']} "
        f"wq_enabled={stats['wq_enabled']} iq_enabled={stats['iq_enabled']}"
    )
    if stats["iq_enabled"] <= 0 or stats["wq_enabled"] != 0:
        raise RuntimeError(f"unexpected quantizer state: {stats}")

    inputs = _make_inputs(args.batch_size, args.seq_len, vocab, device)

    print("\n[5] w4a4-IQ eager (伪量化)...")
    t_iq_e = _time_eager(model, inputs, warmup=args.warmup, iters=args.iters)
    r_iq_e = _summarize("w4a4-IQ eager", t_iq_e)
    _print_result(r_iq_e, t_iq_e, baseline=r_bf16_e.ms_mean)

    del model
    del inputs
    torch.cuda.empty_cache()

    print("\n[6] w4a4-IQ CUDA graph (伪量化 + graph, subprocess)...")
    t_iq_g, st_iq = _run_cudagraph_subprocess(
        {**graph_payload_base, "mode": "w4a4"},
        label="w4a4-IQ",
    )
    if t_iq_g is None:
        r_iq_g = BenchResult("w4a4-IQ cudagraph", 0.0, 0.0, ok=False, detail=st_iq)
        _print_result(r_iq_g)
    else:
        r_iq_g = _summarize("w4a4-IQ cudagraph replay", t_iq_g)
        _print_result(r_iq_g, t_iq_g, baseline=r_iq_e.ms_mean)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    _print_result(r_bf16_e, t_bf16_e)
    if t_bf16_g is not None:
        _print_result(r_bf16_g, t_bf16_g, baseline=r_bf16_e.ms_mean)
        print(
            f"  BF16 graph speedup vs eager: "
            f"{r_bf16_e.ms_mean / r_bf16_g.ms_mean:.2f}x "
            f"({r_bf16_e.ms_mean - r_bf16_g.ms_mean:+.2f} ms)"
        )
    else:
        print(f"  BF16 cudagraph: FAILED ({st_bf16})")

    _print_result(r_iq_e, t_iq_e, baseline=r_bf16_e.ms_mean)
    if t_iq_g is not None:
        _print_result(r_iq_g, t_iq_g, baseline=r_iq_e.ms_mean)
        print(
            f"  w4a4-IQ graph speedup vs its eager: "
            f"{r_iq_e.ms_mean / r_iq_g.ms_mean:.2f}x "
            f"({r_iq_e.ms_mean - r_iq_g.ms_mean:+.2f} ms)"
        )
        print(f"  w4a4-IQ graph vs BF16 eager: {r_iq_g.ms_mean / r_bf16_e.ms_mean:.2f}x")
        if t_bf16_g is not None:
            print(f"  w4a4-IQ graph vs BF16 graph: {r_iq_g.ms_mean / r_bf16_g.ms_mean:.2f}x")
    else:
        print(f"  w4a4-IQ cudagraph: FAILED ({st_iq})")
        if t_bf16_g is not None:
            print(
                "  => BF16 graph OK but IQ graph FAIL: fake-quant / IQ ops likely block "
                "CUDA graph (matches production enforce_eager for w4a4)."
            )
        else:
            print(
                "  => BF16 graph also failed on this HF-eager proxy; use IQ-eager vs BF16-eager "
                "here (or bench_rollout_qdq_overhead.py) for QDQ cost. Graph delta needs "
                "VeXact CudaGraphManager path to measure cleanly."
            )

    print(
        "\n  Note: HF eager attn + torch.cuda.CUDAGraph proxy, not VeXact CudaGraphManager. "
        "Relative ratios are the signal."
    )
    return 0


if __name__ == "__main__":
    # Required for CUDA + spawn on some platforms.
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
