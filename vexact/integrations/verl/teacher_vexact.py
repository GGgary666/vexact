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

"""Pinned-VeRL compatibility shim for distillation teacher ``inference.name=vexact``.

Import via ``VERL_USE_EXTERNAL_MODULES`` **after** ``register`` / ``qaopd`` and
**before** the first Hydra instantiate of ``DistillationConfig``:

    export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register,\\
        vexact.integrations.verl.qaopd,\\
        vexact.integrations.verl.teacher_vexact

Installs:

1. ``DistillationTeacherModelConfig._validate_topk_logprobs`` — accept ``vexact``
2. Agent-loop teacher timing — expose ``timing_s/teacher_logprobs`` and
   ``timing_s/agent_loop/teacher_logprobs/{min,max,mean}`` without forking VeRL

Compatible with VeRL commit ``61f29997fb026a5a269dafccfe2f3bb800e32ef4``.
"""

from __future__ import annotations

import logging
from typing import Optional


logger = logging.getLogger(__name__)

_PINNED_VERL_HINT = "61f29997fb026a5a269dafccfe2f3bb800e32ef4"
_TOPK_SENTINEL = "_vexact_teacher_topk_shim"
_TIMING_SENTINEL = "_vexact_teacher_timing_shim"
_TOPK_INSTALLED = False
_TIMING_INSTALLED = False


def _install_validate_topk_logprobs_shim() -> None:
    global _TOPK_INSTALLED
    if _TOPK_INSTALLED:
        return

    try:
        from verl.workers.config.distillation import DistillationTeacherModelConfig
    except Exception as exc:  # pragma: no cover - import environment
        raise ImportError(
            "vexact.integrations.verl.teacher_vexact requires "
            f"verl.workers.config.distillation.DistillationTeacherModelConfig "
            f"(pinned VeRL {_PINNED_VERL_HINT}): {exc}"
        ) from exc

    if not hasattr(DistillationTeacherModelConfig, "_validate_topk_logprobs"):
        raise ImportError(
            "DistillationTeacherModelConfig._validate_topk_logprobs is missing; "
            f"teacher_vexact only supports pinned VeRL {_PINNED_VERL_HINT}"
        )

    original = DistillationTeacherModelConfig._validate_topk_logprobs
    if getattr(original, _TOPK_SENTINEL, False):
        _TOPK_INSTALLED = True
        return

    def _wrapped(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        if engine_name != "vexact":
            return original(self, use_topk=use_topk, topk=topk)

        if topk <= 0:
            raise ValueError(f"distillation_loss topk must be > 0 for vexact teacher, got {topk}")
        # Per-request prompt_logprobs carries K; no boot-time max_logprobs to set.
        return None

    setattr(_wrapped, _TOPK_SENTINEL, True)
    DistillationTeacherModelConfig._validate_topk_logprobs = _wrapped
    _TOPK_INSTALLED = True
    logger.info(
        "[vexact] Installed DistillationTeacherModelConfig._validate_topk_logprobs "
        "shim for inference.name=vexact (pinned VeRL %s)",
        _PINNED_VERL_HINT,
    )


def _ensure_teacher_logprobs_metrics_field(agent_loop_mod) -> None:
    """Add ``teacher_logprobs`` to ``AgentLoopMetrics`` if missing (pydantic v2)."""
    AgentLoopMetrics = agent_loop_mod.AgentLoopMetrics
    fields = getattr(AgentLoopMetrics, "model_fields", None)
    if fields is not None and "teacher_logprobs" in fields:
        return

    try:
        from pydantic import create_model
    except Exception:  # pragma: no cover
        logger.warning(
            "[vexact] Cannot extend AgentLoopMetrics with teacher_logprobs "
            "(pydantic create_model unavailable); teacher timing may be omitted."
        )
        return

    Patched = create_model(
        "AgentLoopMetrics",
        __base__=AgentLoopMetrics,
        teacher_logprobs=(float, 0.0),
    )
    agent_loop_mod.AgentLoopMetrics = Patched
    logger.info("[vexact] Extended AgentLoopMetrics with teacher_logprobs timing field")


def _install_teacher_timing_shim() -> None:
    """Wrap agent-loop teacher scoring so its wall time is reported in timing_s/*."""
    global _TIMING_INSTALLED
    if _TIMING_INSTALLED:
        return

    try:
        from verl.experimental.agent_loop import agent_loop as agent_loop_mod
    except Exception as exc:  # pragma: no cover - optional until distillation runs
        logger.warning(
            "[vexact] teacher timing shim skipped (agent_loop import failed): %s",
            exc,
        )
        return

    AgentLoopWorker = getattr(agent_loop_mod, "AgentLoopWorker", None)
    if AgentLoopWorker is None or not hasattr(AgentLoopWorker, "_compute_teacher_logprobs"):
        logger.warning(
            "[vexact] teacher timing shim skipped: AgentLoopWorker._compute_teacher_logprobs missing"
        )
        return

    if getattr(AgentLoopWorker._compute_teacher_logprobs, _TIMING_SENTINEL, False):
        _TIMING_INSTALLED = True
        return

    _ensure_teacher_logprobs_metrics_field(agent_loop_mod)

    try:
        from verl.utils.profiler.performance import simple_timer
    except Exception:  # pragma: no cover
        import contextlib
        import time

        @contextlib.contextmanager
        def simple_timer(name, timing_dict):
            t0 = time.perf_counter()
            yield
            timing_dict[name] = timing_dict.get(name, 0.0) + (time.perf_counter() - t0)

    original_compute = AgentLoopWorker._compute_teacher_logprobs
    # Some VeRL revisions expose teacher scoring but not `_performance_metrics`
    # (aggregation lives elsewhere or was renamed). Compute-path timing still
    # works; only the timing_s/agent_loop/* rollup is optional.
    original_perf = getattr(AgentLoopWorker, "_performance_metrics", None)

    async def _compute_teacher_logprobs_timed(self, output, *args, **kwargs):
        timing: dict[str, float] = {}
        with simple_timer("teacher_logprobs", timing):
            await original_compute(self, output, *args, **kwargs)
        elapsed = float(timing.get("teacher_logprobs", 0.0))
        metrics = getattr(output, "metrics", None)
        if metrics is not None:
            try:
                metrics.teacher_logprobs = elapsed
            except Exception:
                # Pydantic model without the field — best-effort via __dict__.
                try:
                    object.__setattr__(metrics, "teacher_logprobs", elapsed)
                except Exception:
                    pass

    setattr(_compute_teacher_logprobs_timed, _TIMING_SENTINEL, True)
    AgentLoopWorker._compute_teacher_logprobs = _compute_teacher_logprobs_timed

    if original_perf is None:
        _TIMING_INSTALLED = True
        logger.warning(
            "[vexact] AgentLoopWorker._performance_metrics missing; installed "
            "per-request teacher_logprobs timing only (no agent_loop/* rollup). "
            "Pinned VeRL hint: %s",
            _PINNED_VERL_HINT,
        )
        return

    def _performance_metrics_with_teacher(self, metrics, output):
        timing = original_perf(self, metrics, output)
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            return timing

        vals = []
        for chunk in metrics:
            for metric in chunk:
                if not isinstance(metric, dict):
                    continue
                vals.append(float(metric.get("teacher_logprobs", 0.0)))
        if not vals:
            return timing

        arr = np.asarray(vals, dtype=np.float64)
        timing["agent_loop/teacher_logprobs/min"] = float(arr.min())
        timing["agent_loop/teacher_logprobs/max"] = float(arr.max())
        timing["agent_loop/teacher_logprobs/mean"] = float(arr.mean())
        # Flat key → trainer ``timing_s/teacher_logprobs`` via compute_timing_metrics.
        timing["teacher_logprobs"] = float(arr.mean())

        # Recompute slowest including teacher wall time when available.
        try:
            t_gen = np.asarray(
                [float(m.get("generate_sequences", 0.0)) for chunk in metrics for m in chunk],
                dtype=np.float64,
            )
            t_tool = np.asarray(
                [float(m.get("tool_calls", 0.0)) for chunk in metrics for m in chunk],
                dtype=np.float64,
            )
            t_score = np.asarray(
                [float(m.get("compute_score", 0.0)) for chunk in metrics for m in chunk],
                dtype=np.float64,
            )
            if t_gen.shape == arr.shape:
                slowest = int(np.argmax(t_gen + t_tool + t_score + arr))
                timing["agent_loop/slowest/teacher_logprobs"] = float(arr[slowest])
        except Exception:
            pass
        return timing

    setattr(_performance_metrics_with_teacher, _TIMING_SENTINEL, True)
    AgentLoopWorker._performance_metrics = _performance_metrics_with_teacher
    _TIMING_INSTALLED = True
    logger.info(
        "[vexact] Installed agent-loop teacher timing shim "
        "(timing_s/teacher_logprobs, agent_loop/teacher_logprobs/*)"
    )


def install_teacher_vexact_shims() -> None:
    """Install top-k validation + teacher timing shims (idempotent)."""
    _install_validate_topk_logprobs_shim()
    _install_teacher_timing_shim()


install_teacher_vexact_shims()

print(f"[vexact] Teacher VeXact compatibility shim loaded at {__file__}")
