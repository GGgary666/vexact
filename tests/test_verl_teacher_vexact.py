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

"""Tests for VeRL teacher_vexact top-k config shim and replica naming helpers."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Optional

import pytest


def _install_fake_distillation_teacher(monkeypatch, *, original_impl=None):
    """Minimal DistillationTeacherModelConfig with pinned-style match."""

    def _default_validate(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")
        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )

    class DistillationTeacherModelConfig:
        def __init__(self, name: str, engine_kwargs: Optional[dict] = None):
            self.inference = SimpleNamespace(name=name, engine_kwargs=engine_kwargs or {})

        _validate_topk_logprobs = original_impl or _default_validate

    distill_cfg = types.ModuleType("verl.workers.config.distillation")
    distill_cfg.DistillationTeacherModelConfig = DistillationTeacherModelConfig

    workers_cfg = types.ModuleType("verl.workers.config")
    workers_cfg.distillation = distill_cfg
    workers_pkg = types.ModuleType("verl.workers")
    workers_pkg.config = workers_cfg
    verl_pkg = types.ModuleType("verl")
    verl_pkg.workers = workers_pkg

    modules = {
        "verl": verl_pkg,
        "verl.workers": workers_pkg,
        "verl.workers.config": workers_cfg,
        "verl.workers.config.distillation": distill_cfg,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return DistillationTeacherModelConfig


def _reload_teacher_vexact():
    sys.modules.pop("vexact.integrations.verl.teacher_vexact", None)
    import vexact.integrations.verl.teacher_vexact as mod

    importlib.reload(mod)
    return mod


def test_validate_topk_accepts_vexact(monkeypatch):
    DistillationTeacherModelConfig = _install_fake_distillation_teacher(monkeypatch)
    _reload_teacher_vexact()

    cfg = DistillationTeacherModelConfig("vexact")
    cfg._validate_topk_logprobs(use_topk=True, topk=64)
    assert "vllm" not in cfg.inference.engine_kwargs


def test_validate_topk_vllm_unchanged(monkeypatch):
    DistillationTeacherModelConfig = _install_fake_distillation_teacher(monkeypatch)
    _reload_teacher_vexact()

    cfg = DistillationTeacherModelConfig("vllm", engine_kwargs={})
    cfg._validate_topk_logprobs(use_topk=True, topk=32)
    assert cfg.inference.engine_kwargs["vllm"]["max_logprobs"] == 32


def test_validate_topk_sglang_noop(monkeypatch):
    DistillationTeacherModelConfig = _install_fake_distillation_teacher(monkeypatch)
    _reload_teacher_vexact()

    cfg = DistillationTeacherModelConfig("sglang")
    cfg._validate_topk_logprobs(use_topk=True, topk=16)
    assert cfg.inference.engine_kwargs == {}


def test_validate_topk_unknown_engine_still_raises(monkeypatch):
    DistillationTeacherModelConfig = _install_fake_distillation_teacher(monkeypatch)
    _reload_teacher_vexact()

    cfg = DistillationTeacherModelConfig("other")
    with pytest.raises(NotImplementedError, match="other"):
        cfg._validate_topk_logprobs(use_topk=True, topk=8)


def test_validate_topk_rejects_nonpositive_k(monkeypatch):
    DistillationTeacherModelConfig = _install_fake_distillation_teacher(monkeypatch)
    _reload_teacher_vexact()

    cfg = DistillationTeacherModelConfig("vexact")
    with pytest.raises(ValueError, match="topk"):
        cfg._validate_topk_logprobs(use_topk=True, topk=0)


def test_shim_idempotent(monkeypatch):
    DistillationTeacherModelConfig = _install_fake_distillation_teacher(monkeypatch)
    mod = _reload_teacher_vexact()
    first = DistillationTeacherModelConfig._validate_topk_logprobs
    importlib.reload(mod)
    second = DistillationTeacherModelConfig._validate_topk_logprobs
    assert first is second
    assert getattr(first, "_vexact_teacher_topk_shim", False)


def _install_fake_agent_loop(monkeypatch):
    """Minimal AgentLoopWorker surface for teacher timing shim."""
    from pydantic import BaseModel

    class AgentLoopMetrics(BaseModel):
        generate_sequences: float = 0.0
        tool_calls: float = 0.0
        compute_score: float = 0.0
        num_preempted: int = -1

    class AgentLoopWorker:
        async def _compute_teacher_logprobs(self, output, *args, **kwargs):
            return None

        def _performance_metrics(self, metrics, output):
            return {
                "agent_loop/generate_sequences/mean": 1.0,
                "agent_loop/tool_calls/mean": 0.0,
                "agent_loop/compute_score/mean": 0.0,
            }

    agent_loop = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop.AgentLoopMetrics = AgentLoopMetrics
    agent_loop.AgentLoopWorker = AgentLoopWorker

    agent_pkg = types.ModuleType("verl.experimental.agent_loop")
    agent_pkg.agent_loop = agent_loop
    exp_pkg = types.ModuleType("verl.experimental")
    exp_pkg.agent_loop = agent_pkg
    verl_pkg = sys.modules.get("verl") or types.ModuleType("verl")
    verl_pkg.experimental = exp_pkg

    modules = {
        "verl": verl_pkg,
        "verl.experimental": exp_pkg,
        "verl.experimental.agent_loop": agent_pkg,
        "verl.experimental.agent_loop.agent_loop": agent_loop,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return agent_loop


def test_teacher_timing_shim_records_metrics(monkeypatch):
    import asyncio

    _install_fake_distillation_teacher(monkeypatch)
    agent_loop = _install_fake_agent_loop(monkeypatch)
    mod = _reload_teacher_vexact()

    Worker = agent_loop.AgentLoopWorker
    assert getattr(Worker._compute_teacher_logprobs, "_vexact_teacher_timing_shim", False)

    # create_model should have added teacher_logprobs
    Metrics = agent_loop.AgentLoopMetrics
    assert "teacher_logprobs" in Metrics.model_fields

    output = SimpleNamespace(metrics=Metrics())
    worker = Worker()
    asyncio.run(worker._compute_teacher_logprobs(output, [], [], False, None))
    assert output.metrics.teacher_logprobs >= 0.0

    timing = worker._performance_metrics(
        [[{"generate_sequences": 2.0, "tool_calls": 0.0, "compute_score": 0.1, "teacher_logprobs": 1.5}]],
        SimpleNamespace(),
    )
    assert timing["teacher_logprobs"] == pytest.approx(1.5)
    assert timing["agent_loop/teacher_logprobs/mean"] == pytest.approx(1.5)
    assert mod is not None


def test_teacher_timing_shim_without_performance_metrics(monkeypatch):
    """VeRL revisions may lack `_performance_metrics`; shim must still import."""
    import asyncio

    from pydantic import BaseModel

    _install_fake_distillation_teacher(monkeypatch)

    class AgentLoopMetrics(BaseModel):
        generate_sequences: float = 0.0

    class AgentLoopWorker:
        async def _compute_teacher_logprobs(self, output, *args, **kwargs):
            return None

    agent_loop = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop.AgentLoopMetrics = AgentLoopMetrics
    agent_loop.AgentLoopWorker = AgentLoopWorker
    agent_pkg = types.ModuleType("verl.experimental.agent_loop")
    agent_pkg.agent_loop = agent_loop
    exp_pkg = types.ModuleType("verl.experimental")
    exp_pkg.agent_loop = agent_pkg
    verl_pkg = sys.modules.get("verl") or types.ModuleType("verl")
    verl_pkg.experimental = exp_pkg
    for name, mod in {
        "verl": verl_pkg,
        "verl.experimental": exp_pkg,
        "verl.experimental.agent_loop": agent_pkg,
        "verl.experimental.agent_loop.agent_loop": agent_loop,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    _reload_teacher_vexact()
    Worker = agent_loop.AgentLoopWorker
    assert getattr(Worker._compute_teacher_logprobs, "_vexact_teacher_timing_shim", False)
    assert not hasattr(Worker, "_performance_metrics")

    output = SimpleNamespace(metrics=agent_loop.AgentLoopMetrics())
    asyncio.run(Worker()._compute_teacher_logprobs(output))
    assert output.metrics.teacher_logprobs >= 0.0


def test_build_role_qualified_names():
    from vexact.integrations.verl.role_names import build_vexact_role_names

    student = build_vexact_role_names(
        replica_rank=0, node_rank=0, is_teacher_model=False, is_reward_model=False, name_suffix=""
    )
    teacher = build_vexact_role_names(
        replica_rank=0, node_rank=0, is_teacher_model=True, is_reward_model=False, name_suffix="_teacher_model"
    )
    teacher2 = build_vexact_role_names(
        replica_rank=1, node_rank=0, is_teacher_model=True, is_reward_model=False, name_suffix="_teacher_model"
    )
    assert student["server_name"] != teacher["server_name"]
    assert teacher["server_name"] != teacher2["server_name"]
    assert student["driver_id"] != teacher["driver_id"]
    assert "teacher" in teacher["server_name"]
    assert teacher["driver_id"].startswith("vexact_teacher")
    assert student["server_name"] == "vexact_server_student_0_0"
    assert student["driver_id"] == "vexact_student_replica_0"


def test_placement_worker_class_is_not_server_adapter():
    from vexact.integrations.verl.placement_worker import VeXactPlacementWorker

    worker = VeXactPlacementWorker(config=None, model_config=None, device_mesh=None)
    assert worker is not None
