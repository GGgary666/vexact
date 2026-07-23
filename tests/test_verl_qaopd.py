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

"""Tests for VeRL QAOPD registration and top-k dispatcher wrapping."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from vexact.quantization.qaopd import QAOPD_LOSS_MODE, compute_qaopd_topk_loss


def _install_fake_verl_distillation(monkeypatch):
    """Minimal verl.trainer.distillation.losses stub for registration tests."""
    settings_registry = {}
    loss_registry = {}

    class DistillationLossSettings:
        def __init__(self, names, use_topk=False, use_estimator=False):
            self.names = [names] if isinstance(names, str) else list(names)
            self.use_topk = use_topk
            self.use_estimator = use_estimator
            if sum([use_topk, use_estimator]) != 1:
                raise ValueError("Expected only one of use_estimator, use_topk")

    def register_distillation_loss(loss_settings):
        def decorator(func):
            for name in loss_settings.names:
                if name in loss_registry:
                    raise ValueError(f"already registered: {name}")
                loss_registry[name] = func
                settings_registry[name] = loss_settings
            return func

        return decorator

    def get_distillation_loss_settings(name):
        return settings_registry[name]

    def get_distillation_loss_fn(name):
        return loss_registry[name]

    def compute_distillation_loss_range(distillation_losses, response_mask):
        return {}

    def compute_topk_loss(config, distillation_config, data, student_logits, data_format):
        return {
            "distillation_losses": torch.zeros(student_logits.shape[:2]),
            "student_mass": torch.zeros(student_logits.shape[:2]),
            "teacher_mass": torch.zeros(student_logits.shape[:2]),
        }

    losses_mod = types.ModuleType("verl.trainer.distillation.losses")
    losses_mod.DistillationLossSettings = DistillationLossSettings
    losses_mod.register_distillation_loss = register_distillation_loss
    losses_mod.get_distillation_loss_settings = get_distillation_loss_settings
    losses_mod.get_distillation_loss_fn = get_distillation_loss_fn
    losses_mod.compute_distillation_loss_range = compute_distillation_loss_range
    losses_mod.compute_topk_loss = compute_topk_loss
    losses_mod.DISTILLATION_LOSS_REGISTRY = loss_registry
    losses_mod.DISTILLATION_SETTINGS_REGISTRY = settings_registry

    distill_pkg = types.ModuleType("verl.trainer.distillation")
    distill_pkg.losses = losses_mod

    trainer_pkg = types.ModuleType("verl.trainer")
    verl_pkg = types.ModuleType("verl")

    # padding + metric helpers used by registered final loss
    padding_mod = types.ModuleType("verl.workers.utils.padding")

    def no_padding_2_padding(x, data):
        return x

    padding_mod.no_padding_2_padding = no_padding_2_padding

    metric_mod = types.ModuleType("verl.utils.metric")

    class AggregationType:
        MIN = "min"
        MAX = "max"
        MEAN = "mean"
        SUM = "sum"

    class Metric:
        def __init__(self, aggregation=None, value=None):
            self.aggregation = aggregation
            self.value = value

    metric_mod.AggregationType = AggregationType
    metric_mod.Metric = Metric

    ulysses_mod = types.ModuleType("verl.utils.ulysses")
    ulysses_mod.get_ulysses_sequence_parallel_world_size = lambda: 1
    ulysses_mod.slice_input_tensor = lambda x, dim=1: x

    workers_utils = types.ModuleType("verl.workers.utils")
    workers_pkg = types.ModuleType("verl.workers")
    utils_pkg = types.ModuleType("verl.utils")

    modules = {
        "verl": verl_pkg,
        "verl.trainer": trainer_pkg,
        "verl.trainer.distillation": distill_pkg,
        "verl.trainer.distillation.losses": losses_mod,
        "verl.workers": workers_pkg,
        "verl.workers.utils": workers_utils,
        "verl.workers.utils.padding": padding_mod,
        "verl.utils": utils_pkg,
        "verl.utils.metric": metric_mod,
        "verl.utils.ulysses": ulysses_mod,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return losses_mod, loss_registry, settings_registry


def test_register_qaopd_loss_idempotent(monkeypatch):
    losses_mod, loss_registry, settings_registry = _install_fake_verl_distillation(monkeypatch)
    # Force re-import of qaopd integration against the stub.
    sys.modules.pop("vexact.integrations.verl.qaopd", None)
    qaopd = importlib.import_module("vexact.integrations.verl.qaopd")
    # Reset module globals for a clean register call.
    qaopd._REGISTERED = False
    qaopd._TOPK_PATCHED = False
    assert qaopd.register_vexact_qaopd_losses(validate=False) is True
    assert QAOPD_LOSS_MODE in loss_registry
    assert settings_registry[QAOPD_LOSS_MODE].use_topk is True
    # Idempotent
    assert qaopd.register_vexact_qaopd_losses(validate=False) is True
    assert getattr(losses_mod.compute_topk_loss, "_vexact_qaopd_wrapped", False) is True


def test_wrapped_topk_dispatches_qaopd_vs_builtin(monkeypatch):
    losses_mod, _, _ = _install_fake_verl_distillation(monkeypatch)
    sys.modules.pop("vexact.integrations.verl.qaopd", None)
    qaopd = importlib.import_module("vexact.integrations.verl.qaopd")
    qaopd._REGISTERED = False
    qaopd._TOPK_PATCHED = False
    qaopd.register_vexact_qaopd_losses(validate=False)

    torch.manual_seed(0)
    bsz, seqlen, vocab, k = 1, 4, 16, 4
    student = torch.randn(bsz, seqlen, vocab)
    teacher = torch.randn(bsz, seqlen, vocab)
    vals, ids = torch.topk(teacher, k=k, dim=-1)
    t_logp = F.log_softmax(teacher, dim=-1).gather(-1, ids)

    # Nested teacher tensors as in VeRL.
    teacher_logprobs = torch.nested.nested_tensor(list(t_logp), layout=torch.jagged)
    teacher_ids = torch.nested.nested_tensor(list(ids), layout=torch.jagged)
    # nested_tensor from list of [S,K] gives batch; values path expects jagged from
    # padded batch. Use a simple fake with .is_nested / .values().
    class _Nested:
        def __init__(self, dense):
            self._dense = dense

        @property
        def is_nested(self):
            return True

        def values(self):
            # Flatten batch into (total_nnz, k) as VeRL does for remove-padding path.
            return self._dense.reshape(-1, self._dense.shape[-1])

    data = {
        "teacher_logprobs": _Nested(t_logp),
        "teacher_ids": _Nested(ids),
    }
    # Student logits in rmpad path are (1, total_nnz, V)
    student_rmpad = student.reshape(1, -1, vocab)

    qaopd_cfg = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode=QAOPD_LOSS_MODE,
            topk=k,
            log_prob_min_clamp=None,
        )
    )
    builtin_cfg = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode="forward_kl_topk",
            topk=k,
            log_prob_min_clamp=None,
        )
    )
    actor_cfg = SimpleNamespace(strategy="veomni")

    out_qaopd = losses_mod.compute_topk_loss(
        actor_cfg, qaopd_cfg, data, student_rmpad, "thd"
    )
    expected = compute_qaopd_topk_loss(
        student_rmpad,
        t_logp.reshape(1, -1, k),
        ids.reshape(1, -1, k),
        kl_type="mixed",
        mixed_kl_weight=0.5,
    )
    assert torch.allclose(
        out_qaopd["distillation_losses"], expected["distillation_losses"], atol=1e-5
    )

    out_builtin = losses_mod.compute_topk_loss(
        actor_cfg, builtin_cfg, data, student_rmpad, "thd"
    )
    # Builtin stub returns zeros.
    assert torch.count_nonzero(out_builtin["distillation_losses"]) == 0


def test_validate_rejects_cpa(monkeypatch):
    losses_mod, _, _ = _install_fake_verl_distillation(monkeypatch)
    sys.modules.pop("vexact.integrations.verl.qaopd", None)
    qaopd = importlib.import_module("vexact.integrations.verl.qaopd")
    monkeypatch.setenv("VEXACT_CPA_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a4")
    with pytest.raises(RuntimeError, match="CPA"):
        qaopd.validate_qaopd_runtime_guards()


def test_qaopd_guard_rejects_policy_gradient(monkeypatch):
    _install_fake_verl_distillation(monkeypatch)
    sys.modules.pop("vexact.integrations.verl.qaopd", None)
    qaopd = importlib.import_module("vexact.integrations.verl.qaopd")
    monkeypatch.setenv("VEXACT_QAOPD_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a4")
    monkeypatch.delenv("VEXACT_CPA_ENABLE", raising=False)
    monkeypatch.setenv("VEXACT_QAOPD_USE_POLICY_GRADIENT", "1")
    with pytest.raises(RuntimeError, match="use_policy_gradient"):
        qaopd.validate_qaopd_runtime_guards(require_qat=True, reject_cpa=True)


def test_validate_requires_w4a4_qat(monkeypatch):
    losses_mod, _, _ = _install_fake_verl_distillation(monkeypatch)
    sys.modules.pop("vexact.integrations.verl.qaopd", None)
    qaopd = importlib.import_module("vexact.integrations.verl.qaopd")
    monkeypatch.delenv("VEXACT_CPA_ENABLE", raising=False)
    monkeypatch.delenv("VEXACT_QAT_ENABLE", raising=False)
    with pytest.raises(RuntimeError, match="VEXACT_QAT_ENABLE"):
        qaopd.validate_qaopd_runtime_guards()
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a16")
    with pytest.raises(RuntimeError, match="w4a4"):
        qaopd.validate_qaopd_runtime_guards()


def test_enable_qaopd_runs_guards_when_flag_set(monkeypatch):
    losses_mod, _, _ = _install_fake_verl_distillation(monkeypatch)
    sys.modules.pop("vexact.integrations.verl.qaopd", None)
    qaopd = importlib.import_module("vexact.integrations.verl.qaopd")
    qaopd._REGISTERED = False
    qaopd._TOPK_PATCHED = False
    monkeypatch.setenv("VEXACT_QAOPD_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a4")
    monkeypatch.delenv("VEXACT_CPA_ENABLE", raising=False)
    assert qaopd.enable_qaopd() is True
