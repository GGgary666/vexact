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

"""Tests for the VeOmni CPA forward_step wrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vexact.integrations.verl import fsdp_enable_cpa
from vexact.quantization.cpa import CPAConfig


class _FakeQuantizer(nn.Module):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self._enabled = enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True


class _FakeModule(nn.Module):
    def __init__(self, student_logits_unused=None):
        super().__init__()
        self.q = _FakeQuantizer(True)
        self.forward_calls = []
        self._teacher_log_probs = torch.tensor([[0.2, -0.5]], dtype=torch.float32)

    def forward(self, **kwargs):
        self.forward_calls.append(
            {
                "training": self.training,
                "q_enabled": self.q.is_enabled,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return SimpleNamespace(logits=torch.zeros(1, 1, 1))


class _FakeEngine:
    def __init__(self):
        self.module = _FakeModule()
        self.orig_calls = 0
        self.student_log_probs = torch.tensor([[0.0, -1.0]], dtype=torch.float32, requires_grad=True)

    def prepare_model_inputs(self, micro_batch):
        return {"input_ids": torch.ones(1, 2, dtype=torch.long)}, {}

    def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func=None):
        # Already response-shaped; identity converter used in tests.
        return {"log_probs": self.module._teacher_log_probs.clone()}


def _identity_to_response(log_probs, data):
    return log_probs


def _make_micro_batch(student_lp=None):
    response_mask = torch.tensor([[1.0, 1.0]])
    return {
        "response_mask": response_mask,
        "batch_num_tokens": 2.0,
        "dp_size": 1,
        "input_ids": torch.ones(1, 4, dtype=torch.long),
    }


def test_wrap_forward_step_adds_cpa_on_train_path():
    engine = _FakeEngine()
    cfg = CPAConfig(enable=True, coef=0.001)

    def orig(self, micro_batch, loss_function, forward_only):
        self.orig_calls += 1
        loss = self.student_log_probs.sum() * 0.0 + 1.0  # scalar with grad path via student
        # Attach student logprobs through a fake model_output that still has grad.
        model_output = {"log_probs": self.student_log_probs}
        metrics = {}
        output = {"model_output": model_output, "loss": float(loss.detach().item()), "metrics": metrics}
        return loss, output

    wrapped = fsdp_enable_cpa.wrap_forward_step(
        orig,
        cfg,
        to_response_log_probs=_identity_to_response,
        device_name="cpu",
        quantizer_type=_FakeQuantizer,
    )

    loss_fn = MagicMock()
    micro_batch = _make_micro_batch()
    total_loss, output = wrapped(engine, micro_batch, loss_fn, forward_only=False)

    assert engine.orig_calls == 1
    assert len(engine.module.forward_calls) == 1
    teacher_call = engine.module.forward_calls[0]
    assert teacher_call["training"] is False
    assert teacher_call["q_enabled"] is False
    assert teacher_call["grad_enabled"] is False
    # State restored after wrapper.
    assert engine.module.training is True
    assert engine.module.q.is_enabled is True

    assert "actor/cpa_loss" in output["metrics"]
    assert output["metrics"]["actor/cpa_coef"] == pytest.approx(0.001)
    assert "actor/cpa_logprob_gap_mean" in output["metrics"]
    assert abs(output["loss"] - float(total_loss.detach().item())) < 1e-6
    assert float(total_loss.detach().item()) != 1.0  # CPA term applied
    # Metrics must be numeric — verl reduce_metrics does np.mean on them.
    for key, val in output["metrics"].items():
        if key.startswith("actor/cpa_"):
            assert isinstance(val, (int, float)), f"{key}={val!r} must be numeric"


def test_wrap_forward_step_skips_forward_only_and_infer():
    engine = _FakeEngine()
    cfg = CPAConfig(enable=True, coef=0.001)
    calls = {"n": 0}

    def orig(self, micro_batch, loss_function, forward_only):
        calls["n"] += 1
        loss = torch.tensor(1.0)
        return loss, {"model_output": {"log_probs": self.student_log_probs}, "loss": 1.0, "metrics": {}}

    wrapped = fsdp_enable_cpa.wrap_forward_step(
        orig,
        cfg,
        to_response_log_probs=_identity_to_response,
        device_name="cpu",
        quantizer_type=_FakeQuantizer,
    )
    loss, output = wrapped(engine, _make_micro_batch(), None, forward_only=True)
    assert calls["n"] == 1
    assert len(engine.module.forward_calls) == 0
    assert "actor/cpa_loss" not in output["metrics"]
    assert float(loss.item()) == 1.0

    loss, output = wrapped(engine, _make_micro_batch(), MagicMock(), forward_only=True)
    assert len(engine.module.forward_calls) == 0


def test_wrap_forward_step_idempotent_install_marker():
    cfg = CPAConfig(enable=True, coef=0.001)

    def orig(self, micro_batch, loss_function, forward_only):
        return torch.tensor(1.0), {"model_output": {}, "loss": 1.0, "metrics": {}}

    w1 = fsdp_enable_cpa.wrap_forward_step(orig, cfg, to_response_log_probs=_identity_to_response)
    w2 = fsdp_enable_cpa.wrap_forward_step(w1, cfg, to_response_log_probs=_identity_to_response)
    assert w2 is w1
    assert getattr(w1, "_vexact_cpa_wrapped", False) is True


def test_call_prepare_model_outputs_passes_logits_processor_func_when_required():
    class EngineNew:
        def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
            assert logits_processor_func is None
            return {"log_probs": "ok-new", "logits_processor_func": logits_processor_func}

    class EngineOld:
        def prepare_model_outputs(self, output, output_args, micro_batch):
            return {"log_probs": "ok-old"}

    out_new = fsdp_enable_cpa._call_prepare_model_outputs(
        EngineNew(),
        raw_output=object(),
        output_args={},
        micro_batch={},
        logits_processor_func=None,
    )
    out_old = fsdp_enable_cpa._call_prepare_model_outputs(
        EngineOld(),
        raw_output=object(),
        output_args={},
        micro_batch={},
        logits_processor_func=None,
    )
    assert out_new["log_probs"] == "ok-new"
    assert out_old["log_probs"] == "ok-old"


def test_move_micro_batch_to_device_moves_tensors():
    batch = {
        "response_mask": torch.ones(1, 2),
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "batch_num_tokens": 2.0,
    }
    moved = fsdp_enable_cpa._move_micro_batch_to_device(batch, torch.device("cpu"))
    assert moved["response_mask"].device.type == "cpu"
    assert moved["batch_num_tokens"] == 2.0


def test_get_non_tensor_unwraps_nontensordata():
    """TensorDict.get returns NonTensorData; CPA must unwrap before int()/bool()."""
    pytest.importorskip("tensordict")
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData

    td = TensorDict({"response_mask": torch.ones(1, 2)}, batch_size=[1])
    td["dp_size"] = NonTensorData(2)
    td["batch_num_tokens"] = NonTensorData(4.0)

    assert fsdp_enable_cpa._get_non_tensor(td, "dp_size", default=1) == 2
    assert fsdp_enable_cpa._get_non_tensor(td, "batch_num_tokens", default=None) == 4.0
    # Must be plain Python ints/floats so `int(...)` / truthiness is safe.
    assert type(fsdp_enable_cpa._get_non_tensor(td, "dp_size", default=1)) is int
    assert type(fsdp_enable_cpa._get_non_tensor(td, "batch_num_tokens", default=None)) is float



def test_augment_moves_micro_batch_via_to_before_teacher(monkeypatch):
    """CPA must call micro_batch.to(device) because orig forward_step only rebinds locally."""
    engine = _FakeEngine()
    cfg = CPAConfig(enable=True, coef=0.001)
    seen = {"to_called_with": None, "teacher_got": None}

    class FakeTD(dict):
        def to(self, device):
            seen["to_called_with"] = device
            out = FakeTD({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.items()})
            return out

    def fake_teacher(eng, micro_batch, *, device_name, quantizer_type):
        seen["teacher_got"] = type(micro_batch).__name__
        return {"log_probs": torch.tensor([[0.2, -0.5]])}

    monkeypatch.setattr(fsdp_enable_cpa, "_run_teacher_forward", fake_teacher)

    loss = torch.tensor(1.0)
    output = {
        "model_output": {"log_probs": engine.student_log_probs},
        "loss": 1.0,
        "metrics": {},
    }
    micro_batch = FakeTD(_make_micro_batch())
    fsdp_enable_cpa.augment_train_loss_with_cpa(
        engine,
        micro_batch,
        loss,
        output,
        cfg,
        to_response_log_probs=_identity_to_response,
        device_name="cpu",
        quantizer_type=_FakeQuantizer,
    )
    assert seen["to_called_with"] is not None
    assert seen["teacher_got"] == "FakeTD"


def test_enable_training_cpa_patches_target(monkeypatch):
    cfg = CPAConfig(enable=True, coef=0.001)
    sentinel = object()

    class FakeVeOmni:
        def forward_step(self, micro_batch, loss_function, forward_only):
            return sentinel

    fake_mod = SimpleNamespace(VeOmniEngineWithLMHead=FakeVeOmni)
    import sys

    # Prefer exact module path lookup used by enable_training_cpa.
    monkeypatch.setitem(sys.modules, "verl.workers.engine.veomni.transformer_impl", fake_mod)

    assert fsdp_enable_cpa.enable_training_cpa(cfg) is True
    assert getattr(FakeVeOmni.forward_step, "_vexact_cpa_wrapped", False) is True
    # Second install is idempotent.
    assert fsdp_enable_cpa.enable_training_cpa(cfg) is True
