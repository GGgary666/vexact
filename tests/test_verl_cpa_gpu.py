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

"""GPU CPA smoke test against real verl padding / metric helpers.

Requires CUDA + an installed ``verl`` package. Skips otherwise.
Covers the production failure mode where student log_probs live on GPU while
the caller still holds a CPU nested ``TensorDict`` (offsets on CPU).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vexact.integrations.verl import fsdp_enable_cpa
from vexact.quantization.cpa import CPAConfig

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CPA GPU test requires CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("verl") is None, reason="CPA GPU test requires verl"),
]


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
    def __init__(self):
        super().__init__()
        self.q = _FakeQuantizer(True)

    def forward(self, **kwargs):
        # prepare_model_outputs builds teacher nested log_probs from offsets.
        return SimpleNamespace(logits=None)


class _RealVerlStyleEngine:
    """Minimal engine whose prepare_model_outputs hits nested_tensor_from_jagged."""

    def __init__(self, device: torch.device):
        self.module = _FakeModule().to(device)
        self.device = device

    def prepare_model_inputs(self, micro_batch):
        return {"input_ids": micro_batch["input_ids"]}, {}

    def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func=None):
        # Mirrors verl FSDPEngineWithLMHead NO_PADDING path: offsets come from
        # micro_batch["input_ids"], values from GPU compute.
        assert logits_processor_func is None
        offsets = micro_batch["input_ids"].offsets()
        total_nnz = int(offsets[-1].item())
        values = torch.randn(total_nnz, device=self.device, dtype=torch.float32)
        log_probs = torch.nested.nested_tensor_from_jagged(values, offsets)
        return {"log_probs": log_probs}


def _build_cpu_nested_micro_batch():
    """Build a CPU nested TensorDict shaped like verl NO_PADDING actor batches."""
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    prompt_lens = [3, 2]
    response_lens = [2, 3]
    max_response_len = max(response_lens)
    batch_size = len(prompt_lens)

    prompt_list = [torch.arange(1, pl + 1, dtype=torch.long) for pl in prompt_lens]
    response_list = [torch.arange(100, 100 + rl, dtype=torch.long) for rl in response_lens]
    full_list = [torch.cat([p, r], dim=0) for p, r in zip(prompt_list, response_list, strict=True)]

    prompts = torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged)
    responses = torch.nested.as_nested_tensor(response_list, layout=torch.jagged)
    input_ids = torch.nested.as_nested_tensor(full_list, layout=torch.jagged)

    response_mask = torch.zeros(batch_size, max_response_len, dtype=torch.float32)
    for i, rl in enumerate(response_lens):
        response_mask[i, :rl] = 1.0

    # attention_mask kept for API completeness; nested path uses prompts/responses.
    attention_mask = torch.ones(batch_size, max(prompt_lens) + max_response_len)

    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "attention_mask": attention_mask,
        },
        batch_size=[batch_size],
    )
    tu.assign_non_tensor_data(data, "max_response_len", max_response_len)
    tu.assign_non_tensor_data(data, "batch_num_tokens", float(response_mask.sum().item()))
    tu.assign_non_tensor_data(data, "dp_size", 1)
    return data, prompt_lens, response_lens


def _nested_log_probs_like(input_ids_nested: torch.Tensor, device: torch.device, requires_grad: bool):
    offsets = input_ids_nested.offsets().to(device=device)
    total_nnz = int(offsets[-1].item())
    values = torch.randn(total_nnz, device=device, dtype=torch.float32, requires_grad=requires_grad)
    return torch.nested.nested_tensor_from_jagged(values, offsets)


def test_cpa_gpu_real_verl_nested_batch_device_align():
    """CPA must move CPU nested micro_batch onto CUDA before teacher jagged rebuild."""
    from verl.utils.device import get_device_name
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    device = torch.device("cuda:0")
    micro_batch_cpu, _, _ = _build_cpu_nested_micro_batch()
    assert micro_batch_cpu["input_ids"].offsets().device.type == "cpu"

    # Student log_probs already on GPU (as after verl forward_step), while the
    # caller's micro_batch reference is still the CPU TensorDict.
    student_raw = _nested_log_probs_like(micro_batch_cpu["input_ids"], device, requires_grad=True)

    engine = _RealVerlStyleEngine(device)
    cfg = CPAConfig(enable=True, coef=0.001)
    loss = torch.tensor(1.0, device=device, requires_grad=True)
    output = {
        "model_output": {"log_probs": student_raw},
        "loss": 1.0,
        "metrics": {
            "actor/pg_loss": Metric(aggregation=AggregationType.MEAN, value=0.5),
        },
    }

    total_loss, out = fsdp_enable_cpa.augment_train_loss_with_cpa(
        engine,
        micro_batch_cpu,
        loss,
        output,
        cfg,
        to_response_log_probs=no_padding_2_padding,
        device_name=get_device_name(),
        quantizer_type=_FakeQuantizer,
    )

    assert "actor/cpa_loss" in out["metrics"]
    assert "actor/cpa_logprob_gap_mean" in out["metrics"]
    assert out["metrics"]["actor/cpa_coef"] == pytest.approx(0.001)
    assert isinstance(out["metrics"]["actor/cpa_loss"], Metric)
    assert float(total_loss.detach().item()) != 1.0
    assert total_loss.requires_grad


def test_cpa_gpu_without_move_hits_device_mismatch():
    """Regression: nested_tensor_from_jagged asserts when offsets stay on CPU."""
    device = torch.device("cuda:0")
    micro_batch_cpu, _, _ = _build_cpu_nested_micro_batch()
    engine = _RealVerlStyleEngine(device)

    with pytest.raises(Exception):
        engine.prepare_model_outputs(
            output=SimpleNamespace(logits=None),
            output_args={},
            micro_batch=micro_batch_cpu,  # CPU offsets
            logits_processor_func=None,
        )
