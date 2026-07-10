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

import importlib.util
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vexact.quantization import QATConfig
from vexact.quantization.fold import (
    copy_weight_into_param,
    disable_weight_quantizers,
    fold_weight,
    fold_weights_generator,
    get_weight_quantizer_map,
    load_weights_with_optional_prefold,
)

_HAS_MODELOPT = importlib.util.find_spec("modelopt") is not None


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))


def test_effective_prefold_weights_always_on_when_qat_enabled():
    assert QATConfig(enable=True, mode="w4a16").effective_prefold_weights is True
    assert QATConfig(enable=True, mode="w4a4").effective_prefold_weights is True
    assert QATConfig(enable=True, mode="w4a16", prefold_weights=False).effective_prefold_weights is True
    assert QATConfig(enable=True, mode="w4a16", prefold_weights=True).effective_prefold_weights is True
    assert QATConfig(enable=True, mode="w4a4", prefold_weights=True).effective_prefold_weights is True


def test_fold_weight_uses_quantizer():
    wq = MagicMock(side_effect=lambda x: x + 1)
    wq.is_enabled = True
    wq.to = MagicMock(return_value=None)
    weight = torch.zeros(2, 2)
    folded = fold_weight(weight, wq)
    wq.assert_called_once()
    assert folded.shape == weight.shape


def test_fold_weight_uses_fp32_input_for_quantizer_and_restores_param_dtype():
    seen_dtypes = []

    class DTypeCheckingQuantizer:
        is_enabled = True

        def to(self, device):
            return self

        def __call__(self, x):
            seen_dtypes.append(x.dtype)
            return x + 1

    weight = torch.zeros(2, 2, dtype=torch.bfloat16)
    folded = fold_weight(weight, DTypeCheckingQuantizer())

    assert seen_dtypes == [torch.float32]
    assert folded.dtype == torch.bfloat16


def test_copy_weight_into_param_prefold():
    param = nn.Parameter(torch.zeros(2, 2))
    loaded = torch.full((2, 2), 3.0)
    wq = MagicMock(side_effect=lambda x: x * 2)
    wq.is_enabled = True
    wq.to = MagicMock(return_value=None)

    copy_weight_into_param(param, loaded, wq, prefold=True)
    assert torch.all(param.data == 6.0)


def test_copy_weight_into_param_raw():
    param = nn.Parameter(torch.zeros(2, 2))
    loaded = torch.full((2, 2), 3.0)

    copy_weight_into_param(param, loaded, prefold=False)
    assert torch.all(param.data == 3.0)


def test_fold_weight_works_when_quantizer_disabled():
    wq = MagicMock()
    wq.is_enabled = False
    wq.enable = MagicMock(side_effect=lambda: setattr(wq, "is_enabled", True))
    wq.disable = MagicMock(side_effect=lambda: setattr(wq, "is_enabled", False))
    wq.to = MagicMock(return_value=None)
    wq.side_effect = lambda x: x + 1

    weight = torch.zeros(2, 2)
    folded = fold_weight(weight, wq)
    wq.enable.assert_called_once()
    wq.disable.assert_called_once()
    assert folded.shape == weight.shape


def test_get_weight_quantizer_map_from_wrapper():
    inner = nn.Module()
    inner._vexact_weight_quantizer_map = {"a": "wq"}
    wrapper = nn.Module()
    wrapper.model = inner

    assert get_weight_quantizer_map(wrapper) == {"a": "wq"}


def test_load_weights_with_optional_prefold():
    model = _Linear()
    wq = MagicMock(side_effect=lambda x: x + 0.5)
    wq.is_enabled = True
    wq.to = MagicMock(return_value=None)
    wq_map = {"weight": wq}

    load_weights_with_optional_prefold(
        model,
        [("weight", torch.full((2, 2), 1.0))],
        weight_quantizer_map=wq_map,
    )
    assert torch.all(model.weight.data == 1.5)


def test_training_side_fold_casts_fp32_sender_weight_to_bf16_by_default():
    """FSDP full_tensor is often fp32; fold must match live bf16 Parameters."""

    class IdentityQuantizer(nn.Module):
        is_enabled = True

        def enable(self):
            self.is_enabled = True

        def disable(self):
            self.is_enabled = False

        def forward(self, value):
            return value

    layer_name = "model.layers.0.self_attn.q_proj.weight"
    fp32_weight = torch.randn(2, 2, dtype=torch.float32)
    quantizer = IdentityQuantizer()

    out = dict(
        fold_weights_generator(
            [(layer_name, fp32_weight)],
            {layer_name: quantizer},
        )
    )
    folded = out[layer_name]
    assert folded.dtype == torch.bfloat16
    expected = fold_weight(fp32_weight.to(torch.bfloat16), quantizer)
    assert torch.equal(folded, expected)


def test_training_side_fold_keeps_bf16_input_dtype():
    class IdentityQuantizer(nn.Module):
        is_enabled = True

        def enable(self):
            self.is_enabled = True

        def disable(self):
            self.is_enabled = False

        def forward(self, value):
            return value

    layer_name = "model.layers.0.self_attn.q_proj.weight"
    bf16_weight = torch.ones(2, 2, dtype=torch.bfloat16)
    quantizer = IdentityQuantizer()

    out = dict(
        fold_weights_generator(
            [(layer_name, bf16_weight)],
            {layer_name: quantizer},
        )
    )
    assert out[layer_name].dtype == torch.bfloat16


@pytest.mark.skipif(not _HAS_MODELOPT, reason="modelopt not installed")
def test_disable_weight_quantizers_noop_without_quantizers():
    model = _Linear()
    # No TensorQuantizer children; should return 0 without error when modelopt present.
    # build path requires QuantModule; plain Linear just counts 0.
    from vexact.quantization.fold import count_enabled_quantizers

    assert count_enabled_quantizers(model, name_suffix="weight_quantizer") == 0
