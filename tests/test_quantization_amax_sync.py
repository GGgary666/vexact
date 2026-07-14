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

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vexact.quantization import QATConfig
from vexact.quantization.amax_sync import (
    apply_input_amax_buffers,
    chain_weights_with_input_amax,
    fill_input_amax_sentinel,
    input_amax_buffer_name,
    is_input_amax_key,
    iter_input_amax_buffers,
    split_weights_and_input_amax,
)


def test_w4a4_defaults_calibrate_true():
    cfg = QATConfig(enable=True, mode="w4a4")
    assert cfg.calibrate is True
    assert cfg.effective_calibrate is True


def test_w4a16_defaults_calibrate_false():
    cfg = QATConfig(enable=True, mode="w4a16")
    assert cfg.calibrate is False
    assert cfg.effective_calibrate is False


def test_w4a4_calibrate_false_raises():
    with pytest.raises(ValueError, match="requires calibrate=True"):
        QATConfig(enable=True, mode="w4a4", calibrate=False)


def test_w4a4_calibrate_false_ok_when_disabled():
    cfg = QATConfig(enable=False, mode="w4a4", calibrate=False)
    assert cfg.calibrate is False


def test_w4a16_calibrate_true_allowed():
    cfg = QATConfig(enable=True, mode="w4a16", calibrate=True)
    assert cfg.calibrate is True


def test_from_env_w4a4_calibrate_false_raises(monkeypatch):
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a4")
    monkeypatch.setenv("VEXACT_QAT_CALIBRATE", "0")
    with pytest.raises(ValueError, match="requires calibrate=True"):
        QATConfig.from_env()


def test_from_env_w4a16_calibrate_false_ok(monkeypatch):
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a16")
    monkeypatch.setenv("VEXACT_QAT_CALIBRATE", "0")
    cfg = QATConfig.from_env()
    assert cfg.mode == "w4a16"
    assert cfg.calibrate is False


def test_is_input_amax_key():
    assert is_input_amax_key("model.layers.0.mlp.down_proj.input_quantizer._amax")
    assert is_input_amax_key("foo.input_quantizer.amax")
    assert not is_input_amax_key("model.layers.0.mlp.down_proj.weight")
    assert not is_input_amax_key("model.layers.0.mlp.down_proj.weight_quantizer._amax")


def test_iter_input_amax_buffers_raises_on_missing_amax():
    iq = MagicMock()
    iq.amax = None
    with pytest.raises(RuntimeError, match="missing amax"):
        list(iter_input_amax_buffers({"layer.input_quantizer": iq}))


def test_iter_and_chain_input_amax_buffers():
    amax = torch.tensor([1.5])
    iq = MagicMock()
    iq.amax = amax
    items = list(iter_input_amax_buffers({"mod.input_quantizer": iq}))
    assert items == [("mod.input_quantizer._amax", amax)]

    weights = [("a.weight", torch.ones(2))]
    chained = list(chain_weights_with_input_amax(weights, {"mod.input_quantizer": iq}))
    assert chained[0][0] == "a.weight"
    assert chained[1][0] == "mod.input_quantizer._amax"


def test_split_weights_and_input_amax():
    items = [
        ("a.weight", torch.ones(2)),
        ("m.input_quantizer._amax", torch.tensor(3.0)),
        ("b.bias", torch.zeros(1)),
    ]
    weights, amaxes = split_weights_and_input_amax(items)
    assert [n for n, _ in weights] == ["a.weight", "b.bias"]
    assert [n for n, _ in amaxes] == ["m.input_quantizer._amax"]


def test_apply_input_amax_buffers_max_merge():
    class _IQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_amax", torch.tensor([1.0, 4.0]))

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_quantizer = _IQ()

    model = nn.Sequential(_Block())
    # named_buffers: "0.input_quantizer._amax"
    name = "0.input_quantizer._amax"
    assert name in dict(model.named_buffers())

    updated = apply_input_amax_buffers(
        model,
        [(name, torch.tensor([3.0, 2.0]))],
    )
    assert updated == 1
    buf = dict(model.named_buffers())[name]
    assert torch.equal(buf, torch.tensor([3.0, 4.0]))


def test_load_input_amax_exact_overwrites_dummy():
    from vexact.quantization.amax_sync import load_input_amax_exact

    class _IQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_amax", torch.tensor([0.5, 0.5]))

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_quantizer = _IQ()

    model = nn.Sequential(_Block())
    name = "0.input_quantizer._amax"
    # Dummy calib left [0.5, 0.5]; ckpt restore must exact-copy, not max-merge.
    updated = load_input_amax_exact(
        model,
        [(name, torch.tensor([3.0, 2.0]))],
        strict=False,
    )
    assert updated == 1
    buf = dict(model.named_buffers())[name]
    assert torch.equal(buf, torch.tensor([3.0, 2.0]))


def test_load_input_amax_exact_strict_requires_all_enabled(monkeypatch):
    from vexact.quantization import amax_sync
    from vexact.quantization.amax_sync import load_input_amax_exact

    class _IQ(nn.Module):
        def __init__(self, val):
            super().__init__()
            self.register_buffer("_amax", torch.tensor([val]))

        @property
        def amax(self):
            return self._amax

        @property
        def is_enabled(self):
            return True

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Module()
            self.a.input_quantizer = _IQ(1.0)
            self.b = nn.Module()
            self.b.input_quantizer = _IQ(1.0)

    model = _M()
    iq_map = {
        "a.input_quantizer": model.a.input_quantizer,
        "b.input_quantizer": model.b.input_quantizer,
    }
    monkeypatch.setattr(amax_sync, "build_input_quantizer_map", lambda _m: iq_map)

    with pytest.raises(RuntimeError, match="Export restore incomplete"):
        load_input_amax_exact(
            model,
            [("a.input_quantizer._amax", torch.tensor([9.0]))],
            strict=True,
        )

    n = load_input_amax_exact(
        model,
        [
            ("a.input_quantizer._amax", torch.tensor([9.0])),
            ("b.input_quantizer._amax", torch.tensor([8.0])),
        ],
        strict=True,
    )
    assert n == 2
    assert torch.equal(model.a.input_quantizer._amax, torch.tensor([9.0]))
    assert torch.equal(model.b.input_quantizer._amax, torch.tensor([8.0]))


def test_fill_input_amax_sentinel_without_modelopt():
    # fill_input_amax_sentinel requires modelopt TensorQuantizer isinstance checks;
    # with plain modules it should no-op (0 filled).
    class _IQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_amax", torch.tensor([2.0]))

        @property
        def amax(self):
            return self._amax

        @property
        def is_enabled(self):
            return True

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_quantizer = _IQ()

    model = _M()
    # Without modelopt, _require_tensor_quantizer may fail — skip if unavailable.
    modelopt = pytest.importorskip("modelopt")
    del modelopt
    filled = fill_input_amax_sentinel(model, value=-1.0)
    assert filled == 0  # not a real TensorQuantizer


def test_input_amax_buffer_name():
    assert input_amax_buffer_name("x.input_quantizer") == "x.input_quantizer._amax"
    assert input_amax_buffer_name("x.input_quantizer._amax") == "x.input_quantizer._amax"
