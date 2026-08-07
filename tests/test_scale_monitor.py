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

"""Tests for NVFP4 global_scale / amax monitoring helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

from vexact.quantization.scale_monitor import (
    _NVFP4_GLOBAL_SCALE_DENOM,
    amax_to_global_scale,
    assert_calib_frozen,
    collect_global_scale_stats,
    fingerprint_input_global_scales,
    get_quantizer_stats,
)


class _FakeTQ(nn.Module):
    def __init__(self, amax, *, enabled=True, calib=False, dynamic=False):
        super().__init__()
        self.is_enabled = enabled
        self._if_calib = calib
        self._dynamic = dynamic
        if amax is not None:
            self.register_buffer("_amax", torch.tensor(amax, dtype=torch.float32))

    @property
    def amax(self):
        return getattr(self, "_amax", None)


def test_amax_to_global_scale():
    amax = torch.tensor(2688.0)  # 6 * 448
    scale = amax_to_global_scale(amax)
    assert float(scale.item()) == pytest.approx(1.0)


def test_get_quantizer_stats_and_fingerprint():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Module()
            self.layers.input_quantizer = _FakeTQ(2.0)
            self.layers.weight_quantizer = _FakeTQ(4.0)
            self.other = _FakeTQ(None, enabled=True)

    with patch(
        "vexact.quantization.scale_monitor._require_tensor_quantizer",
        return_value=_FakeTQ,
    ):
        m = M()
        # Treat FakeTQ as TensorQuantizer via isinstance patching: named_modules
        # isinstance check uses the returned class.
        stats = get_quantizer_stats(m)
        assert stats["total"] == 3
        assert stats["enabled"] == 3
        assert stats["with_amax"] == 2
        assert stats["positive_amax"] == 2
        assert stats["calib_enabled"] == 0

        fp = fingerprint_input_global_scales(m)
        assert "layers.input_quantizer" in fp
        assert fp["layers.input_quantizer"] == pytest.approx(2.0 / _NVFP4_GLOBAL_SCALE_DENOM)

        iq = collect_global_scale_stats(m, name_suffix="input_quantizer")
        assert iq["amax"]["count"] == 1.0
        assert iq["amax"]["mean"] == pytest.approx(2.0)


def test_assert_calib_frozen_raises():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_quantizer = _FakeTQ(1.0, calib=True)

    with patch(
        "vexact.quantization.scale_monitor._require_tensor_quantizer",
        return_value=_FakeTQ,
    ):
        with pytest.raises(RuntimeError, match="disable_calib"):
            assert_calib_frozen(M())
