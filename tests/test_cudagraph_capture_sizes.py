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

"""Unit tests for CudaGraphManager.build_capture_sizes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_cudagraph_utils():
    """Load cudagraph_utils without pulling heavy batch_invariant / triton deps."""
    repo = Path(__file__).resolve().parents[1]
    pkg = "vexact"
    if pkg not in sys.modules:
        m = ModuleType(pkg)
        m.__path__ = [str(repo / "vexact")]
        sys.modules[pkg] = m

    bio = ModuleType("vexact.batch_invariant_ops")
    bio.__path__ = []
    sys.modules["vexact.batch_invariant_ops"] = bio

    kv = ModuleType("vexact.batch_invariant_ops.kv_cache_context")
    kv.KVCacheStore = object
    kv.set_kv_cache_context = lambda *a, **k: None
    sys.modules["vexact.batch_invariant_ops.kv_cache_context"] = kv

    cfg = ModuleType("vexact.config")

    class CacheConfig:  # noqa: D401 - stub
        pass

    cfg.CacheConfig = CacheConfig
    sys.modules["vexact.config"] = cfg

    rt = ModuleType("vexact.core.runtime_data")
    rt.InputBuffers = object
    core = ModuleType("vexact.core")
    core.__path__ = []
    sys.modules["vexact.core"] = core
    sys.modules["vexact.core.runtime_data"] = rt

    path = repo / "vexact" / "inferencer" / "cudagraph_utils.py"
    spec = importlib.util.spec_from_file_location("vexact.inferencer.cudagraph_utils", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vexact.inferencer.cudagraph_utils"] = mod
    # Ensure parent package exists for relative clarity.
    inf = ModuleType("vexact.inferencer")
    inf.__path__ = [str(repo / "vexact" / "inferencer")]
    sys.modules.setdefault("vexact.inferencer", inf)
    spec.loader.exec_module(mod)
    return mod


_cudagraph_utils = _load_cudagraph_utils()
CudaGraphManager = _cudagraph_utils.CudaGraphManager


def test_build_capture_sizes_power_of_two():
    assert CudaGraphManager.build_capture_sizes(2048) == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]


def test_build_capture_sizes_non_power_of_two_caps_at_max():
    """QAOPD recipe uses max_prompt+max_response+1 (=1537); must not emit 2048."""
    sizes = CudaGraphManager.build_capture_sizes(1537)
    assert sizes[-1] == 1537
    assert max(sizes) == 1537
    assert 1024 in sizes
    assert 2048 not in sizes


def test_build_capture_sizes_one():
    assert CudaGraphManager.build_capture_sizes(1) == [1]


def test_build_capture_sizes_rejects_non_positive():
    with pytest.raises(ValueError):
        CudaGraphManager.build_capture_sizes(0)
