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

"""Correctness tests for vLLM-aligned NVFP4 fake quant (vexact.quantization.nvfp4)."""

from __future__ import annotations

import pytest
import torch
from vllm.platforms import current_platform

from vexact.quantization.nvfp4 import nvfp4_fake_quant
from vexact.quantization.nvfp4.reference import nvfp4_fake_quant_vllm_qdq
from vexact.quantization.nvfp4.triton_kernel import (
    nvfp4_fake_quant_triton_block1d,
    nvfp4_fake_quant_triton_fused,
    triton_is_available,
)
from vexact.quantization.nvfp4.reference import nvfp4_global_amax, nvfp4_global_scale

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

if not current_platform.has_device_capability(100):
    pytest.skip("NVFP4 fake quant tests require SM100 (B200)", allow_module_level=True)

if not hasattr(torch.ops._C, "scaled_fp4_quant"):
    pytest.skip("vLLM scaled_fp4_quant unavailable", allow_module_level=True)


def _make_x(shape: tuple[int, int], seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, dtype=torch.bfloat16, generator=gen) * 0.5).cuda()


@pytest.mark.parametrize(
    "shape",
    [(1, 64), (16, 128), (32, 4096), (64, 7168), (128, 7168)],
)
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("backend", ["fused", "block1d"])
def test_nvfp4_fake_quant_triton_matches_vllm_qdq(
    shape: tuple[int, int], seed: int, backend: str
) -> None:
    if not triton_is_available():
        pytest.skip("Triton backend unavailable")
    x = _make_x(shape, seed)
    ref = nvfp4_fake_quant_vllm_qdq(x)
    gs = nvfp4_global_scale(nvfp4_global_amax(x))
    if backend == "fused":
        out = nvfp4_fake_quant_triton_fused(x, gs)
    else:
        out = nvfp4_fake_quant_triton_block1d(x, gs)
    diff = (out.float() - ref.float()).abs()
    assert diff.max().item() == 0.0, f"max diff {diff.max().item()} backend={backend}"


@pytest.mark.parametrize("shape", [(16, 128), (32, 4096), (64, 512)])
def test_nvfp4_fake_quant_eager_matches_vllm_qdq(shape: tuple[int, int]) -> None:
    x = _make_x(shape, 7 if shape != (16, 128) else 0)
    ref = nvfp4_fake_quant_vllm_qdq(x)
    out = nvfp4_fake_quant(x, backend="eager")
    diff = (out.float() - ref.float()).abs()
    assert diff.max().item() == 0.0


@pytest.mark.parametrize("shape", [(32, 4096)])
def test_nvfp4_fake_quant_auto_matches_vllm_qdq(shape: tuple[int, int]) -> None:
    x = _make_x(shape, 3)
    ref = nvfp4_fake_quant_vllm_qdq(x)
    out = nvfp4_fake_quant(x, backend="auto")
    diff = (out.float() - ref.float()).abs()
    assert diff.max().item() == 0.0
