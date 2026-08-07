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

"""Public API for vLLM-aligned NVFP4 fake quantization."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import torch

from .reference import (
    nvfp4_fake_quant_eager,
    nvfp4_fake_quant_vllm_qdq,
    nvfp4_global_amax,
    nvfp4_global_scale,
)
from .triton_kernel import (
    nvfp4_fake_quant_triton,
    nvfp4_fake_quant_triton_block1d,
    nvfp4_fake_quant_triton_fused,
    triton_is_available,
)

Backend = Literal["auto", "triton", "triton_block1d", "triton_staged", "vllm_qdq", "eager"]


class Nvfp4FakeQuantBackend(str, Enum):
    AUTO = "auto"
    TRITON = "triton"
    TRITON_BLOCK1D = "triton_block1d"
    TRITON_STAGED = "triton_staged"
    VLLM_QDQ = "vllm_qdq"
    EAGER = "eager"


def _resolve_backend(backend: Backend) -> Nvfp4FakeQuantBackend:
    if backend == "auto":
        return Nvfp4FakeQuantBackend.TRITON if triton_is_available() else Nvfp4FakeQuantBackend.VLLM_QDQ
    return Nvfp4FakeQuantBackend(backend)


@torch.inference_mode()
def nvfp4_fake_quant(
    x: torch.Tensor,
    global_amax: torch.Tensor | None = None,
    *,
    backend: Backend = "auto",
    block_size: int = 16,
) -> torch.Tensor:
    """Apply NVFP4 fake quantization aligned with vLLM B200 ``scaled_fp4_quant``.

    Args:
        x: bf16/fp16 tensor, last dim divisible by ``block_size``.
        global_amax: optional per-tensor amax; computed from ``x`` when omitted.
        backend:
            - ``auto``: fused Triton on Hopper+ if available, else vLLM QDQ.
            - ``triton``: fused 2D-tiled autotuned Triton (preferred).
            - ``triton_block1d``: fused 1D-over-blocks autotuned Triton.
            - ``triton_staged``: legacy torch SF + Triton round (A/B only).
            - ``vllm_qdq``: real quant + dequant via vLLM C++ (ground truth).
            - ``eager``: pure PyTorch reference.
        block_size: NVFP4 block size (16 for B200).

    Returns:
        Fake-quantized tensor with same shape/dtype as ``x``.
    """
    if block_size != 16:
        raise ValueError("Only block_size=16 is supported for NVFP4 on B200.")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"nvfp4_fake_quant expects bf16/fp16, got {x.dtype}")

    resolved = _resolve_backend(backend)
    if global_amax is None:
        global_amax = nvfp4_global_amax(x)

    if resolved == Nvfp4FakeQuantBackend.VLLM_QDQ:
        return nvfp4_fake_quant_vllm_qdq(x, global_amax)
    if resolved == Nvfp4FakeQuantBackend.EAGER:
        return nvfp4_fake_quant_eager(x, global_amax)

    gs = nvfp4_global_scale(global_amax)
    if resolved == Nvfp4FakeQuantBackend.TRITON_STAGED:
        return nvfp4_fake_quant_triton(x, gs, block_size=block_size)
    if resolved == Nvfp4FakeQuantBackend.TRITON_BLOCK1D:
        return nvfp4_fake_quant_triton_block1d(x, gs, block_size=block_size)
    return nvfp4_fake_quant_triton_fused(x, gs, block_size=block_size)
