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

"""E2M1 helpers shared by reference and Triton paths."""

from __future__ import annotations

import torch

from .constants import BLOCK_SIZE

_E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Match vLLM ``test_nvfp4_quant.cast_to_fp4`` bucket boundaries."""
    sign = torch.sign(x)
    ax = x.abs()
    out = torch.zeros_like(ax)
    out[(ax > 0.25) & (ax < 0.75)] = 0.5
    out[(ax >= 0.75) & (ax <= 1.25)] = 1.0
    out[(ax > 1.25) & (ax < 1.75)] = 1.5
    out[(ax >= 1.75) & (ax <= 2.5)] = 2.0
    out[(ax > 2.5) & (ax < 3.5)] = 3.0
    out[(ax >= 3.5) & (ax <= 5.0)] = 4.0
    out[ax > 5.0] = 6.0
    return out * sign


def unpack_fp4_bytes(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    m, packed_n = packed.shape
    flat = packed.flatten()
    low = flat & 0x0F
    high = (flat >> 4) & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    mag = (combined & 0x07).to(torch.long)
    lut = _E2M1_MAGNITUDES.to(device=packed.device)
    return (lut[mag] * torch.where(signs, -1.0, 1.0)).reshape(m, packed_n * 2)


def recover_swizzled_block_scales(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    scale_n = n // BLOCK_SIZE
    rounded_m = (m + 127) // 128 * 128
    rounded_n = (scale_n + 3) // 4 * 4
    tmp = scale.view(1, rounded_m // 128, rounded_n // 4, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    return tmp.reshape(rounded_m, rounded_n).to(torch.float32)[:m, :scale_n]


def dequantize_nvfp4_packed(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    m, packed_k = packed.shape
    k = packed_k * 2
    values = unpack_fp4_bytes(packed).reshape(m, k // BLOCK_SIZE, BLOCK_SIZE)
    sf = recover_swizzled_block_scales(block_scale.view(torch.float8_e4m3fn), m, k)
    out = (values * (sf / global_scale.to(torch.float32)).unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)
