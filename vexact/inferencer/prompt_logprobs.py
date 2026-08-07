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

"""Full-vocabulary top-k prompt logprobs for distillation teacher scoring.

Matches pinned VeRL / vLLM ``prompt_logprobs`` contract used by
``AsyncTeacherLLMServerManager``:

* Select top-k over the **full vocabulary** (not renormalized within top-k).
* ``logprob = topk_logit - logsumexp(full_vocab_logits)``.
* Callers append a dummy last row of zeros for the final prompt token.
"""

from __future__ import annotations

from typing import Optional, Sequence, TypeVar, Union

import torch
from torch import Tensor


ListMatrix = list[list[Union[int, float]]]
T = TypeVar("T")


def validate_prompt_logprobs_params(
    *,
    seq_len: int,
    k: int,
    vocab_size: int,
    temperature: float,
) -> None:
    if seq_len <= 0:
        raise ValueError(f"prompt_logprobs scoring rejects empty sequences (seq_len={seq_len})")
    if k <= 0:
        raise ValueError(f"prompt_logprobs k must be > 0, got {k}")
    if k > vocab_size:
        raise ValueError(f"prompt_logprobs k ({k}) exceeds vocab_size ({vocab_size})")
    if temperature != 1.0:
        raise ValueError(
            f"prompt_logprobs scoring requires temperature=1.0, got {temperature}"
        )


def compute_full_vocab_topk_logprobs(
    logits: Tensor,
    k: int,
    *,
    chunk_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Compute sorted top-k token IDs and full-vocab logprobs.

    Args:
        logits: ``[T, V]`` logits (any floating dtype; converted to FP32 in chunks).
        k: Number of top tokens.
        chunk_size: Rows processed per chunk to bound peak FP32 memory.

    Returns:
        ``(ids [T, K] int64, logprobs [T, K] float32)``
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [T, V], got shape {tuple(logits.shape)}")
    t, v = logits.shape
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if k > v:
        raise ValueError(f"k ({k}) exceeds vocab size ({v})")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    ids_out = torch.empty((t, k), dtype=torch.int64, device=logits.device)
    logprobs_out = torch.empty((t, k), dtype=torch.float32, device=logits.device)

    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        chunk = logits[start:end].to(dtype=torch.float32)
        topk_logits, topk_ids = torch.topk(chunk, k=k, dim=-1, sorted=True)
        chunk_logprobs = topk_logits - torch.logsumexp(chunk, dim=-1, keepdim=True)
        ids_out[start:end] = topk_ids
        logprobs_out[start:end] = chunk_logprobs

    return ids_out, logprobs_out


def score_only_topk_from_packed_logits(
    logits: Tensor,
    *,
    num_tokens: int,
    k: int,
    chunk_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Top-k over packed prefill logits, dropping CUDA-graph pad rows.

    Args:
        logits: ``[1, T_pad, V]`` or ``[T_pad, V]`` model outputs. When CUDA
            graphs are enabled, ``T_pad`` is the capture size and may exceed the
            real packed token count ``num_tokens``.
        num_tokens: Real packed token count for this step (sum of
            ``tokens_this_step`` / ``batch_position_ids.shape[1]``).
        k: Top-k width.
        chunk_size: Forwarded to :func:`compute_full_vocab_topk_logprobs`.

    Returns:
        ``(ids [num_tokens, K], logprobs [num_tokens, K])``.
    """
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"expected logits batch dim 1, got shape {tuple(logits.shape)}")
        packed = logits[0]
    elif logits.ndim == 2:
        packed = logits
    else:
        raise ValueError(f"logits must be [1,T,V] or [T,V], got shape {tuple(logits.shape)}")

    if packed.shape[0] < num_tokens:
        raise RuntimeError(
            f"score-only packed logits rows ({packed.shape[0]}) < num_tokens ({num_tokens})"
        )
    return compute_full_vocab_topk_logprobs(packed[:num_tokens], k=k, chunk_size=chunk_size)


def append_dummy_prompt_logprob_row(
    ids: Union[Tensor, Sequence[Sequence[int]]],
    logprobs: Union[Tensor, Sequence[Sequence[float]]],
    *,
    k: Optional[int] = None,
) -> tuple[Union[Tensor, list[list[int]]], Union[Tensor, list[list[float]]]]:
    """Append VeRL dummy last-prompt-token row: ids=0, logprobs=0.0."""
    if isinstance(ids, Tensor):
        if not isinstance(logprobs, Tensor):
            raise TypeError("ids and logprobs must both be Tensor or both be nested lists")
        width = int(ids.shape[-1]) if ids.numel() else int(k or 0)
        if width <= 0:
            raise ValueError("cannot append dummy row without k or non-empty ids")
        dummy_ids = torch.zeros((1, width), dtype=ids.dtype if ids.numel() else torch.int64, device=ids.device)
        dummy_lp = torch.zeros((1, width), dtype=torch.float32, device=logprobs.device)
        if ids.numel() == 0:
            return dummy_ids.to(dtype=torch.int64), dummy_lp
        return torch.cat([ids, dummy_ids], dim=0), torch.cat([logprobs.float(), dummy_lp], dim=0)

    if ids:
        width = len(ids[0])
    elif k is not None and k > 0:
        width = int(k)
    else:
        raise ValueError("cannot append dummy row to empty prompt_ids without k")
    out_ids = [list(row) for row in ids]
    out_lp = [list(row) for row in logprobs]
    out_ids.append([0] * width)
    out_lp.append([0.0] * width)
    return out_ids, out_lp


def finalize_scored_prompt_logprobs(
    scored_ids: list[list[int]],
    scored_logprobs: list[list[float]],
    *,
    seq_len: int,
    k: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Drop last prefill row and append VeRL dummy → ``[seq_len, k]``."""
    if len(scored_ids) != seq_len or len(scored_logprobs) != seq_len:
        raise ValueError(
            f"expected {seq_len} scored rows before finalize, "
            f"got ids={len(scored_ids)} logprobs={len(scored_logprobs)}"
        )
    return append_dummy_prompt_logprob_row(scored_ids[:-1], scored_logprobs[:-1], k=k)
