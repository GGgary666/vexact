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

"""Unit tests for full-vocab top-k prompt_logprobs helpers (VeRL contract)."""

import pytest
import torch

from vexact.inferencer.prompt_logprobs import (
    append_dummy_prompt_logprob_row,
    compute_full_vocab_topk_logprobs,
    finalize_scored_prompt_logprobs,
    validate_prompt_logprobs_params,
)


def test_full_vocab_topk_shape_and_values():
    torch.manual_seed(0)
    t, v, k = 5, 8, 4
    logits = torch.randn(t, v, dtype=torch.bfloat16)
    ids, logprobs = compute_full_vocab_topk_logprobs(logits, k=k, chunk_size=2)

    assert ids.shape == (t, k)
    assert logprobs.shape == (t, k)
    assert ids.dtype == torch.int64
    assert logprobs.dtype == torch.float32

    logits_fp32 = logits.float()
    expected_vals, expected_ids = torch.topk(logits_fp32, k=k, dim=-1, sorted=True)
    expected_logprobs = expected_vals - torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
    assert torch.equal(ids, expected_ids)
    assert torch.allclose(logprobs, expected_logprobs, atol=1e-5, rtol=1e-5)


def test_full_vocab_topk_not_renormalized_in_k():
    logits = torch.tensor(
        [
            [10.0, 9.0, 0.0, -5.0],
            [1.0, 0.5, 0.25, 0.0],
        ],
        dtype=torch.float32,
    )
    ids, logprobs = compute_full_vocab_topk_logprobs(logits, k=2)
    # Within-top-k renormalization would make exp(logprobs).sum(dim=-1) == 1.
    mass = torch.exp(logprobs).sum(dim=-1)
    assert torch.all(mass < 1.0 - 1e-6)
    # Full-vocab: logprobs == topk - logsumexp(full)
    topk_vals, topk_ids = torch.topk(logits, k=2, dim=-1, sorted=True)
    expected = topk_vals - torch.logsumexp(logits, dim=-1, keepdim=True)
    assert torch.equal(ids, topk_ids)
    assert torch.allclose(logprobs, expected)


def test_append_dummy_prompt_logprob_row():
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    logprobs = torch.tensor([[-0.1, -0.2, -0.3], [-1.0, -2.0, -3.0]], dtype=torch.float32)
    out_ids, out_logprobs = append_dummy_prompt_logprob_row(ids, logprobs)
    assert out_ids.shape == (3, 3)
    assert out_logprobs.shape == (3, 3)
    assert torch.equal(out_ids[:2], ids)
    assert torch.allclose(out_logprobs[:2], logprobs)
    assert torch.equal(out_ids[-1], torch.zeros(3, dtype=torch.int64))
    assert torch.equal(out_logprobs[-1], torch.zeros(3, dtype=torch.float32))


def test_append_dummy_on_lists():
    ids = [[1, 2], [3, 4]]
    logprobs = [[-0.1, -0.2], [-0.3, -0.4]]
    out_ids, out_logprobs = append_dummy_prompt_logprob_row(ids, logprobs)
    assert out_ids == [[1, 2], [3, 4], [0, 0]]
    assert out_logprobs == [[-0.1, -0.2], [-0.3, -0.4], [0.0, 0.0]]


def test_validate_prompt_logprobs_params():
    validate_prompt_logprobs_params(seq_len=3, k=2, vocab_size=10, temperature=1.0)
    with pytest.raises(ValueError, match="empty"):
        validate_prompt_logprobs_params(seq_len=0, k=2, vocab_size=10, temperature=1.0)
    with pytest.raises(ValueError, match="k"):
        validate_prompt_logprobs_params(seq_len=3, k=0, vocab_size=10, temperature=1.0)
    with pytest.raises(ValueError, match="vocab"):
        validate_prompt_logprobs_params(seq_len=3, k=11, vocab_size=10, temperature=1.0)
    with pytest.raises(ValueError, match="temperature"):
        validate_prompt_logprobs_params(seq_len=3, k=2, vocab_size=10, temperature=0.7)


def test_k_greater_than_rows_chunking_edge():
    logits = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    ids, logprobs = compute_full_vocab_topk_logprobs(logits, k=4, chunk_size=1)
    assert ids.shape == (3, 4)
    assert torch.equal(ids[0], torch.tensor([3, 2, 1, 0]))


def test_finalize_scored_prompt_logprobs_contract():
    # Simulate S=5 prefill rows then finalize → 4 real + dummy
    scored_ids = [[i, i + 1] for i in range(5)]
    scored_lp = [[-float(i), -float(i) - 0.1] for i in range(5)]
    out_ids, out_lp = finalize_scored_prompt_logprobs(scored_ids, scored_lp, seq_len=5, k=2)
    assert len(out_ids) == 5
    assert out_ids[:4] == scored_ids[:4]
    assert out_ids[-1] == [0, 0]
    assert out_lp[-1] == [0.0, 0.0]


def test_finalize_seq_len_one():
    out_ids, out_lp = finalize_scored_prompt_logprobs([[9, 8]], [[-1.0, -2.0]], seq_len=1, k=2)
    assert out_ids == [[0, 0]]
    assert out_lp == [[0.0, 0.0]]


def test_preempt_clears_score_accumulators():
    from transformers import GenerationConfig

    from vexact.core.request import InferenceRequest

    req = InferenceRequest(
        request_id="s1",
        generation_config=GenerationConfig(max_new_tokens=1),
        input_ids_list=[1, 2, 3],
        score_only=True,
        prompt_logprobs_k=2,
    )
    req.scored_topk_ids = [[1, 2]]
    req.scored_topk_logprobs = [[-0.1, -0.2]]
    req.preempt()
    assert req.scored_topk_ids == []
    assert req.scored_topk_logprobs == []


def test_score_only_topk_slices_cudagraph_padding():
    """CUDA-graph capture pads logits to graph size; score-only must use real T.

    Repro of production failure: consumed=231, total=256 unused rows.
    """
    from vexact.inferencer.prompt_logprobs import score_only_topk_from_packed_logits

    real_tokens = 3
    pad_tokens = 8  # capture size > real packed tokens
    vocab = 16
    k = 2

    logits = torch.zeros(1, pad_tokens, vocab)
    for t in range(pad_tokens):
        logits[0, t, t % vocab] = 10.0

    topk_ids, topk_lp = score_only_topk_from_packed_logits(logits, num_tokens=real_tokens, k=k)
    assert topk_ids.shape == (real_tokens, k)
    assert topk_lp.shape == (real_tokens, k)
    assert int(topk_ids[0, 0].item()) == 0

    # Eager path: already exact length.
    exact = logits[:, :real_tokens, :]
    ids2, lp2 = score_only_topk_from_packed_logits(exact, num_tokens=real_tokens, k=k)
    assert torch.equal(ids2, topk_ids)
    assert torch.allclose(lp2, topk_lp)

    with pytest.raises(RuntimeError, match="packed logits rows"):
        score_only_topk_from_packed_logits(logits[:, :2, :], num_tokens=real_tokens, k=k)
