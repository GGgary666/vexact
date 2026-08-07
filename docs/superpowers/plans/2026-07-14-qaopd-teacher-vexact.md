# QAOPD Teacher VeXact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Implementation landed in-tree (2026-07-14). Local unit tests for prompt_logprobs / teacher shim / request fields pass; scheduler GPU tests and remote 1-step smoke still need CUDA env.

**Goal:** 让 QAOPD distillation teacher 走 VeXact BF16（无 QAT），对外兼容 pinned VeRL 的 vLLM `prompt_logprobs` 契约，达到与 student 同引擎家族的数值路径。

**Architecture:** 不改 VeRL 源码。通过 `teacher_vexact` 扩展 `_validate_topk_logprobs`；`VeXactReplica` 支持 `is_teacher_model`/`name_suffix` 与 colocated placement-only worker；teacher `generate(prompt_logprobs=K)` 映射为 score-only request，经现有 IPC→scheduler→inferencer 在 prefill logits 上分块算 full-vocab top-k，返回 `[S,K]` + 末位 dummy。

**Tech Stack:** Python, PyTorch, msgspec IPC, Ray, pinned VeRL `61f29997`, pytest。

**Spec:** `docs/superpowers/specs/2026-07-14-qaopd-teacher-vexact-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `vexact/inferencer/prompt_logprobs.py` (new) | Pure helpers: chunked full-vocab top-k logprobs + append dummy row |
| `vexact/integrations/verl/teacher_vexact.py` (new) | Idempotent `_validate_topk_logprobs` shim for `engine=vexact` |
| `vexact/core/request.py` | Score-only fields on Driver/Inference request & output |
| `vexact/core/runtime_data.py` | Packed top-k IDs/logprobs on `InferencerOutput` |
| `vexact/inferencer/inferencer.py` | Score-only batch: compute packed top-k, skip sampling |
| `vexact/core/scheduler.py` | Accumulate chunk rows, clear on preempt, finalize with dummy |
| `vexact/worker/driver_worker.py` | Mark active requests FAILED on gen-loop exception |
| `vexact/worker/worker_proxy.py` | Forward score fields Driver↔Inference |
| `vexact/integrations/verl/async_server.py` | Teacher role/naming, placement worker, generate branch, guards |
| `exp_scripts/qaopd/run_qwen3_1b7_gsm8k.sh` | Teacher → vexact, TP=1×4 replicas, BF16, no QAT |
| `tests/test_prompt_logprobs.py` (new) | Pure math + dummy + non-renorm |
| `tests/test_verl_teacher_vexact.py` (new) | Shim, naming, guards |
| `tests/test_{request_channel,scheduler,inferencer}.py` | Regressions / score path |
| `examples/README.md` | Document VeXact BF16 teacher default |

---

### Task 1: Pure top-k prompt_logprobs helper

**Files:**
- Create: `vexact/inferencer/prompt_logprobs.py`
- Test: `tests/test_prompt_logprobs.py`

- [ ] **Step 1: Write failing tests for full-vocab top-k + dummy**

```python
import torch
from vexact.inferencer.prompt_logprobs import (
    append_dummy_prompt_logprob_row,
    compute_full_vocab_topk_logprobs,
)

def test_full_vocab_topk_not_renormalized_in_k():
    # logits [4, 8], K=3 → ids/logprobs shape [4,3]
    # logprobs == topk_logits - logsumexp(full), mass usually < 1
    ...

def test_append_dummy_row():
    # S-1 real rows → S rows; last is zeros ids/logprobs
    ...
```

- [ ] **Step 2: Run tests — expect ImportError / fail**

```bash
pytest tests/test_prompt_logprobs.py -q
```

- [ ] **Step 3: Implement helpers**

```python
def compute_full_vocab_topk_logprobs(logits: torch.Tensor, k: int, *, chunk_size: int = 256):
    """logits [T,V] → (ids [T,K] int64, logprobs [T,K] float32) via chunked FP32."""
    ...

def append_dummy_prompt_logprob_row(ids, logprobs):
    """Append VeRL dummy last row: ids=0, logprobs=0."""
    ...
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_prompt_logprobs.py -q
```

- [ ] **Step 5: Commit** (only if user asks)

---

### Task 2: Teacher config shim (`teacher_vexact`)

**Files:**
- Create: `vexact/integrations/verl/teacher_vexact.py`
- Test: `tests/test_verl_teacher_vexact.py`

- [ ] **Step 1: Failing tests** — vexact accepts topk; vllm/sglang unchanged; idempotent; missing symbol fail-fast

- [ ] **Step 2: Implement** wrapping `DistillationTeacherModelConfig._validate_topk_logprobs` with sentinel; non-vexact delegates; vexact only checks `topk>0`

- [ ] **Step 3: `pytest tests/test_verl_teacher_vexact.py -q -k validate`**

---

### Task 3: Request / runtime_data score-only fields

**Files:**
- Modify: `vexact/core/request.py`, `vexact/core/runtime_data.py`, `vexact/worker/worker_proxy.py`
- Test: `tests/test_request_channel.py` (+ small unit asserts in `test_prompt_logprobs.py` or new helpers)

- [ ] **Step 1: Failing tests** — DriverRequest roundtrip with `score_only=True`, `prompt_logprobs_k=K`; output carries `prompt_ids`/`prompt_logprobs`; generation defaults omit fields (msgspec omit_defaults)

- [ ] **Step 2: Add fields**

```python
# DriverRequest / InferenceRequest
score_only: bool = False
prompt_logprobs_k: int = 0
# InferenceRequest accumulators
scored_topk_ids: list[list[int]] = field(default_factory=list)
scored_topk_logprobs: list[list[float]] = field(default_factory=list)

# DriverRequestOutput
prompt_ids: list[list[int]] | None = None
prompt_logprobs: list[list[float]] | None = None

# InferencerOutput
topk_token_ids: Tensor  # packed [total_tokens, K] or empty
topk_logprobs: Tensor
```

- [ ] **Step 3: Wire `from_driver_request` / `to_driver_request_output` / preempt clears scored_*`

- [ ] **Step 4: Pass existing `test_request_channel.py` + new cases**

---

### Task 4: Inferencer score-only path

**Files:**
- Modify: `vexact/inferencer/inferencer.py`
- Test: `tests/test_inferencer.py` (or extend with CPU-mocked packed logits unit if GPU-heavy)

- [ ] **Step 1: Failing test** — all-score-only batch returns packed top-k matching helper; mixed gen/score raises; different K raises

- [ ] **Step 2: In `_select_tokens` / `infer` last-rank path:**
  - If any `score_only`: require all score_only and same K
  - Use **all packed prefill logits** (not just last-token positions)
  - Call `compute_full_vocab_topk_logprobs`
  - Return dummy token_ids (zeros) + empty sample logprobs; fill InferencerOutput topk fields
  - Skip sampling for score-only

- [ ] **Step 3: Pass tests**

---

### Task 5: Scheduler accumulate / preempt / finalize

**Files:**
- Modify: `vexact/core/scheduler.py`
- Test: `tests/test_scheduler.py` + `tests/test_prompt_logprobs.py` integration cases

- [ ] **Step 1: Failing tests**
  - Non-chunked score-only: after one prefill step, FINISHED with S rows (S-1 real + dummy), `generated_tokens=[]`
  - Chunked: same final `[S,K]`
  - Preempt mid-prefill: clears scored_* then re-prefill → no duplicate rows

- [ ] **Step 2: Implement `update` branch for score_only:**
  - Slice packed top-k by `tokens_this_step` / query ranges
  - Append to `scored_topk_*`
  - When `num_computed_tokens == len(input_ids)`: append dummy, finalize (no decode)
  - `_preempt_request` / `preempt()` clears scored accumulators

- [ ] **Step 3: Pass scheduler + generation regression**

---

### Task 6: FAILED propagation in driver loop

**Files:**
- Modify: `vexact/worker/driver_worker.py`
- Test: extend `tests/test_driver_worker_proxy.py` or unit with mock scheduler

- [ ] **Step 1: On gen-loop exception, fail all active requests with reason and put on result queue; then re-raise/break**

- [ ] **Step 2: `VeXactServer.generate` must raise if `status==FAILED` (Task 7)**

---

### Task 7: Async server — role, placement worker, generate branch

**Files:**
- Modify: `vexact/integrations/verl/async_server.py`
- Test: `tests/test_verl_teacher_vexact.py`

- [ ] **Step 1: `VeXactReplica.__init__` accepts `is_teacher_model`, `name_suffix`; pass to super**

- [ ] **Step 2: Placement-only Ray worker for colocated teacher** (no `ServerAdapter(device_mesh=None)`)

- [ ] **Step 3: Unique names:**
  - Ray: `vexact_server_teacher_{rank}_{node}{name_suffix}`
  - `driver_id`: `vexact_teacher{name_suffix}_replica_{rank}`

- [ ] **Step 4: Teacher launch guards:** TP=DP=PP=1, QAT disabled, dtype BF16 (or unset→bf16), log `role=teacher dtype=bf16 qat=disabled`

- [ ] **Step 5: `generate`:**
  - Pop `prompt_logprobs` / `max_tokens` **before** asserts
  - If `prompt_logprobs` set: require `is_teacher_model`, temp=1.0, build score-only `DriverRequest`
  - Map output → `TokenOutput(token_ids=[], extra_fields={prompt_ids, prompt_logprobs})`
  - On FAILED: raise with request_id + reason

- [ ] **Step 6: Pass `tests/test_verl_teacher_vexact.py`**

---

### Task 8: Recipe + docs

**Files:**
- Modify: `exp_scripts/qaopd/run_qwen3_1b7_gsm8k.sh`, `examples/README.md`

- [ ] Teacher `inference.name=vexact`, TP/DP/PP=1, `num_replicas` covering 4 GPUs, `enforce_eager=True`, QAT off
- [ ] `VERL_USE_EXTERNAL_MODULES=...,qaopd,teacher_vexact`
- [ ] README note: default teacher = VeXact BF16, 4×1-GPU replicas

---

### Task 9: Verification gate

- [ ] `pytest tests/test_prompt_logprobs.py tests/test_verl_teacher_vexact.py tests/test_request_channel.py tests/test_scheduler.py tests/test_verl_qaopd.py -q`
- [ ] Optional GPU: `tests/test_verl_teacher_vexact_gpu.py` (HF short-seq + batch-composition invariance) when CUDA available
- [ ] Remote smoke `TOTAL_TRAINING_STEPS=1`：GPU4–7 无 `VLLM::Worker`

---

## Out of scope (this plan)

- VeRL/VeOmni source edits
- Teacher TP>1 / PP>1
- Changing `qaopd_mixed_kl_topk` formula
- Bit-exact match to vLLM
