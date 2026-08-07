# QAOPD Teacher 走 VeXact BF16 设计

**日期**: 2026-07-14  
**状态**: 已完成技术自检，待用户确认  
**相关**: QAOPD on-policy distillation（`qaopd_mixed_kl_topk`）
**VeRL 基线**: `61f29997fb026a5a269dafccfe2f3bb800e32ef4`

本文只描述上述 pinned VeRL；较旧 `verl_repo_0326` 的单 teacher、
`DistillationConfig` 内嵌 engine match 和 standalone teacher 路径不作为实现依据。

## 1. 背景与目标

当前 QAOPD recipe 布局为：

| GPU | 引擎 | 角色 |
|-----|------|------|
| 0–3 | VeXact W4A4 | Student actor + on-policy rollout |
| 4–7 | **vLLM** BF16 | 冻结 teacher，采 top-k logprobs |

用户目标：**数值一致**——teacher 也走 VeXact batch-invariant 路径，避免 vLLM 与 student 引擎数值差。

锁定约束：

- Teacher：**BF16、不开 QAT**（对齐 NeMo：冻结 BF16 teacher + W4A4 student）
- 不 fork / 不改 VeRL 源码文件
- 复用现有 `AsyncTeacherLLMServerManager` 与 QAOPD loss
- 方案：**A** — 在 VeXact 上兼容 vLLM 的 `prompt_logprobs` 契约

## 2. 非目标

- 不改变 `qaopd_mixed_kl_topk` 的 NeMo mixed KL 语义
- Teacher 不开启 fake-quant / W4A4
- 不把 student 与 teacher 合并到同一 GPU pool（仍默认 4+4）
- 不追求与 vLLM bit-exact；追求同引擎家族（VeXact）下的可复现数值路径

## 3. 技术核验结论与现状阻塞

以下结论已对照当前 VeXact 和 pinned VeRL 的真实接口核验：

1. **VeRL 外部模块加载时机可用。** `verl/__init__.py` 在包初始化时按顺序导入
   `VERL_USE_EXTERNAL_MODULES`，早于训练阶段把 Hydra 配置转换为
   `DistillationConfig`。因此可以在外部模块中注册 QAOPD loss 并安装兼容 shim。
   该环境变量必须在进程第一次 `import verl` 前设置，并传播到 Ray driver/worker
   的 runtime environment。

2. **Pinned VeRL 的限制位于
   `DistillationTeacherModelConfig._validate_topk_logprobs`。**
   `DistillationConfig.__post_init__` 会对每个 teacher 调用
   `validate_and_prepare_for_distillation(...)`；其中 top-k 校验支持 `vllm` 和
   `sglang`，但 `vexact` 会落入默认分支并抛 `NotImplementedError`。失败发生在
   `omega_conf_to_dataclass` / dataclass instantiate 阶段，不是 Hydra YAML compose。

3. **Teacher 外部契约确实是 vLLM 风格 `prompt_logprobs`。**
   `AsyncTeacherLLMServerManager` 调用：

   ```python
   generate(
       prompt_ids=sequence_ids,
       sampling_params={
           "max_tokens": 1,
           "temperature": 1.0,
           "prompt_logprobs": topk,
       },
   )
   ```

   并读取 `TokenOutput.extra_fields["prompt_ids"]` 和
   `extra_fields["prompt_logprobs"]`。

4. **原设计中“在 `VeXactServer.generate` 直接 forward”不可行。**
   `VeXactServer` 所在 Ray actor 只持有 `VeXact/DriverClient`；真正模型位于
   `WorkerProcManager` 创建的 GPU 子进程。scoring 必须沿 request IPC →
   scheduler → inferencer → worker 的现有数据面执行。

5. **Pinned teacher 使用 colocated resource pool，而不是 standalone。**
   `MultiTeacherModelManager` 为每个 teacher 切分 resource pool，随后调用
   `replica.init_colocated(sub_pool)`。当前
   `VeXactReplica.get_ray_class_with_init_args()` 会构造
   `ServerAdapter(device_mesh=None)`，但 `ServerAdapter.__init__` 会访问
   `device_mesh.size()`。Teacher 需要独立的、仅负责 Ray GPU placement 的轻量
   worker，不能复用训练侧 hybrid adapter。

6. **teacher replica 构造签名和角色隔离均不完整。**
   Pinned manager 会传 `is_teacher_model=True` 和由 teacher key 派生的
   `name_suffix`；当前 `VeXactReplica` 两者都不接受。Student 与 teacher 还会使用
   `vexact_server_{replica}_{node}` 和 `verl_rollout_replica_{replica}`，会造成 Ray
   actor 名、IPC socket 和 weight-transfer socket 冲突。所有 server/driver ID
   必须包含 `student|teacher|reward` 角色及 teacher `name_suffix`。

7. **首版并行边界必须收紧。** 当前 VeXact 配置只把
   `pipeline_model_parallel_size` 映射到内部 `ParallelConfig`，并不实现 VeRL
   `tensor_model_parallel_size>1` 的语义。首版 teacher 明确只支持
   `TP=DP=PP=1`；4 张 teacher GPU 表示 4 个单卡 BF16 replica，而不是 TP=4。

## 4. 修订后的架构

对 VeRL 继续呈现方案 A 的 vLLM-compatible 接口，但 VeXact 内部新增
**score-only request**，复用现有 prefill 数据面：

```mermaid
flowchart LR
  studentSeq["student prompt+response"] --> teacherMgr["AsyncTeacherLLMServerManager"]
  teacherMgr -->|"generate(prompt_logprobs=k, max_tokens=1)"| vexactSrv["VeXactServer adapter"]
  vexactSrv --> scoreReq["score-only DriverRequest"]
  scoreReq --> ipc["request IPC"]
  ipc --> sched["VeXact scheduler / prefill"]
  sched --> infer["BF16 batch-invariant inferencer"]
  infer --> topk["chunked FP32 full-vocab top-k"]
  topk --> scoreOut["score output IPC"]
  scoreOut --> extra["TokenOutput.extra_fields"]
  extra --> qaopd["qaopd_mixed_kl_topk"]
```

GPU 布局保持：

- 0–3：student actor + VeXact rollout（W4A4 QAT）
- 4–7：4 个单卡 VeXact teacher replica（BF16、无 QAT）
- 进程列表中不再出现 `VLLM::Worker`

### 4.1 Control plane

- `teacher_vexact.py` 安装 idempotent
  `DistillationTeacherModelConfig._validate_topk_logprobs` compatibility shim。
- `VeXactReplica` 接收并转发 `is_teacher_model` 和 `name_suffix`。
- colocated teacher 使用轻量 Ray resource worker，仅占用并暴露 GPU placement；
  不实例化 `ServerAdapter`，不访问 `device_mesh`，也不参与权重同步。
- Ray actor 名和内部 `driver_id` 使用角色与 suffix，例如
  `vexact_server_teacher_teacher_model_0_0` 与
  `vexact_teacher_teacher_model_replica_0_node_0`。
- Teacher 启动时校验：`TP=DP=PP=1`、resolved dtype 为 BF16、QAT 配置为空；
  任一条件不满足均 fail-fast，不静默降级。
- Pinned teacher 在 `free_cache_engine=True` 时会调用 colocated `wake_up/sleep`；
  该路径必须真实可用，不能套用 standalone 的 no-op 分支。

### 4.2 Data plane

- `VeXactServer.generate` 必须在通用生成参数断言之前识别并移除
  `prompt_logprobs` / `max_tokens`，随后构造 score-only request。
- score-only request 进入已有 request channel 和 scheduler；模型仍在受管 GPU worker
  中执行，不引入第二份模型。
- Inferencer 在每个 prefill chunk 上取得 packed logits，并只为 score-only batch
  计算 top-k；scheduler 按每个 request 的 `tokens_this_step` 切分并累积结果。
- 当完整 prompt prefill 完成后直接 finalize，不采样额外 token、不进入 decode。
- 若发生 preemption，必须清空该 request 已累积的 scoring 输出，避免 re-prefill
  重复行。
- Teacher server 是专用实例；首版要求一个 infer batch 中全部为 score-only request
  且 `k` 相同。遇到生成/scoring 混合或不同 `k` 时明确报错。

## 5. `prompt_logprobs` 精确契约

设输入序列为 `x[0:S]`，teacher logits 为 `z[0:S, V]`：

- 返回 shape 必须为 `[S, K]`，其中 `K=topk`。
- 返回第 `i` 行（`0 <= i < S-1`）来自 `z[i]`，即在上下文
  `x[:i+1]` 后预测 `x[i+1]` 时的 teacher 分布。
- 第 `S-1` 行是 VeRL 约定的 dummy：
  `ids=[0]*K`、`logprobs=[0.0]*K`。
- top-k token IDs 从 **full-vocabulary raw logits** 上选择，按降序排列。
- 返回值是 full-vocabulary 概率下的 log-prob，而不是 top-k 内重归一化：

  ```python
  topk_logits, topk_ids = torch.topk(logits_fp32, k=K, dim=-1, sorted=True)
  topk_logprobs = topk_logits - torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
  ```

  QAOPD loss 之后才执行 top-k 内的重归一化。
- 为限制峰值显存，按 token-row chunk 转 FP32 并计算
  `topk + logsumexp`；禁止物化完整 `[S,V]` FP32 `log_softmax` 副本。
- 只把 `[S,K]` IDs/logprobs 经 IPC 返回；logprobs 对外为 FP32。
- `temperature` 必须等于 `1.0`，`K>0` 且 `K<=vocab_size`，输入序列不能为空。

## 6. 文件级改动

| 文件 | 明确职责 |
|------|----------|
| `vexact/integrations/verl/teacher_vexact.py`（新） | 扩展 pinned VeRL `_validate_topk_logprobs`；仅对 `engine=vexact` 生效 |
| `vexact/integrations/verl/async_server.py` | teacher role + `name_suffix`、colocated placement worker、唯一命名、BF16/no-QAT guards；把 `prompt_logprobs` 映射到 score-only request 和 `extra_fields` |
| `vexact/core/request.py` | 为 driver/inference request 与 output 增加显式 scoring mode、top-k 参数和 `[S,K]` 结果字段 |
| `vexact/core/runtime_data.py` | 为 inferencer→scheduler 增加 packed top-k IDs/logprobs；普通生成保持空 tensor 默认值 |
| `vexact/inferencer/inferencer.py` | 在 prefill logits 上分块计算 full-vocab top-k logprobs；score-only 时不采样 |
| `vexact/core/scheduler.py` | 按 request 累积 chunk 结果、preemption 清理、末位 dummy、score-only finalize |
| `vexact/worker/driver_worker.py` | 将 scoring 数据面异常转成 FAILED request output，避免 generation loop 退出后调用方永久等待 |
| `exp_scripts/qaopd/run_qwen3_1b7_gsm8k.sh` | teacher engine 改为 `vexact`；锁定单卡 replica、BF16、无 QAT、eager |
| `tests/test_verl_teacher_vexact.py`（新） | config shim、replica role/suffix、colocated placement worker、命名和参数 guard |
| `tests/test_prompt_logprobs.py`（新） | 位置对齐、full-vocab logprob、dummy、chunked prefill、preemption |
| `tests/test_verl_teacher_vexact_gpu.py`（新） | 短序列 HF 对照与 batch-composition invariance（GPU） |
| `tests/test_{request_channel,scheduler,inferencer}.py` | 新字段序列化、FAILED 传播和普通 generation 回归 |
| `examples/README.md` | 明确默认 teacher 为 VeXact BF16、4 个单卡 replica |

首版明确不修改：

- VeRL / VeOmni 安装树源码
- `vexact/quantization/qaopd.py` 的 loss 公式
- `vexact/distributed/pp_messager.py`（teacher PP>1 本版不支持）
- student 的现有 generation/weight-sync 行为

## 7. Pinned VeRL top-k compatibility shim

锁定 patch 目标为
`DistillationTeacherModelConfig._validate_topk_logprobs`，不包装整个
`DistillationConfig.__post_init__`，避免复制或扰动多 teacher 的长度准备逻辑。

实现策略：

1. 保存原始 `_validate_topk_logprobs`，并以 sentinel 保证只包装一次。
2. 非 `engine=vexact` 时无条件委托原函数，保持 pinned VeRL 的 vLLM/SGlang
   行为完全不变。
3. 对 `engine=vexact`：
   - 若 distillation 未启用、loss 不使用 top-k 或 `topk is None`，保持原有短路语义；
   - 否则只校验 `topk>0`，不写 boot-time `max_logprobs`；实际 K 由每次 teacher
     request 的 `prompt_logprobs` 参数传入。
4. 不捕获并吞掉其他配置错误，也不改写 prompt/response 长度。
5. 若目标类/方法缺失或签名不兼容，external module import 时 fail-fast，并明确
   报告只兼容 VeRL `61f29997...`。

该 shim 只解除 pinned VeRL 对 top-k engine 名称的硬编码，不改变
`MultiTeacherModelManager`、resource-pool 切分或 teacher manager 数据流。

## 8. Recipe 锁定配置

```bash
# 4 张 teacher GPU = 4 个单卡 replica，不是 TP=4
TEACHER_TP=1

distillation.teacher_models.teacher_model.inference.name=vexact
distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1
distillation.teacher_models.teacher_model.inference.data_parallel_size=1
distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size=1
distillation.teacher_models.teacher_model.inference.enforce_eager=True

# 显式表达且由 server guard 校验：teacher 不量化
++distillation.teacher_models.teacher_model.inference.engine_kwargs.vexact.qat.enable=False
++distillation.teacher_models.teacher_model.inference.engine_kwargs.vexact.attn_impl=${vexact_attn_implementation}
```

外部模块按依赖顺序加载：

```text
vexact.integrations.verl.register,
vexact.integrations.verl.qaopd,
vexact.integrations.verl.teacher_vexact
```

- `register` 先注册 `vexact` rollout replica。
- `qaopd` 再注册 `qaopd_mixed_kl_topk`，使
  `DistillationLossConfig.__post_init__` 能解析 loss settings。
- `teacher_vexact` 最后安装 top-k engine compatibility shim。
- 该值必须在首次 `import verl` 前设置，并通过 Ray runtime environment 保证
  driver 和 worker 看到相同模块列表。

## 9. 错误处理与兼容边界

- `prompt_logprobs` scoring 只在 `is_teacher_model=True` 的 VeXact server 可用。
- 非 teacher server 收到 scoring 请求时 fail-fast，避免误用 student rollout。
- Teacher 收到普通生成请求、混合 request kind、不同 `K`、非 1.0 temperature、
  QAT enabled、非 BF16、或 `TP/DP/PP != 1` 时均给出带配置值的错误。
- Pinned teacher 使用 `init_colocated`；`wake_up/sleep` 必须在该模式下完成实际
  memory-saver 操作并等待未完成 scoring request 排空。
- scoring 失败必须把 request 标记为 FAILED 并经 request channel 返回；不得让
  generation thread 退出后令调用方永久等待。
- `VeXactServer` 收到 FAILED output 时必须把携带 request ID 和 worker 原因的异常
  抛回 `AsyncTeacherLLMServerManager`，不能构造空的成功 `TokenOutput`。
- 所有新增 request/output 字段带默认值，确保普通 generation 的 msgspec
  序列化和已有测试保持兼容。

## 10. 测试计划

### 10.1 纯单元测试

- synthetic logits：
  - `S=5, K=4` → IDs/logprobs shape `[5,4]`
  - 前 4 行等于手工 `topk(logits)` 与 full-vocab `logsumexp`
  - 第 5 行严格为 dummy zeros
  - 证明结果不是 top-k 内重归一化（top-k probability mass 通常 `<1`）
- 参数 guard：空序列、`K<=0`、`K>V`、temperature 非 1.0。
- `_validate_topk_logprobs` shim：
  - vexact top-k 校验成功且不写 `engine_kwargs.vllm`
  - vLLM 与 SGLang 行为与原实现一致
  - 原异常原样传播
  - 重复 import 不产生双层 wrapper
- role/namespace：student 与多个 teacher key 的 actor 名、driver ID、IPC 地址均不相同。

### 10.2 Scheduler / request 数据面

- 非 chunked 和 chunked prefill 产生相同 `[S,K]`。
- preemption → re-prefill 不重复 top-k 行。
- score-only 在 prefill 结束立即完成，`new_token_ids=[]`，不执行 decode。
- 普通 generation 回归测试保持不变。

### 10.3 GPU 集成

- 用短序列与 HF BF16 reference 对照，验证 causal shift、top-k IDs 和 logprobs。
- 核心数值目标：同一序列单独 scoring 与和其他长度序列共同 batching 时，
  top-k IDs 和 FP32 logprobs 完全一致；同时改变 batch 顺序再验证一次。
- 运行 `TOTAL_TRAINING_STEPS=1` QAOPD smoke：
  - GPU4–7 为 VeXact teacher worker，无 `VLLM::Worker`
  - 日志明确显示 `role=teacher dtype=bf16 qat=disabled`
  - 训练产生有限的 `distillation/*` metrics 并完成一步更新

## 11. 风险与缓解

| 风险 | 缓解 |
|------|------|
| causal shift / dummy 行错一位 | synthetic + HF reference 双重测试锁定 `z[i] -> row i` |
| `[tokens,vocab]` logits 与 FP32 归一化显存过高 | scheduler 限制 packed token budget；按 token rows 转 FP32，禁止完整 FP32 log-softmax 副本 |
| chunked prefill 或 preemption 导致重复/缺行 | request 保存累积状态；preempt 明确清空；两种路径做等价测试 |
| colocated teacher 复用 hybrid adapter 启动失败 | 新增 placement-only Ray worker；teacher 不构造 `ServerAdapter(device_mesh=None)` |
| student/teacher 同 rank 或多 teacher key 命名冲突 | actor、driver、request/control/weight socket 使用 role + `name_suffix` |
| VeRL 升级后 monkeypatch 漂移 | idempotent wrapper + 符号/签名 guard + 非 vexact 路径回归 |
| 当前 VeXact 无 TP 语义 | 首版 fail-fast 限定 TP=DP=PP=1；通过 4 个单卡 replica 吞吐扩展 |
| request 内部异常导致调用永久挂起 | FAILED output 必须经 IPC 返回，并补错误路径测试 |

## 12. 验收标准

- [ ] `inference.name=vexact` 时 pinned `_validate_topk_logprobs` 通过且其他 engine 无回归
- [ ] Teacher colocated resource pool 成功启动，不访问空 `device_mesh`
- [ ] Student/多个 teacher key 的 Ray actor 与所有本地 IPC 命名无冲突
- [ ] Teacher resolved dtype 为 BF16，QAT 为 disabled；违规配置 fail-fast
- [ ] `prompt_ids` / `prompt_logprobs` 严格满足 `[S,K]` 契约和末位 dummy
- [ ] chunked/non-chunked/preemption scoring 结果一致
- [ ] 同一序列跨 batch composition/order 的 top-k IDs 与 logprobs 一致
- [ ] GPU4–7 为 VeXact teacher worker，无 `VLLM::Worker`
- [ ] QAOPD 1-step smoke 成功并输出有限的 distillation metrics
- [ ] 现有普通 VeXact generation、QAT rollout 和 QAOPD loss 单测无回归

## 13. 已锁定决策

- Teacher 默认 `enforce_eager=True`，稳定和可核验优先于 CUDA graph 吞吐。
- Scoring 使用现有 worker/prefill request 数据面，不在 Ray async server 进程直接
  forward，也不加载第二份模型。
- 不使用 control channel 直接调用 `DriverWorker.score()`：该方案会绕过 scheduler
  的并发 batching / paged-attention 上下文，并让 control-server 线程与 generation
  线程竞争同一模型。显式 score-only request 改动文件更多，但复用现有模型执行、
  KV 管理、排队和 request lifecycle，符合数值一致与训练吞吐目标。
- Pinned teacher 走 colocated resource pool；placement-only worker 只提供 Ray
  GPU placement，真正 BF16 模型仍由 VeXact managed worker 子进程持有。
- 首版 teacher 仅支持单卡 replica（TP=DP=PP=1）；4 卡用于 4 replica。
- 对外保持 VeRL 的 vLLM-compatible `prompt_logprobs` 契约；内部不依赖 vLLM。
