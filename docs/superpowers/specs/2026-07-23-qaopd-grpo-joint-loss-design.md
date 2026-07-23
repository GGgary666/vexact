# QAOPD + GRPO 联合 Loss 设计

**日期**: 2026-07-23  
**状态**: 已批准  
**相关**: `qaopd_mixed_kl_topk`、fixed input `global_scale`、teacher VeXact BF16  
**VeRL 基线**: `61f29997fb026a5a269dafccfe2f3bb800e32ef4`（与既有 QAOPD teacher 设计一致）

## 1. 背景与动机

当前 QAOPD 训练为 **纯 on-policy top-k mixed KL**：student 在自身 rollout 上对齐 teacher 分布，**不使用** verifier / GRPO task reward。固定 input `global_scale` 已证明有助于 QAT 数值稳定，应保留。

观察：纯 distill 爬到理想 GSM8K acc 需要较多 step。根因之一是目标不对齐——KL 逼近 teacher，而 acc 由 outcome verifier 定义。

文献与 verl 官方 OPD 均支持将 distill 与 task-reward RL 相加；RG-OPD 等用 reward 做 gate，本设计 **不做 gate**，只做加法联合。

## 2. 已确认决策

| 项 | 选择 |
|----|------|
| 目标形态 | 引入 verifier/GRPO，接受「不再是纯 distill」 |
| 组合方式 | **A**：\(\mathcal{L}=\mathcal{L}_{\mathrm{GRPO}}+\lambda\mathcal{L}_{\mathrm{distill}}\) |
| 落地路径 | **路径 2 / Branch A**：Phase0 确认原生联合 → **新建**联合 recipe；不改旧纯 distill 脚本；不做 vexact loss 薄适配 |
| \(\lambda\) 初值 | **1.0**；前 ~50 step 按 `pg_loss` vs `distill_loss` 量级再调 |
| 保留 | fixed input `global_scale`；`qaopd_mixed_kl_topk` 公式不变 |
| 明确不做 | RG-OPD gate、PG-OPD（`use_policy_gradient`）、CPA、升级整个 VeRL pin、修改旧 recipe |

## 3. 目标与非目标

### 目标

- Student：W4A4 QAT + **calib 后冻结的 input global_scale**
- Teacher：BF16 VeXact（既有 teacher 路径）
- Distill：现有 `qaopd_mixed_kl_topk`（NeMo 语义不变）
- Policy：标准 GRPO（pinned VeRL `ppo_loss`）
- 最终 loss：\(\mathcal{L}=\mathcal{L}_{\mathrm{GRPO}}+\lambda\mathcal{L}_{\mathrm{distill}}\)，默认 \(\lambda=1.0\)
- 交付：新建联合 recipe + guard/文档；旧纯 distill 脚本保持不变作基线

### 非目标

- 不改 top-k mixed KL 数值公式
- 不做 reward-gate / trajectory filter
- 不启用 CPA（与「标准 GRPO term + distill」不同；继续 `reject_cpa`）
- **不修改** `exp_scripts/qaopd/run_qwen3_1b7_gsm8k.sh`（纯 distill 基线）
- 不追求与文档版最新 verl bit-exact API 名；以 **pinned 提交** 为准

## 4. 路径 2 分阶段方案

### 4.1 Phase 0 — 探针 pinned VeRL（已完成）

在 `/Users/gonggu/Desktop/workplace/rl_mismatch/projects/verl@61f29997` 确认：

| 问题 | 结果 |
|------|------|
| 是否有 `use_task_rewards`？ | **Y** — `verl/workers/config/distillation.py::DistillationLossConfig` |
| 默认值？ | `use_task_rewards=True`；YAML 默认同为 `true` |
| `distillation_loss_coef` 字段名？ | `distillation_loss_coef`（默认 `1.0`） |
| 联合公式是否已是 `pg + coef * distill`？ | **Y** — `verl/trainer/distillation/losses.py::distillation_ppo_loss`：`use_task_rewards=False` 时 `policy_loss=0`，再 `policy_loss += distill_loss * coef` |
| Hook 点（若需薄适配） | **不需要** — 原生 Hydra 即可 |
| 当前纯 distill recipe | `use_task_rewards=False`，`rollout.n=1` — **保持不动** |

**结论**：薄适配取消；走原生 Hydra（Branch A）。新建 `exp_scripts/qaopd/run_qwen3_1b7_gsm8k_grpo_joint.sh`。

### 4.2 Phase 1 — 验证（新建联合 recipe）

#### 共同运行时约束

- `VEXACT_QAT_ENABLE=1`，`VEXACT_QAT_MODE=w4a4`
- fixed input `global_scale`（现有 post-calib freeze + scale monitor）
- `VEXACT_QAOPD_ENABLE=1`，`loss_mode=qaopd_mixed_kl_topk`
- `VEXACT_CPA_ENABLE=0`
- `actor_rollout_ref.model.use_fused_kernels=False`
- `use_policy_gradient=false`：保持 GKD 式 top-k 直传

#### 脚本布局

| 脚本 | 角色 |
|------|------|
| `exp_scripts/qaopd/run_qwen3_1b7_gsm8k.sh` | 纯 distill 基线（不改） |
| `exp_scripts/qaopd/run_qwen3_1b7_gsm8k_grpo_joint.sh` | GRPO + distill 联合 |

联合脚本关键 Hydra：

```yaml
distillation:
  distillation_loss:
    loss_mode: qaopd_mixed_kl_topk
    use_task_rewards: true
    distillation_loss_coef: 1.0
    use_policy_gradient: false
actor_rollout_ref:
  rollout:
    n: 4
reward:
  reward_manager:
    name: dapo
```

#### λ 调参规则

- 初值 `1.0`（`DISTILLATION_LOSS_COEF`）
- 前约 50 step：比较 `|pg_loss|` 与 `|distill_loss|`
- 若相差 ≥ 10×：按比例调整 \(\lambda\)
- 不引入自适应 \(\lambda\) 代码

#### Smoke 与短跑

1. 联合脚本 `TOTAL_TRAINING_STEPS=1` smoke
2. 旧纯 distill 脚本 1-step 回归
3. 短跑（50–100 step）对比两脚本 val reward/acc

### 4.3 Phase 2 — 固化进 vexact

1. Runtime guard：继续拒 CPA；`VEXACT_QAOPD_USE_POLICY_GRADIENT=1` fail-fast
2. 文档：`examples/README.md` 并列两脚本
3. 测试：guard 单测；既有 QAOPD 单测无回归
4. 本 spec 状态为「已批准」

## 5. 架构与数据流（概念）

```text
prompts
  → student VeXact W4A4 rollout (on-policy, n=4)
  → reward / GRPO advantage  (task signal)
  → teacher VeXact BF16 top-k logprobs (distill signal)
  → student train forward (full logits, no fused LCE)
       ├─ L_GRPO  from advantages + policy ratio
       └─ L_distill from qaopd_mixed_kl_topk
  → L = L_GRPO + λ * L_distill
  → optimizer (QAT weights; global_scale frozen)
```

## 6. 验收标准

| 类别 | 标准 |
|------|------|
| 正确性 | 联合脚本 `final = GRPO + λ·QAOPD`；默认 λ=1.0 |
| 基线 | 旧纯 distill 脚本未改动，仍可对比 |
| QAT | fixed global_scale 行为不变；scale monitor 正常 |
| 主指标 | 同硬件、同 step 预算下，联合的 val acc（或等价 reward）**不低于**纯 distill，或更少 step 达到同一 acc |
| 稳定 | 无 NaN；smoke 与短跑可完成 |

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| GRPO 与 distill 量级失衡 | λ=1 起步 + 50-step 量级规则 |
| Step 变慢（n=4） | 接受 wall-clock 略增；主看 acc/step |
| 与旧注释「QAOPD replaces policy loss」冲突 | 更新 `qaopd.py` docstring 与 README |

## 8. 开放问题（已决议）

- GRPO `n`：联合脚本默认 **4**（旧纯 distill 保持 `n=1`）
- 短跑对比：沿用现有 `TEST_FREQ` / step 预算，两脚本并列跑
