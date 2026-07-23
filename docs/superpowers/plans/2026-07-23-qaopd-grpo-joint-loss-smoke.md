# QAOPD GRPO Joint Loss — Smoke Checklist

Local prechecks (2026-07-23): `bash -n` OK on both scripts; Hydra flag assertions OK;
`pytest --noconftest tests/test_verl_qaopd.py tests/test_quantization_qaopd.py` → 24 passed.
Old script `run_qwen3_1b7_gsm8k.sh` untouched (`use_task_rewards=False`, `n=1`).

## Cluster runs (manual)

```bash
# 1) Joint 1-step smoke
TOTAL_TRAINING_STEPS=1 ALLOW_RANDOM_CALIB=1 CALIB_DATA=random \
  bash exp_scripts/qaopd/run_qwen3_1b7_gsm8k_grpo_joint.sh

# 2) Pure distill regression
TOTAL_TRAINING_STEPS=1 ALLOW_RANDOM_CALIB=1 CALIB_DATA=random \
  bash exp_scripts/qaopd/run_qwen3_1b7_gsm8k.sh

# 3) Short compare (50–100 steps), same hardware
TOTAL_TRAINING_STEPS=100 bash exp_scripts/qaopd/run_qwen3_1b7_gsm8k_grpo_joint.sh
TOTAL_TRAINING_STEPS=100 bash exp_scripts/qaopd/run_qwen3_1b7_gsm8k.sh

# 4) If |pg_loss| vs |distillation/loss| differ ≥10× in first ~50 steps:
DISTILLATION_LOSS_COEF=0.5|1.0|1.5 bash exp_scripts/qaopd/run_qwen3_1b7_gsm8k_grpo_joint.sh
```

Expect: finite `actor/pg_loss` (joint) + `distillation/loss`; no NaN; input global_scale stable.
