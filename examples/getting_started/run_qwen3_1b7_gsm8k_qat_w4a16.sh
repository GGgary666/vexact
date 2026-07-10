#!/usr/bin/env bash
# Qwen3-1.7B GRPO on GSM8K with QAT w4a16 (weight-only) fake quantization.
#
# This is a lower-risk variant of run_qwen3_1b7_gsm8k_qat_w4a4.sh: it only
# fake-quantizes WEIGHTS (activations stay at native precision), which is the
# only QAT+GRPO configuration verified to converge by both NeMo-RL's QARL docs
# and verl's own native FSDP QAT support -- full W4A4 (weight+activation) is
# documented by NeMo-RL as having "known convergence issues" on GRPO. If you
# are seeing large rollout_corr/kl, chi2, or ppl_ratio blowups with the w4a4
# script, try this one first before assuming an integration bug.
#
# This mirrors run_qwen3_1b7_gsm8k.sh but enables Model-Optimizer (modelopt)
# fake quantization on BOTH sides:
#   - Training side (VeOmni FSDP actor only, NOT ref -- see below): via the
#     external_lib hook `vexact.integrations.verl.fsdp_enable_qat`, which
#     reads VEXACT_QAT_* env vars and monkey-patches VeOmni's model-building
#     and meta-init weight-loading entry points.
#   - Rollout side (VeXact): via
#     `actor_rollout_ref.rollout.engine_kwargs.vexact.qat.*`.
#
# The QAT config is defined ONCE below and forwarded to both sides so they
# insert identical quantizers.
#
# VeXact's "w4a16" alias is NVFP4 E2M1 weight-only: it reuses the exact same
# `NVFP4_DEFAULT_CFG` base as `w4a4` (so the weight quantizer's numerics are
# identical between the two modes -- same 16-element dynamic block scaling),
# but forces `*input_quantizer` off (see resolve_quant_cfg() in
# vexact/quantization/quantize.py; modelopt has no built-in all-layer
# NVFP4-weight-only constant, only MoE/MLP-restricted ones like
# `NVFP4_MLP_WEIGHT_ONLY_CFG`, so vexact builds the all-layer variant itself).
# Both the weight quantizer (always) and the activation quantizer (in w4a4)
# use `block_sizes={"type": "dynamic", ...}`, i.e. the scale is recomputed
# fresh from the live tensor on every forward -- there is no frozen/stale
# `amax` buffer to keep in sync between training and rollout for either mode.
#
# Rollout weight path (training-side fold, mandatory when QAT is on):
#   - Actor still trains with live weight fake quant + STE.
#   - Training-side fold casts FSDP float32 full_tensor weights to bf16 before
#     fold_weight so rollout receives the same QDQ source as actor old_log_prob.
#   - Rollout never runs live weight_quantizer (plain Linear for w4a16).
#
# The reference policy is intentionally kept in full precision here -- see
# ``VEXACT_QAT_SKIP_REF`` below (verl has no ``ref.model.external_lib``).
#
# Requires the `qat` extra (nvidia-modelopt), e.g. `uv sync --extra qat ...`.
set -euo pipefail
set -x

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export VERL_LOGGING_LEVEL=INFO
export NCCL_DEBUG=ERROR
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHDYNAMO_VERBOSE=1
export TORCH_COMPILE_DISABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# VeOmni Liger Patch
export VEOMNI_USE_LIGER_KERNEL=0

# ---------------------------------------------------------------------------
# QAT configuration (single source of truth for both training and rollout).
# ---------------------------------------------------------------------------
QAT_MODE="${QAT_MODE:-w4a16}"                # w4a4 (NVFP4 W+A) | w4a16 (NVFP4 weight-only)
QAT_CFG="${QAT_CFG:-}"                        # optional explicit modelopt cfg/recipe; overrides mode
QAT_IGNORE_PATTERNS="${QAT_IGNORE_PATTERNS:-lm_head,embed_tokens}"
QAT_CALIBRATE="${QAT_CALIBRATE:-0}"          # 0 = dynamic amax (recommended), 1 = static (needs amax sync)
# Weight fold is mandatory for QAT; kept for Hydra/config compatibility only.
QAT_PREFOLD_WEIGHTS="${QAT_PREFOLD_WEIGHTS:-1}"

# Training side reads these env vars (external_lib hook cannot see Hydra config).
export VEXACT_QAT_ENABLE=1
export VEXACT_QAT_MODE="${QAT_MODE}"
export VEXACT_QAT_IGNORE_PATTERNS="${QAT_IGNORE_PATTERNS}"
export VEXACT_QAT_CALIBRATE="${QAT_CALIBRATE}"
export VEXACT_QAT_PREFOLD_WEIGHTS=1
export VEXACT_QAT_SKIP_REF="${VEXACT_QAT_SKIP_REF:-1}"
if [[ -n "${QAT_CFG}" ]]; then
  export VEXACT_QAT_CFG="${QAT_CFG}"
fi

project_name='GRPO-QAT-W4A16-QWEN3-1-7b-Base-GSM8K'
ROOT_DIR="${ROOT_DIR:-/xpfs/fp4/gg/exp_ckpts/${project_name}}"
EXP_NAME=${EXP_NAME:-'0706_gsm8k_mini_512_w4a16'}
MODEL_PATH="${MODEL_PATH:-/xpfs/fp4/models/Qwen3-1.7B-Base}"
TRAIN_FILE="${TRAIN_FILE:-/xpfs/fp4/gg/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-/xpfs/fp4/gg/data/gsm8k/test.parquet}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
# Cap rollout KV cache to actual data limits (model config may advertise 32768).
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_PROFILER="${ROLLOUT_PROFILER:-0}"
DATA_SEED="${DATA_SEED:-42}"

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-/root/data/gg/zip_files/exp_logs/2026_06/${EXP_NAME}/tensorboard}"
LOG_FILE="${ROOT_DIR}/${EXP_NAME}.log"

# Attention backend: fa-invariant/flash_attention_3 (FA3) only supports SM90 (H100).
# On B200/GB200 (SM100+) use the FA4 CUTE kernel instead; on A100 (SM80) use triton-invariant.
vexact_attn_implementation=${INFER_FA_IMPL:-fa-invariant-cute}
# KV pool: default sized for ~64 concurrent seqs at ROLLOUT_MAX_MODEL_LEN (page_size=256).
# 1024 blocks reserves ~28GB KV on 1.7B and leaves little headroom on 8x hybrid nodes.
ROLLOUT_MAX_CONCURRENT=${ROLLOUT_MAX_CONCURRENT:-64}
_blocks_per_seq=$(( (ROLLOUT_MAX_MODEL_LEN + 255) / 256 ))
vexact_max_cache_blocks=${VEXACT_MAX_CACHE_BLOCKS:-$(( _blocks_per_seq * ROLLOUT_MAX_CONCURRENT ))}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-128}
# The QAT hook also pulls in the batch-invariant hook, so a single module here
# enables both on the training workers.
verl_model_external_lib=vexact.integrations.verl.fsdp_enable_qat
# Ref stays full precision via VEXACT_QAT_SKIP_REF (see QAT block above).
veomni_attn_implementation=${VEOMNI_ATTN_IMPLEMENTATION:-"veomni_flash_attention_4_with_sp"}
moe_implementation=${VEOMNI_MOE_IMPLEMENTATION:-"fused"}
enforce_eager=${ENFORCE_EAGER:-True}
if [[ "${ROLLOUT_PROFILER}" == "1" || "${ROLLOUT_PROFILER,,}" == "true" ]]; then
  rollout_profiler_enable=True
else
  rollout_profiler_enable=False
fi

profile_save_path="${ROOT_DIR}/${EXP_NAME}/verl_rollout_profile"
use_fused_kernels=True
fused_kernel_backend=torch
use_liger=False

FSDP_SIZE=${FSDP_SIZE:-8}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}

# Rollout-side QAT bool for Hydra (True/False).
if [[ "${QAT_CALIBRATE}" == "1" || "${QAT_CALIBRATE,,}" == "true" ]]; then
  QAT_CALIBRATE_BOOL=True
else
  QAT_CALIBRATE_BOOL=False
fi
QAT_PREFOLD_WEIGHTS_BOOL=True

mkdir -p "$ROOT_DIR"
mkdir -p "${ROOT_DIR}/${EXP_NAME}"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$profile_save_path"

cp "$SCRIPT_PATH" "$TENSORBOARD_DIR/$(basename "$SCRIPT_PATH")"
{
  echo "Run time: $(date)"
  echo "Script : $SCRIPT_PATH"
  echo "Command: $SCRIPT_PATH $*"
  echo "QAT    : mode=${QAT_MODE} cfg=${QAT_CFG:-<mode-default>} ignore=${QAT_IGNORE_PATTERNS} calibrate=${QAT_CALIBRATE} prefold=1 (mandatory)"
  echo "Rollout: max_model_len=${ROLLOUT_MAX_MODEL_LEN} max_cache_blocks=${vexact_max_cache_blocks} max_num_seqs=${rollout_max_num_seqs} enforce_eager=${enforce_eager} profiler=${rollout_profiler_enable}"
} > "$TENSORBOARD_DIR/run_command.txt"

# Optional explicit modelopt cfg override for the rollout side.
qat_cfg_override=()
if [[ -n "${QAT_CFG}" ]]; then
  qat_cfg_override+=("++actor_rollout_ref.rollout.engine_kwargs.vexact.qat.quant_cfg=${QAT_CFG}")
fi

python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.return_raw_chat=False \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    data.truncation='error' \
    data.seed="${DATA_SEED}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$veomni_attn_implementation \
    actor_rollout_ref.model.external_lib=$verl_model_external_lib \
    actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=$fused_kernel_backend \
    actor_rollout_ref.model.use_liger=$use_liger \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.data_loader_seed="${DATA_SEED}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.veomni.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.veomni.expert_parallel_size="${EP_SIZE}" \
    actor_rollout_ref.actor.veomni.attn_implementation=$veomni_attn_implementation \
    actor_rollout_ref.actor.veomni.moe_implementation=$moe_implementation \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.veomni.param_offload=True \
    actor_rollout_ref.ref.veomni.optimizer_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.name=vexact \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=$vexact_max_cache_blocks \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=$vexact_attn_implementation \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.qat.enable=True \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.qat.mode=$QAT_MODE \
    "++actor_rollout_ref.rollout.engine_kwargs.vexact.qat.ignore_patterns=[${QAT_IGNORE_PATTERNS}]" \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.qat.calibrate=$QAT_CALIBRATE_BOOL \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.qat.prefold_weights=$QAT_PREFOLD_WEIGHTS_BOOL \
    "${qat_cfg_override[@]}" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=$enforce_eager \
    actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.profiler.enable=$rollout_profiler_enable \
    actor_rollout_ref.rollout.profiler.save_path=$profile_save_path \
    reward.reward_manager.name=dapo \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$EXP_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=$ROOT_DIR/$EXP_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=15 \
    "$@" \
    2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
