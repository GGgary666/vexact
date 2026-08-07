#!/usr/bin/env bash
# Export a VeXact/verl BF16 (non-QAT) actor checkpoint to HuggingFace format.
#
# Handles VeOmni mesh_dim_names ('dp_shard',) that stock
# `python -m verl.model_merger` rejects. For QAT / NVFP4 see export_qat_to_hf.sh.
#
# Usage:
#   scripts/export_bf16_to_hf.sh \
#     --actor-ckpt /path/to/global_step_N/actor \
#     --export-dir /path/to/hf_bf16 \
#     -v
#
# Example (GSM8K BF16 GRPO):
#   scripts/export_bf16_to_hf.sh \
#     --actor-ckpt /xpfs/fp4/gg/exp_ckpts/GRPO-NVFP4-QAT-QWEN3-1-7b-Base-GSM8K/0706_gsm8k_mini_512_bf16/global_step_435/actor \
#     --export-dir /xpfs/fp4/gg/exp_ckpts/exports/0706_gsm8k_mini_512_bf16_step435_hf \
#     -v
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${ROOT}/scripts/export_bf16_to_hf.py" "$@"
