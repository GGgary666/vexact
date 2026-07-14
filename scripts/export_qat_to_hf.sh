#!/usr/bin/env bash
# Export a VeXact/verl QAT actor checkpoint to a real NVFP4 HuggingFace model
# for vLLM: --quantization modelopt_fp4
#
# w4a4: restores input_quantizer.amax from actor ckpt; hf_quant_config
#        quant_algo=NVFP4 (activation-quantized).
# w4a16: weight-only; rewrites ModelOpt's NVFP4 tag to W4A16_NVFP4 so vLLM
#        uses Marlin W4A16 (no activation quant). Random calib is never used.
#
# Usage (w4a4, from actor shards):
#   scripts/export_qat_to_hf.sh \
#     --mode w4a4 \
#     --actor-ckpt /path/to/global_step_N/actor \
#     --export-dir /path/to/nvfp4_hf \
#     --keep-merged -v
#
# w4a16:
#   scripts/export_qat_to_hf.sh \
#     --mode w4a16 \
#     --actor-ckpt /path/to/global_step_N/actor \
#     --export-dir /path/to/nvfp4_hf \
#     --keep-merged -v
#
# Already-merged HF bf16 (w4a4 uses <hf-bf16>/input_amax.pt or --amax-path):
#   scripts/export_qat_to_hf.sh \
#     --mode w4a4 --hf-bf16 /path/to/hf_bf16 \
#     --amax-path /path/to/input_amax.pt \
#     --export-dir /path/to/nvfp4_hf
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${ROOT}/scripts/export_qat_to_hf.py" "$@"
