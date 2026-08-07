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

"""QAT (fake quantization) support for VeXact, backed by NVIDIA Model-Optimizer.

Importing this package does NOT import modelopt; modelopt is only pulled in when
quantization is actually enabled (see :mod:`vexact.quantization.quantize`).
"""

from .amax_sync import (
    apply_input_amax_buffers,
    build_input_quantizer_map,
    chain_weights_with_input_amax,
    get_training_input_quantizer_map,
    iter_input_amax_buffers,
    load_input_amax_exact,
    set_training_input_quantizer_map,
)
from .config import (
    DEFAULT_IGNORE_PATTERNS,
    MODE_TO_MODELOPT_CFG,
    VALID_MODES,
    QATConfig,
)
from .cpa import (
    CPA_LOSS_TYPES,
    CPAConfig,
    compute_cpa_loss,
    compute_cpa_token_objective,
    temporarily_disable_fake_quant_and_eval,
)
from .qaopd import (
    QAOPD_KL_TYPES,
    QAOPD_LOSS_MODE,
    QAOPDConfig,
    compute_qaopd_token_kl,
    compute_qaopd_topk_loss,
    gather_student_topk_logits,
    renormalize_topk_log_probs,
)
from .export import (
    export_nvfp4_hf_checkpoint,
    extract_input_amax_from_state_dict,
    filter_hf_weight_state_dict,
    is_modelopt_extra_state_key,
    merge_input_amax_shards,
    merge_plain_tensor_shards,
    normalize_mesh_dim_names,
    prepare_model_for_nvfp4_export,
    rewrite_nvfp4_quant_algo_for_mode,
    validate_nvfp4_export_dir,
)
from .fold import (
    fold_weight,
    fold_weights_generator,
    get_training_weight_quantizer_map,
    get_weight_quantizer_map,
    maybe_prepare_rollout_prefold,
    prepare_rollout_prefold,
    set_training_weight_quantizer_map,
)
from .quantize import (
    build_calibration_forward_loop,
    is_model_quantized,
    quantize_model,
    resolve_calibration_forward_loop,
    resolve_quant_cfg,
)
from .scale_monitor import (
    amax_to_global_scale,
    collect_global_scale_stats,
    fingerprint_input_global_scales,
    get_quantizer_stats,
    log_scale_monitor,
)


__all__ = [
    "QATConfig",
    "CPAConfig",
    "CPA_LOSS_TYPES",
    "QAOPDConfig",
    "QAOPD_KL_TYPES",
    "QAOPD_LOSS_MODE",
    "MODE_TO_MODELOPT_CFG",
    "VALID_MODES",
    "DEFAULT_IGNORE_PATTERNS",
    "quantize_model",
    "resolve_quant_cfg",
    "is_model_quantized",
    "build_calibration_forward_loop",
    "resolve_calibration_forward_loop",
    "get_quantizer_stats",
    "collect_global_scale_stats",
    "amax_to_global_scale",
    "fingerprint_input_global_scales",
    "log_scale_monitor",
    "compute_cpa_loss",
    "compute_cpa_token_objective",
    "temporarily_disable_fake_quant_and_eval",
    "compute_qaopd_token_kl",
    "compute_qaopd_topk_loss",
    "gather_student_topk_logits",
    "renormalize_topk_log_probs",
    "fold_weight",
    "fold_weights_generator",
    "get_training_weight_quantizer_map",
    "get_weight_quantizer_map",
    "maybe_prepare_rollout_prefold",
    "prepare_rollout_prefold",
    "set_training_weight_quantizer_map",
    "build_input_quantizer_map",
    "iter_input_amax_buffers",
    "chain_weights_with_input_amax",
    "apply_input_amax_buffers",
    "load_input_amax_exact",
    "get_training_input_quantizer_map",
    "set_training_input_quantizer_map",
    "export_nvfp4_hf_checkpoint",
    "extract_input_amax_from_state_dict",
    "filter_hf_weight_state_dict",
    "is_modelopt_extra_state_key",
    "merge_input_amax_shards",
    "merge_plain_tensor_shards",
    "normalize_mesh_dim_names",
    "prepare_model_for_nvfp4_export",
    "rewrite_nvfp4_quant_algo_for_mode",
    "validate_nvfp4_export_dir",
]
