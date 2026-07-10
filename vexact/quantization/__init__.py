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

from .config import (
    DEFAULT_IGNORE_PATTERNS,
    MODE_TO_MODELOPT_CFG,
    VALID_MODES,
    QATConfig,
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
    is_model_quantized,
    quantize_model,
    resolve_quant_cfg,
)


__all__ = [
    "QATConfig",
    "MODE_TO_MODELOPT_CFG",
    "VALID_MODES",
    "DEFAULT_IGNORE_PATTERNS",
    "quantize_model",
    "resolve_quant_cfg",
    "is_model_quantized",
    "fold_weight",
    "fold_weights_generator",
    "get_training_weight_quantizer_map",
    "get_weight_quantizer_map",
    "maybe_prepare_rollout_prefold",
    "prepare_rollout_prefold",
    "set_training_weight_quantizer_map",
]
