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

"""vLLM-aligned NVFP4 fake quantization (standalone from ModelOpt).

Public entrypoint: :func:`nvfp4_fake_quant`.
"""

from .api import nvfp4_fake_quant, nvfp4_global_amax, nvfp4_global_scale
from .modelopt_patch import (
    install_vllm_aligned_nvfp4_kernel,
    is_vllm_aligned_nvfp4_kernel_installed,
    uninstall_vllm_aligned_nvfp4_kernel,
)

__all__ = [
    "nvfp4_fake_quant",
    "nvfp4_global_amax",
    "nvfp4_global_scale",
    "install_vllm_aligned_nvfp4_kernel",
    "uninstall_vllm_aligned_nvfp4_kernel",
    "is_vllm_aligned_nvfp4_kernel_installed",
]
