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

"""NVFP4 numeric constants (no hard dependency on vLLM at import time)."""

from __future__ import annotations

BLOCK_SIZE = 16
# float4_e2m1 max / torch.float8_e4m3fn max — match vLLM scalar_types / finfo.
E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
