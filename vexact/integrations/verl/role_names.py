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

"""Role-qualified Ray / driver names for student vs teacher VeXact replicas."""

from __future__ import annotations


def build_vexact_role_names(
    *,
    replica_rank: int,
    node_rank: int,
    is_teacher_model: bool = False,
    is_reward_model: bool = False,
    name_suffix: str = "",
) -> dict[str, str]:
    """Build unique Ray actor and VeXact driver identifiers.

    ``name_suffix`` should already include a leading underscore when non-empty
    (pinned VeRL ``RolloutReplica`` stores ``f"_{name_suffix}"``).
    """
    if is_reward_model:
        role = "reward"
    elif is_teacher_model:
        role = "teacher"
    else:
        role = "student"

    suffix = name_suffix or ""
    server_name = f"vexact_server_{role}_{replica_rank}_{node_rank}{suffix}"
    driver_id = f"vexact_{role}{suffix}_replica_{replica_rank}"
    return {
        "role": role,
        "server_name": server_name,
        "driver_id": driver_id,
    }
