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


import logging
import os
import time
from typing import Generator, Optional

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _is_global_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _qat_mode_label(*, has_iq_map: bool) -> str:
    mode = os.environ.get("VEXACT_QAT_MODE", "").strip()
    if mode:
        return mode
    return "w4a4" if has_iq_map else "w4a16"


def _resolve_qat_step(kwargs: dict, adapter: "ServerAdapter") -> int:
    step = kwargs.get("global_steps", kwargs.get("step"))
    if step is not None:
        return int(step)
    # verl's update_weights path does not currently pass global_steps; fall back
    # to a per-adapter QAT sync ordinal so logs stay monotonically readable.
    adapter._qat_sync_step = getattr(adapter, "_qat_sync_step", 0) + 1
    return int(adapter._qat_sync_step)


def _amax_applied_from_receive_result(result) -> int:
    """Best-effort extract applied amax count from receive_weights return value."""
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    if isinstance(result, (list, tuple)):
        counts = [x for x in result if isinstance(x, int)]
        return max(counts) if counts else 0
    return 0


# Baseline input global_scale fingerprint (name -> scale); set on first monitor log.
_input_global_scale_baseline = None  # type: Optional[dict]


def _scale_monitor_every() -> int:
    """Log global_scale drift every N QAT sync steps (env ``VEXACT_QAT_SCALE_LOG_EVERY``)."""
    raw = os.environ.get("VEXACT_QAT_SCALE_LOG_EVERY", "50")
    try:
        return max(1, int(raw))
    except ValueError:
        return 50


def _maybe_log_input_global_scale_drift(iq_map: dict, *, step: int) -> None:
    """Periodically log input global_scale stats and drift vs post-calib baseline."""
    global _input_global_scale_baseline
    if step % _scale_monitor_every() != 0:
        return
    try:
        from vexact.quantization.scale_monitor import (
            _NVFP4_GLOBAL_SCALE_DENOM,
            amax_to_global_scale,
        )
    except Exception:  # pragma: no cover
        return

    scales: list[float] = []
    current: dict[str, float] = {}
    for name, iq in iq_map.items():
        amax = getattr(iq, "amax", None)
        if amax is None:
            continue
        scale = float(amax_to_global_scale(amax).amax().item())
        current[name] = scale
        scales.append(scale)

    if not scales:
        return

    import statistics as st

    logger.info(
        "[vexact-qat] step=%s input global_scale[min/mean/max]=%.6g/%.6g/%.6g (n=%d)",
        step,
        min(scales),
        st.mean(scales),
        max(scales),
        len(scales),
    )

    if _input_global_scale_baseline is None:
        _input_global_scale_baseline = dict(current)
        logger.info(
            "[vexact-qat] step=%s recorded input global_scale baseline (n=%d, denom=%.1f)",
            step,
            len(current),
            _NVFP4_GLOBAL_SCALE_DENOM,
        )
        return

    drifted = 0
    max_rel = 0.0
    for name, scale in current.items():
        base = _input_global_scale_baseline.get(name)
        if base is None or base == 0.0:
            continue
        rel = abs(scale - base) / abs(base)
        max_rel = max(max_rel, rel)
        if rel > 1e-5:
            drifted += 1
    if drifted:
        logger.warning(
            "[vexact-qat] step=%s input global_scale drift vs baseline: "
            "changed=%d/%d max_rel=%.3e (expected frozen after calib)",
            step,
            drifted,
            len(current),
            max_rel,
        )
    else:
        logger.info(
            "[vexact-qat] step=%s input global_scale frozen vs baseline (ok)",
            step,
        )


class ServerAdapter(BaseRollout):
    """
    VeXact server adapter for async mode, serves as a client to request VeXact server
    to resume/release/update_weights.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None  # Lazy

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        assert rollout_world_size == device_mesh.size() // device_mesh["dp"].size()
        assert self.replica_rank == device_mesh["dp"].get_local_rank()

        # ZMQ handle for weight transfer, must match Worker.receive_weights /
        # VeXactServer driver_id (role-qualified student names).
        from vexact.integrations.verl.role_names import build_vexact_role_names

        driver_id = build_vexact_role_names(
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            is_teacher_model=False,
            is_reward_model=False,
            name_suffix="",
        )["driver_id"]
        self.zmq_handle = f"ipc:///tmp/vexact-weight-{driver_id}-{self.rollout_rank}.sock"
        logger.info(f"Sender:{self.zmq_handle=}:{get_torch_device().get_device_properties(get_device_id()).uuid}")

        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning("IPC is not supported on your devices. Falling back to shared memory for weight transfer.")

    def _get_server_handle(self) -> ray.actor.ActorHandle:
        """Lazy init server handle because server is launched after hybrid engine."""
        if self.server_handle is None:
            # Async server handle, must match async server ray actor name
            from vexact.integrations.verl.role_names import build_vexact_role_names

            server_name = build_vexact_role_names(
                replica_rank=self.replica_rank,
                node_rank=self.node_rank,
                is_teacher_model=False,
                is_reward_model=False,
                name_suffix="",
            )["server_name"]
            self.server_handle = ray.get_actor(server_name)
        return self.server_handle

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: List of resource tags to resume (e.g. ["weights", "kv_cache"]).
        """
        if self.config.free_cache_engine and self.rollout_rank == 0:
            await self._get_server_handle().wake_up.remote(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine and self.rollout_rank == 0:
            await self._get_server_handle().sleep.remote()

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Update model weights via bucketed IPC transfer to inference workers.

        When QAT is enabled and the training-side weight_quantizer_map is cached,
        weights are folded via the actor's weight_quantizer before being sent to
        rollout (NeMo-RL QARL-style). For w4a4, enabled ``input_quantizer.amax``
        buffers are appended so rollout activation fake-quant matches training.
        """
        from vexact.quantization.amax_sync import (
            chain_weights_with_input_amax,
            get_training_input_quantizer_map,
        )
        from vexact.quantization.fold import (
            fold_weights_generator,
            get_training_weight_quantizer_map,
        )

        wq_map = get_training_weight_quantizer_map()
        iq_map = get_training_input_quantizer_map()
        qat_active = bool(wq_map) or bool(iq_map)
        log_qat = qat_active and _is_global_rank0()
        # Always collect stats when QAT is active so legacy per-rank INFO logs are
        # suppressed; only RANK 0 emits the step-level summary below.
        fold_stats: dict[str, int] = {}
        amax_stats: dict[str, int] = {}
        step = 0
        mode = ""
        t0 = 0.0

        if log_qat:
            step = _resolve_qat_step(kwargs, self)
            mode = _qat_mode_label(has_iq_map=bool(iq_map))
            t0 = time.perf_counter()
            logger.info("[vexact-qat] step=%s update_weights begin (mode=%s)", step, mode)

        if wq_map:
            weights = fold_weights_generator(
                weights, wq_map, stats=fold_stats if qat_active else None
            )

        if iq_map:
            weights = chain_weights_with_input_amax(
                weights, iq_map, stats=amax_stats if qat_active else None
            )

        future = None
        if self.rollout_rank == 0:
            future = self._get_server_handle().receive_weights.remote()

        from .bucketed_weight_transfer import BucketedWeightSender

        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            device=torch.device(f"{get_device_name()}:{get_device_id()}"),
            use_shm=self.use_shm,
        )
        sender.send_weights(weights)

        receive_result = None
        if future is not None:
            receive_result = await future

        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()

        if log_qat:
            if fold_stats:
                logger.info(
                    "[vexact-qat] step=%s weight fold done: folded=%d/%d, cast_fp32_to_bf16=%d",
                    step,
                    fold_stats.get("folded", 0),
                    fold_stats.get("total", 0),
                    fold_stats.get("cast_fp32_to_bf16", 0),
                )
            if iq_map:
                attached = amax_stats.get("attached", 0)
                applied = _amax_applied_from_receive_result(receive_result)
                # If this rank did not own the receive future, fall back to attached.
                if applied == 0 and attached > 0 and receive_result is None:
                    applied = attached
                logger.info(
                    "[vexact-qat] step=%s amax sync done: attached=%d, applied=%d (max-merge)",
                    step,
                    attached,
                    applied,
                )
                _maybe_log_input_global_scale_drift(iq_map, step=step)
            elapsed = time.perf_counter() - t0
            logger.info(
                "[vexact-qat] step=%s update_weights end (elapsed=%.2fs)",
                step,
                elapsed,
            )

    def generate_sequences(self, prompts):
        """Sync generation no longer supported."""
        raise NotImplementedError(
            "ServerAdapter does not support synchronous generate_sequences(). "
            "Use the async server interface via VeXactReplica and VeXactServer instead."
        )
