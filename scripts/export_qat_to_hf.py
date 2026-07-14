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

"""CLI: export a QAT / FSDP actor checkpoint to a real NVFP4 HuggingFace model.

Pipeline:
  1. Optional FSDP/VeOmni shard merge -> HF bf16 + extract ``input_amax.pt``
  2. Load HF bf16 weights
  3. Insert QuantModules; for w4a4 restore ``input_quantizer.amax`` from actor
     ckpt (exact copy; NeMo-RL style). Random calib is opt-in only via
     ``--allow-random-calib``.
  4. ``export_hf_checkpoint`` -> packed NVFP4 + hf_quant_config.json

Deploy with::

    python -m vllm.entrypoints.openai.api_server \\
        --model <export_dir> --quantization modelopt_fp4

w4a4 (default, matches training amax)::

    scripts/export_qat_to_hf.sh \\
      --mode w4a4 \\
      --actor-ckpt /path/to/global_step_N/actor \\
      --export-dir /path/to/nvfp4_hf \\
      --keep-merged -v

Already-merged HF bf16 (reuses ``<hf-bf16>/input_amax.pt`` or ``--amax-path``)::

    scripts/export_qat_to_hf.sh \\
      --mode w4a4 --hf-bf16 /path/to/hf_bf16 \\
      --amax-path /path/to/input_amax.pt \\
      --export-dir /path/to/nvfp4_hf

w4a16 (no input amax)::

    scripts/export_qat_to_hf.sh --mode w4a16 --hf-bf16 ... --export-dir ...

Note: real NVFP4 kernels typically require Blackwell (SM >= 10) GPUs.
Online fold BF16 weights are NOT a substitute for this export.

VeOmni checkpoints use DTensor mesh_dim_names ``('dp_shard',)``, which stock
``verl.model_merger`` rejects (it only allows ``fsdp`` / ``ddp+fsdp``). This
CLI normalizes those names before merging.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vexact.quantization import QATConfig
from vexact.quantization.amax_sync import is_input_amax_key
from vexact.quantization.export import (
    export_nvfp4_hf_checkpoint,
    filter_hf_weight_state_dict,
    is_modelopt_extra_state_key,
    merge_input_amax_shards,
    merge_plain_tensor_shards,
    normalize_mesh_dim_names,
    prepare_model_for_nvfp4_export,
)


logger = logging.getLogger(__name__)

INPUT_AMAX_FILENAME = "input_amax.pt"


def _run_fsdp_merge(
    actor_ckpt: Path,
    target_dir: Path,
    *,
    trust_remote_code: bool = True,
) -> Optional[dict[str, torch.Tensor]]:
    """Merge sharded actor ckpt to HF bf16; return merged input amax (or None).

    Also writes ``{target_dir}/input_amax.pt`` when any input amax was found.
    """
    import os as _os
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover
        from torch.distributed._tensor import DTensor

    # Bind helpers as defaults: nested class methods use LOAD_GLOBAL and do not
    # close over the enclosing function / from-imports reliably across envs.
    _is_extra = is_modelopt_extra_state_key
    _is_iq_amax = is_input_amax_key
    _merge_plain = merge_plain_tensor_shards
    _merge_amax = merge_input_amax_shards
    _filter_hf = filter_hf_weight_state_dict
    _normalize_mesh = normalize_mesh_dim_names

    target_dir.mkdir(parents=True, exist_ok=True)
    # Captured by merge_and_save after _load_and_merge_state_dicts.
    extracted_input_amax: dict[str, torch.Tensor] = {}

    class VeOmniAwareFSDPModelMerger(FSDPModelMerger):
        def _merge_by_placement(self, tensors, placement):
            # 0-dim replicated buffers can show up as Shard/Replicate DTensors.
            if tensors and tensors[0].ndim == 0:
                return tensors[0]
            return super()._merge_by_placement(tensors, placement)

        def _load_and_merge_state_dicts(
            self,
            world_size: int,
            total_shards: int,
            mesh_shape: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            *,
            _is_extra_key=_is_extra,
            _is_input_amax=_is_iq_amax,
            _merge_plain_shards=_merge_plain,
            _merge_amax_shards=_merge_amax,
            _filter_state=_filter_hf,
        ) -> dict[str, torch.Tensor]:
            model_state_dict_lst: list = [None] * total_shards

            def process_one_shard(rank: int, out_list: list):
                model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
                out_list[rank] = torch.load(model_path, map_location="cpu", weights_only=False)

            with ThreadPoolExecutor(max_workers=min(32, _os.cpu_count() or 8)) as executor:
                futures = [
                    executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)
                ]
                for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                    future.result()

            state_dict: dict[str, list] = {}
            param_placements: dict[str, tuple] = {}
            amax_per_rank: list[dict[str, torch.Tensor]] = [{} for _ in range(total_shards)]

            for key in set(model_state_dict_lst[0].keys()):
                # Keep input_quantizer amax for export restore; drop other ModelOpt extras.
                if _is_input_amax(key):
                    for rank, model_state_shard in enumerate(model_state_dict_lst):
                        tensor = model_state_shard.pop(key, None)
                        if tensor is None:
                            continue
                        if isinstance(tensor, DTensor):
                            tensor = tensor._local_tensor
                        amax_per_rank[rank][key] = tensor.detach().cpu()
                    continue
                if _is_extra_key(key):
                    for model_state_shard in model_state_dict_lst:
                        model_state_shard.pop(key, None)
                    continue
                state_dict[key] = []
                for model_state_shard in model_state_dict_lst:
                    tensor = model_state_shard.pop(key)
                    if isinstance(tensor, DTensor):
                        state_dict[key].append(tensor._local_tensor.bfloat16())
                        placements = tuple(tensor.placements)
                        if mesh_dim_names[0] in ("dp", "ddp"):
                            placements = placements[1:]
                        if key not in param_placements:
                            param_placements[key] = placements
                        else:
                            assert param_placements[key] == placements
                    else:
                        # Keep original dtype for 0-dim amax-like buffers; cast weights.
                        if tensor.ndim == 0:
                            state_dict[key].append(tensor.detach().cpu())
                        else:
                            state_dict[key].append(tensor.bfloat16())

            del model_state_dict_lst

            for key in sorted(state_dict):
                shards = state_dict[key]
                if key in param_placements:
                    placements = param_placements[key]
                    if len(mesh_shape) == 1:
                        assert len(placements) == 1
                        state_dict[key] = self._merge_by_placement(shards, placements[0])
                    else:
                        raise NotImplementedError("FSDP + TP is not supported yet")
                else:
                    state_dict[key] = _merge_plain_shards(shards)

            merged_amax = _merge_amax_shards(amax_per_rank)
            extracted_input_amax.clear()
            extracted_input_amax.update(merged_amax)
            if merged_amax:
                logger.info(
                    "Extracted %d input_quantizer amax key(s) from actor ckpt for export restore.",
                    len(merged_amax),
                )
            else:
                logger.warning(
                    "No input_quantizer amax found in actor shards "
                    "(expected for w4a16; required for w4a4 unless --allow-random-calib)."
                )

            # Defensive: drop any remaining ModelOpt extras before HF save.
            return _filter_state(state_dict)

        def merge_and_save(self, *, _normalize_mesh_names=_normalize_mesh) -> None:
            world_size = self._get_world_size()
            rank_zero_state_dict = self._load_rank_zero_state_dict(world_size)
            mesh, mesh_dim_names = self._extract_device_mesh_info(rank_zero_state_dict, world_size)
            raw_names = tuple(mesh_dim_names)
            mesh_dim_names = _normalize_mesh_names(raw_names)
            if mesh_dim_names != raw_names:
                logger.info(
                    "Normalized VeOmni mesh_dim_names %s -> %s for verl FSDP merger",
                    raw_names,
                    mesh_dim_names,
                )
            logger.info("Got device mesh %s, mesh_dim_names %s", mesh, mesh_dim_names)

            total_shards, mesh_shape = self._calculate_shard_configuration(mesh, mesh_dim_names)
            logger.info("Processing model shards with %s %s in total", total_shards, mesh_shape)

            merged_state_dict = self._load_and_merge_state_dicts(
                world_size, total_shards, mesh_shape, mesh_dim_names
            )

            if self.config.operation == "test":
                if not self.config.test_hf_dir:
                    raise ValueError("test_hf_dir must be provided for test operation")
                self._validate_state_dict(merged_state_dict)
            elif self.config.operation == "merge":
                self.save_hf_model_and_tokenizer(merged_state_dict)
                if extracted_input_amax:
                    amax_path = Path(self.config.target_dir) / INPUT_AMAX_FILENAME
                    torch.save(extracted_input_amax, amax_path)
                    logger.info("Wrote input amax sidecar: %s (%d keys)", amax_path, len(extracted_input_amax))
                if self.config.hf_upload:
                    self.upload_to_huggingface()
            else:
                raise ValueError(f"Unknown operation: {self.config.operation}")

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(actor_ckpt),
        target_dir=str(target_dir),
        hf_model_config_path=os.path.join(str(actor_ckpt), "huggingface"),
        trust_remote_code=trust_remote_code,
    )
    logger.info("Running VeOmni-aware FSDP merge: %s -> %s", actor_ckpt, target_dir)
    merger = VeOmniAwareFSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()
    return dict(extracted_input_amax) if extracted_input_amax else None


def _load_input_amax(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise SystemExit(f"--amax-path must be a dict[str, Tensor], got {type(obj)} from {path}")
    out: dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not is_input_amax_key(str(key)):
            continue
        if isinstance(value, torch.Tensor):
            out[str(key)] = value.detach().cpu()
    if not out:
        raise SystemExit(f"No input_quantizer amax keys found in {path}")
    return out


def _resolve_input_amax(
    *,
    mode: str,
    amax_path: Optional[Path],
    hf_bf16: Path,
    merged_amax: Optional[dict[str, torch.Tensor]],
    allow_random_calib: bool,
) -> Optional[dict[str, torch.Tensor]]:
    """Resolve training amax for export; enforce w4a4 defaults."""
    if mode != "w4a4":
        return None

    if merged_amax:
        return merged_amax

    candidates: list[Path] = []
    if amax_path is not None:
        candidates.append(amax_path)
    else:
        candidates.append(hf_bf16 / INPUT_AMAX_FILENAME)

    for path in candidates:
        if path.is_file():
            logger.info("Loading input amax from %s", path)
            return _load_input_amax(path)

    if allow_random_calib:
        logger.warning(
            "w4a4 export: no input_amax found; using random calibration "
            "(--allow-random-calib). Exported input_scale will NOT match training."
        )
        return None

    searched = ", ".join(str(p) for p in candidates)
    raise SystemExit(
        "w4a4 export requires input_quantizer.amax from the actor checkpoint.\n"
        f"  Looked for: {searched}\n"
        "  Re-run with --actor-ckpt (extracts input_amax.pt), or pass --amax-path.\n"
        "  Only for debugging: --allow-random-calib (random calib; not training-faithful)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--actor-ckpt",
        type=Path,
        help="verl/VeOmni actor checkpoint dir (merges shards to HF bf16 + extracts amax)",
    )
    src.add_argument(
        "--hf-bf16",
        type=Path,
        help="Already-merged HuggingFace bf16 directory",
    )
    parser.add_argument(
        "--mode",
        choices=("w4a4", "w4a16"),
        required=True,
        help="QAT mode used during training (must match)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Tokenizer / config source (defaults to --hf-bf16 or merge output)",
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--calib-size", type=int, default=8)
    parser.add_argument("--calib-seq-len", type=int, default=32)
    parser.add_argument(
        "--amax-path",
        type=Path,
        default=None,
        help="Path to input_amax.pt (default: <hf-bf16>/input_amax.pt after merge)",
    )
    parser.add_argument(
        "--allow-random-calib",
        action="store_true",
        help="w4a4 only: allow random-token calib when amax is missing (not training-faithful)",
    )
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep temporary FSDP merge directory under export-dir/_merged_bf16",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    dtype = getattr(torch, args.dtype)
    tmp_ctx = None
    try:
        merged_amax: Optional[dict[str, torch.Tensor]] = None
        if args.actor_ckpt is not None:
            if args.keep_merged:
                hf_bf16 = args.export_dir / "_merged_bf16"
                args.export_dir.mkdir(parents=True, exist_ok=True)
            else:
                tmp_ctx = tempfile.TemporaryDirectory(prefix="vexact_qat_merge_")
                hf_bf16 = Path(tmp_ctx.name) / "hf_bf16"
            merged_amax = _run_fsdp_merge(args.actor_ckpt, hf_bf16)
        else:
            hf_bf16 = args.hf_bf16

        input_amax = _resolve_input_amax(
            mode=args.mode,
            amax_path=args.amax_path,
            hf_bf16=hf_bf16,
            merged_amax=merged_amax,
            allow_random_calib=args.allow_random_calib,
        )

        base = args.base_model or str(hf_bf16)
        qat = QATConfig(
            enable=True,
            mode=args.mode,
            # w4a4 defaults/forces True; w4a16 defaults False via QATConfig.
            # Even with ckpt amax restore we still need a short calib pass so
            # ModelOpt materializes amax buffers before exact overwrite.
            calibrate=(True if args.mode == "w4a4" else False),
            calib_size=args.calib_size,
            calib_seq_len=args.calib_seq_len,
        )

        logger.info("Loading HF model from %s (dtype=%s)", hf_bf16, args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            str(hf_bf16),
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

        prepare_model_for_nvfp4_export(model, qat, input_amax=input_amax)
        export_nvfp4_hf_checkpoint(
            model,
            args.export_dir,
            tokenizer=tokenizer,
            dtype=dtype,
            require_input_scale=(args.mode == "w4a4"),
            mode=args.mode,
        )
        logger.info(
            "Done. Serve with: python -m vllm.entrypoints.openai.api_server "
            "--model %s --quantization modelopt_fp4 "
            "(w4a16 expects hf_quant_config quant_algo=W4A16_NVFP4)",
            args.export_dir,
        )
        return 0
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
