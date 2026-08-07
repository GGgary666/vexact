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

"""CLI: export a BF16 VeOmni/verl FSDP actor checkpoint to HuggingFace format.

For non-QAT runs (e.g. ``exp_scripts/.../run_qwen3_1b7_gsm8k.sh``). Stock
``python -m verl.model_merger`` rejects VeOmni mesh names ``('dp_shard',)``;
this script reuses the VeOmni-aware merge from ``export_qat_to_hf.py``.

Example::

    scripts/export_bf16_to_hf.sh \\
      --actor-ckpt /path/to/global_step_N/actor \\
      --export-dir /path/to/hf_bf16 \\
      -v

QAT / NVFP4 export is separate: ``scripts/export_qat_to_hf.sh``.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


def _load_run_fsdp_merge():
    """Load ``_run_fsdp_merge`` from sibling ``export_qat_to_hf.py`` (not a package)."""
    sibling = Path(__file__).resolve().parent / "export_qat_to_hf.py"
    if not sibling.is_file():
        raise SystemExit(f"Missing sibling merge helper: {sibling}")
    spec = importlib.util.spec_from_file_location("_vexact_export_qat_to_hf", sibling)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load {sibling}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / nested classes resolve ``__module__``.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._run_fsdp_merge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor-ckpt",
        type=Path,
        required=True,
        help="verl/VeOmni actor checkpoint dir (…/global_step_N/actor)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Output HuggingFace BF16 directory",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to the HF save path (default: true)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    actor_ckpt = args.actor_ckpt.resolve()
    export_dir = args.export_dir.resolve()
    if not actor_ckpt.is_dir():
        raise SystemExit(f"--actor-ckpt is not a directory: {actor_ckpt}")
    hf_cfg = actor_ckpt / "huggingface"
    if not hf_cfg.is_dir():
        raise SystemExit(
            f"Expected HF config sidecar at {hf_cfg} (verl actor checkpoint layout)."
        )

    run_fsdp_merge = _load_run_fsdp_merge()
    logger.info("Merging VeOmni/FSDP actor ckpt -> HF bf16")
    logger.info("  actor:  %s", actor_ckpt)
    logger.info("  export: %s", export_dir)
    run_fsdp_merge(actor_ckpt, export_dir, trust_remote_code=args.trust_remote_code)
    logger.info(
        "Done. Load with AutoModelForCausalLM.from_pretrained(%r) "
        "(plain BF16; no --quantization).",
        str(export_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
