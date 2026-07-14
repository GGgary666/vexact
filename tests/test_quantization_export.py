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

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vexact.quantization.export import validate_nvfp4_export_dir


def test_validate_nvfp4_export_dir_ok(tmp_path: Path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "W4A16_NVFP4", "group_size": 16}})
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    # Empty safetensors file is enough if we mock the key scan.
    st = tmp_path / "model.safetensors"
    st.write_bytes(b"not-a-real-safetensor")

    with patch(
        "vexact.quantization.export._collect_safetensor_keys_ending",
        return_value=[],
    ):
        summary = validate_nvfp4_export_dir(tmp_path, require_input_scale=False, mode="w4a16")
    assert summary["quant_algo"] == "W4A16_NVFP4"
    assert summary["safetensors"] == 1


def test_validate_nvfp4_export_dir_requires_input_scale(tmp_path: Path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4"}})
    )
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")

    with patch(
        "vexact.quantization.export._collect_safetensor_keys_ending",
        return_value=[],
    ):
        with pytest.raises(RuntimeError, match="input_scale"):
            validate_nvfp4_export_dir(tmp_path, require_input_scale=True, mode="w4a4")


def test_validate_nvfp4_export_dir_w4a16_rejects_input_scale(tmp_path: Path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "W4A16_NVFP4"}})
    )
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")

    with patch(
        "vexact.quantization.export._collect_safetensor_keys_ending",
        return_value=["model.layers.0.mlp.up_proj.input_scale"],
    ):
        with pytest.raises(RuntimeError, match="must not contain"):
            validate_nvfp4_export_dir(tmp_path, require_input_scale=False, mode="w4a16")


def test_validate_nvfp4_export_dir_wrong_algo(tmp_path: Path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "FP8"}})
    )
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")

    with patch(
        "vexact.quantization.export._collect_safetensor_keys_ending",
        return_value=[],
    ):
        with pytest.raises(RuntimeError, match="W4A16_NVFP4"):
            validate_nvfp4_export_dir(tmp_path, require_input_scale=False, mode="w4a16")


def test_rewrite_nvfp4_quant_algo_for_w4a16(tmp_path: Path):
    from vexact.quantization.export import rewrite_nvfp4_quant_algo_for_mode

    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "kv_cache_quant_algo": None,
                    "exclude_modules": ["lm_head"],
                },
            }
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "quantization_config": {
                    "quant_algo": "NVFP4",
                    "config_groups": {
                        "group_0": {
                            "input_activations": {"num_bits": 4},
                            "weights": {"num_bits": 4},
                        }
                    },
                },
            }
        )
    )

    algo = rewrite_nvfp4_quant_algo_for_mode(tmp_path, "w4a16")
    assert algo == "W4A16_NVFP4"

    hf_q = json.loads((tmp_path / "hf_quant_config.json").read_text())
    assert hf_q["quantization"]["quant_algo"] == "W4A16_NVFP4"

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["quantization_config"]["quant_algo"] == "W4A16_NVFP4"
    assert "input_activations" not in cfg["quantization_config"]["config_groups"]["group_0"]
    assert "weights" in cfg["quantization_config"]["config_groups"]["group_0"]


def test_rewrite_nvfp4_quant_algo_for_w4a4_keeps_nvfp4(tmp_path: Path):
    from vexact.quantization.export import rewrite_nvfp4_quant_algo_for_mode

    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "group_size": 16}})
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

    algo = rewrite_nvfp4_quant_algo_for_mode(tmp_path, "w4a4")
    assert algo == "NVFP4"
    hf_q = json.loads((tmp_path / "hf_quant_config.json").read_text())
    assert hf_q["quantization"]["quant_algo"] == "NVFP4"

def test_prepare_model_for_nvfp4_export_requires_enable():
    from vexact.quantization import QATConfig
    from vexact.quantization.export import prepare_model_for_nvfp4_export

    with pytest.raises(ValueError, match="enable=True"):
        prepare_model_for_nvfp4_export(MagicMock(), QATConfig(enable=False, mode="w4a16"))


def test_normalize_mesh_dim_names_veomni_dp_shard():
    from vexact.quantization.export import normalize_mesh_dim_names

    assert normalize_mesh_dim_names(("dp_shard",)) == ("fsdp",)
    assert normalize_mesh_dim_names(("dp_replicate", "dp_shard")) == ("ddp", "fsdp")
    assert normalize_mesh_dim_names(("fsdp",)) == ("fsdp",)


def test_merge_plain_tensor_shards_handles_scalars_and_replicas():
    import torch

    from vexact.quantization.export import merge_plain_tensor_shards

    scalar = torch.tensor(1.5)
    assert torch.equal(merge_plain_tensor_shards([scalar, scalar.clone()]), scalar)

    replicated = torch.randn(4, 8)
    out = merge_plain_tensor_shards([replicated, replicated.clone()])
    assert torch.equal(out, replicated)

    shards = [torch.randn(2, 8), torch.randn(3, 8)]
    cat = merge_plain_tensor_shards(shards)
    assert cat.shape == (5, 8)


def test_filter_hf_weight_state_dict_drops_quantizer_keys():
    import torch

    from vexact.quantization.export import filter_hf_weight_state_dict, is_modelopt_extra_state_key

    assert is_modelopt_extra_state_key("model.layers.0.mlp.up_proj.input_quantizer._amax")
    assert not is_modelopt_extra_state_key("model.layers.0.mlp.up_proj.weight")

    filtered = filter_hf_weight_state_dict(
        {
            "model.layers.0.mlp.up_proj.weight": torch.zeros(2, 2),
            "model.layers.0.mlp.up_proj.weight_quantizer._amax": torch.tensor(1.0),
        }
    )
    assert list(filtered.keys()) == ["model.layers.0.mlp.up_proj.weight"]


def test_extract_and_merge_input_amax_shards():
    import torch

    from vexact.quantization.export import (
        extract_input_amax_from_state_dict,
        merge_input_amax_shards,
    )

    sd = {
        "model.layers.0.mlp.up_proj.weight": torch.zeros(2, 2),
        "model.layers.0.mlp.up_proj.input_quantizer._amax": torch.tensor(1.25),
        "model.layers.0.mlp.up_proj.weight_quantizer._amax": torch.tensor(9.0),
        "foo.input_quantizer.amax": torch.tensor([2.0, 3.0]),
    }
    extracted = extract_input_amax_from_state_dict(sd)
    assert set(extracted) == {
        "model.layers.0.mlp.up_proj.input_quantizer._amax",
        "foo.input_quantizer.amax",
    }
    assert "weight_quantizer" not in "".join(extracted)

    # Same shape, different values -> torch.maximum
    rank0 = {
        "layer.input_quantizer._amax": torch.tensor([1.0, 4.0]),
        "other.input_quantizer._amax": torch.tensor(2.0),
    }
    rank1 = {
        "layer.input_quantizer._amax": torch.tensor([3.0, 2.0]),
        "other.input_quantizer._amax": torch.tensor(2.0),
    }
    merged = merge_input_amax_shards([rank0, rank1])
    assert torch.equal(merged["layer.input_quantizer._amax"], torch.tensor([3.0, 4.0]))
    assert torch.equal(merged["other.input_quantizer._amax"], torch.tensor(2.0))


def test_prepare_model_for_nvfp4_export_restores_input_amax():
    """When input_amax is provided, final buffers equal injected values (mocked QAT)."""
    import torch
    from torch import nn

    from vexact.quantization import QATConfig
    from vexact.quantization.export import prepare_model_for_nvfp4_export

    class _IQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_amax", torch.tensor([0.1]))

        @property
        def amax(self):
            return self._amax

        @property
        def is_enabled(self):
            return True

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_quantizer = _IQ()

    model = _M()
    injected = {"input_quantizer._amax": torch.tensor([7.5])}
    fake_iq = model.input_quantizer

    with (
        patch("vexact.quantization.export.quantize_model", side_effect=lambda m, c, forward_loop=None: m),
        patch(
            "vexact.quantization.export.build_input_quantizer_map",
            return_value={"input_quantizer": fake_iq},
        ),
        patch(
            "vexact.quantization.amax_sync.build_input_quantizer_map",
            return_value={"input_quantizer": fake_iq},
        ),
        patch("vexact.quantization.export.assert_input_amax_materialized", return_value=1),
    ):
        prepare_model_for_nvfp4_export(
            model,
            QATConfig(enable=True, mode="w4a4"),
            input_amax=injected,
        )

    assert torch.equal(model.input_quantizer._amax, torch.tensor([7.5]))


def test_prepare_model_for_nvfp4_export_rejects_empty_input_amax():
    from vexact.quantization import QATConfig
    from vexact.quantization.export import prepare_model_for_nvfp4_export

    with patch("vexact.quantization.export.quantize_model", side_effect=lambda m, c, forward_loop=None: m):
        with pytest.raises(RuntimeError, match="empty"):
            prepare_model_for_nvfp4_export(
                MagicMock(),
                QATConfig(enable=True, mode="w4a16", calibrate=False),
                input_amax={},
            )
