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

"""Tests for configurable W4A4 dataset calibration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vexact.quantization import QATConfig
from vexact.quantization.quantize import (
    _extract_jsonl_text,
    _load_jsonl_texts,
    build_calibration_forward_loop,
    resolve_calibration_forward_loop,
)


def test_qat_config_calib_fields_from_env(monkeypatch):
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a4")
    monkeypatch.setenv("VEXACT_QAT_CALIB_DATA", "/tmp/calib.jsonl")
    monkeypatch.setenv("VEXACT_QAT_CALIB_TOKENIZER", "/tmp/tok")
    monkeypatch.setenv("VEXACT_QAT_CALIB_SIZE", "128")
    monkeypatch.setenv("VEXACT_QAT_CALIB_SEQ_LEN", "1024")
    monkeypatch.setenv("VEXACT_QAT_CALIB_BATCH_SIZE", "2")
    cfg = QATConfig.from_env()
    assert cfg.calib_data == "/tmp/calib.jsonl"
    assert cfg.calib_tokenizer == "/tmp/tok"
    assert cfg.calib_size == 128
    assert cfg.calib_seq_len == 1024
    assert cfg.calib_batch_size == 2
    assert cfg.uses_dataset_calibration is True


def test_uses_dataset_calibration_false_for_random():
    cfg = QATConfig(enable=True, mode="w4a4", calib_data="random")
    assert cfg.uses_dataset_calibration is False
    cfg2 = QATConfig(enable=True, mode="w4a4", calib_data=None)
    assert cfg2.uses_dataset_calibration is False


def test_qat_config_rejects_non_positive_calib_knobs():
    with pytest.raises(ValueError, match="calib_size"):
        QATConfig(enable=True, mode="w4a4", calib_size=0)
    with pytest.raises(ValueError, match="calib_batch_size"):
        QATConfig(enable=True, mode="w4a4", calib_batch_size=-1)


def test_extract_jsonl_text_prefers_known_keys():
    assert _extract_jsonl_text({"article": "hello world article"}) == "hello world article"
    assert _extract_jsonl_text({"text": "abc", "article": "ignored"}) == "abc"
    assert _extract_jsonl_text({"foo": 1}) is None


def test_load_jsonl_texts_respects_max_samples(tmp_path: Path):
    path = tmp_path / "calib.jsonl"
    rows = [{"text": f"sample {i} with enough chars"} for i in range(10)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    texts = _load_jsonl_texts(str(path), max_samples=3)
    assert len(texts) == 3
    assert texts[0].startswith("sample 0")


def test_load_jsonl_texts_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_jsonl_texts("/no/such/calib.jsonl", max_samples=1)


def test_load_jsonl_texts_empty_raises(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text('{"id": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="No usable text"):
        _load_jsonl_texts(str(path), max_samples=8)


def test_jsonl_calibration_forward_loop_runs_and_restores_attn(tmp_path: Path):
    path = tmp_path / "calib.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": f"calibration sentence number {i}"}) for i in range(4))
        + "\n",
        encoding="utf-8",
    )

    class _Tok:
        pad_token_id = 0
        eos_token = "<eos>"
        pad_token = None
        padding_side = "right"

        def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
            ids = torch.ones(len(texts), min(max_length or 8, 8), dtype=torch.long)
            mask = torch.ones_like(ids)
            return {"input_ids": ids, "attention_mask": mask}

    class _Cfg:
        def __init__(self):
            self._attn_implementation = "fa-invariant"

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.w = nn.Parameter(torch.zeros(1))
            self.calls = []

        def forward(self, input_ids=None, attention_mask=None, use_cache=None, **kwargs):
            self.calls.append(
                {
                    "attn": self.config._attn_implementation,
                    "bsz": int(input_ids.shape[0]),
                    "use_cache": use_cache,
                }
            )
            return input_ids

    cfg = QATConfig(
        enable=True,
        mode="w4a4",
        calib_data=str(path),
        calib_tokenizer="unused",
        calib_size=4,
        calib_seq_len=8,
        calib_batch_size=2,
    )
    with patch("vexact.quantization.quantize._load_tokenizer", return_value=_Tok()):
        loop = build_calibration_forward_loop(cfg)
        model = _Model()
        loop(model)

    assert len(model.calls) == 2  # 4 samples / batch 2
    assert all(c["attn"] == "eager" for c in model.calls)
    assert all(c["use_cache"] is False for c in model.calls)
    assert model.config._attn_implementation == "fa-invariant"


def test_jsonl_calibration_requires_tokenizer(tmp_path: Path):
    path = tmp_path / "calib.jsonl"
    path.write_text(json.dumps({"text": "enough characters here"}) + "\n", encoding="utf-8")
    cfg = QATConfig(enable=True, mode="w4a4", calib_data=str(path), calib_tokenizer=None)
    with pytest.raises(ValueError, match="calib_tokenizer"):
        build_calibration_forward_loop(cfg)


def test_resolve_calibration_forward_loop_requires_dataset_by_default():
    cfg = QATConfig(enable=True, mode="w4a4", calib_data=None)
    with pytest.raises(ValueError, match="real calib dataset"):
        resolve_calibration_forward_loop(cfg)


def test_resolve_calibration_forward_loop_random_opt_in():
    cfg = QATConfig(enable=True, mode="w4a4", calib_data="random")
    loop = resolve_calibration_forward_loop(cfg)
    assert callable(loop)


def test_rollout_qat_forces_random_calib_when_hydra_omits_calib_data():
    """Mirror model_loader rollout path: Hydra qat often has no calib_data."""
    from dataclasses import replace

    hydra_qat = QATConfig(enable=True, mode="w4a4", calibrate=True)
    assert hydra_qat.calib_data is None
    with pytest.raises(ValueError, match="requires a real calib dataset"):
        resolve_calibration_forward_loop(hydra_qat)

    rollout_qat = replace(hydra_qat, calib_data="random", allow_random_calib=True)
    loop = resolve_calibration_forward_loop(rollout_qat)
    assert callable(loop)


def test_resolve_calibration_forward_loop_allow_random_flag():
    cfg = QATConfig(enable=True, mode="w4a4", calib_data=None, allow_random_calib=True)
    loop = resolve_calibration_forward_loop(cfg)
    assert callable(loop)


def test_allow_random_calib_from_env(monkeypatch):
    monkeypatch.setenv("VEXACT_QAT_ENABLE", "1")
    monkeypatch.setenv("VEXACT_QAT_MODE", "w4a4")
    monkeypatch.setenv("VEXACT_QAT_ALLOW_RANDOM_CALIB", "1")
    cfg = QATConfig.from_env()
    assert cfg.allow_random_calib is True


def test_resolve_calibration_forward_loop_disabled():
    cfg = QATConfig(enable=True, mode="w4a16", calibrate=False)
    assert resolve_calibration_forward_loop(cfg) is None


def test_named_dataset_missing_raises_clear_error():
    cfg = QATConfig(
        enable=True,
        mode="w4a4",
        calib_data="not_a_real_dataset_or_file",
        calib_tokenizer="/tmp/tok",
        calib_size=1,
    )

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

    with patch("vexact.quantization.quantize._try_modelopt_dataset_dataloader", return_value=None):
        loop = build_calibration_forward_loop(cfg)
        with pytest.raises(FileNotFoundError, match="neither an existing file"):
            loop(_Model())


def test_caller_provided_forward_loop_wins():
    cfg = QATConfig(enable=True, mode="w4a4", calib_data="/tmp/x.jsonl")
    sentinel = MagicMock()
    assert resolve_calibration_forward_loop(cfg, forward_loop=sentinel) is sentinel
