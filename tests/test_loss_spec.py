"""Tests for LossSpec presets."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.trainer import LossSpec  # noqa: E402


def test_pretrain_spec():
    s = LossSpec.pretrain()
    assert s.use_mlm and not s.use_result and not s.use_score
    assert s.lambda_mlm == 1.0


def test_finetune_multitask_spec():
    s = LossSpec.finetune_multitask()
    assert s.use_mlm and s.use_result and s.use_score
    assert s.lambda_result == 1.0 and s.lambda_score == 0.3


def test_finetune_score_primary_spec():
    s = LossSpec.finetune_score_primary()
    assert s.use_mlm and s.use_score and not s.use_result
    assert s.lambda_score == 1.0 and s.lambda_result == 1.0  # default field, unused
