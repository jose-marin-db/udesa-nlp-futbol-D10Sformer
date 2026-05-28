"""Tests for src/inference/predictions.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.predictions import (  # noqa: E402
    joint_score_probs_to_result_probs,
    score_class_to_goals,
    score_logits_to_probs,
)


def test_score_class_to_goals():
    assert score_class_to_goals(0) == (0, 0)
    assert score_class_to_goals(7) == (1, 1)  # 1*6+1
    assert score_class_to_goals(35) == (5, 5)


def test_joint_probs_sum_to_one():
    logits = torch.zeros(36)
    probs = score_logits_to_probs(logits)
    assert probs.shape == (36,)
    assert np.isclose(probs.sum(), 1.0)
    wld = joint_score_probs_to_result_probs(probs)
    assert wld.shape == (3,)
    assert np.isclose(wld.sum(), 1.0)


def test_one_hot_score_maps_to_correct_result():
    for h in range(6):
        for a in range(6):
            idx = h * 6 + a
            probs = np.zeros(36)
            probs[idx] = 1.0
            wld = joint_score_probs_to_result_probs(probs)
            if h > a:
                assert wld[0] == 1.0 and wld[1] == 0.0 and wld[2] == 0.0
            elif h == a:
                assert wld[0] == 0.0 and wld[1] == 1.0 and wld[2] == 0.0
            else:
                assert wld[0] == 0.0 and wld[1] == 0.0 and wld[2] == 1.0


def test_batched_aggregation():
    probs = np.zeros((2, 36))
    probs[0, 0] = 1.0   # 0-0 draw
    probs[1, 7] = 1.0   # 1-1 draw
    wld = joint_score_probs_to_result_probs(probs)
    assert wld.shape == (2, 3)
    assert np.allclose(wld[:, 1], 1.0)
