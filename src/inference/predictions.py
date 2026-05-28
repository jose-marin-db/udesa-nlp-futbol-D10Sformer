"""
Convert ScoreHead outputs to match-result probabilities.

Score classes are laid out row-major: index = home_goals * GOALS_PER_SIDE + away_goals,
with each side in {0, 1, ..., GOALS_PER_SIDE - 1} (training clamps to 0..5).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

GOALS_PER_SIDE = 6
NUM_SCORE_CLASSES = GOALS_PER_SIDE * GOALS_PER_SIDE


def score_class_to_goals(class_id: int) -> tuple[int, int]:
    """Local ScoreHead class index → (home_goals, away_goals)."""
    return class_id // GOALS_PER_SIDE, class_id % GOALS_PER_SIDE


def score_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    """Softmax over score classes. Accepts shape (C,) or (N, C)."""
    if logits.dim() == 1:
        return F.softmax(logits, dim=-1).detach().cpu().numpy().astype(float)
    return F.softmax(logits, dim=-1).detach().cpu().numpy().astype(float)


def joint_score_probs_to_result_probs(score_probs: np.ndarray) -> np.ndarray:
    """Aggregate joint score distribution into [p_home_win, p_draw, p_away_win].

    Args:
        score_probs: shape (36,) or (N, 36), non-negative, typically sums to 1 per row.

    Returns:
        shape (3,) or (N, 3) in order home_win, draw, away_win.
    """
    single = score_probs.ndim == 1
    if single:
        score_probs = score_probs[np.newaxis, :]

    n = score_probs.shape[0]
    out = np.zeros((n, 3), dtype=float)
    for idx in range(NUM_SCORE_CLASSES):
        h, a = score_class_to_goals(idx)
        p = score_probs[:, idx]
        if h > a:
            out[:, 0] += p
        elif h == a:
            out[:, 1] += p
        else:
            out[:, 2] += p

    if single:
        return out[0]
    return out
