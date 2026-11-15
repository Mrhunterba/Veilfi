"""Privacy metrics (paper Sec. VI-A, Eq. 3).

Adv = TPR - FPR = 2 * balanced_accuracy - 1.
Confidence intervals are bootstrapped over SESSIONS, not windows, because
windows from the same session are correlated.
"""
from __future__ import annotations

import numpy as np


def rates(y_true, y_pred) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos, neg = y_true == 1, y_true == 0
    tpr = float((y_pred[pos] == 1).mean()) if pos.any() else float("nan")
    fpr = float((y_pred[neg] == 1).mean()) if neg.any() else float("nan")
    return tpr, fpr


def advantage(y_true, y_pred) -> float:
    tpr, fpr = rates(y_true, y_pred)
    return tpr - fpr


def session_bootstrap_ci(y_true, y_pred, sessions, n_boot: int = 2000,
                         alpha: float = 0.05, seed: int = 0):
    """Return (point estimate, lower, upper) for Adv, resampling sessions."""
    y_true, y_pred, sessions = map(np.asarray, (y_true, y_pred, sessions))
    ids = np.unique(sessions)
    rng = np.random.default_rng(seed)
    idx_by_session = {s: np.flatnonzero(sessions == s) for s in ids}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([idx_by_session[s] for s in pick])
        a = advantage(y_true[idx], y_pred[idx])
        if not np.isnan(a):
            stats.append(a)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return advantage(y_true, y_pred), float(lo), float(hi)
