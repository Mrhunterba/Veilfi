"""Threshold attackers (paper Sec. IV-F, attackers i and ii; Eqs. 1-2).

Each function takes one window of normalized amplitudes, shape (N, K).
"""
from __future__ import annotations

import numpy as np


def windowed_variance(window: np.ndarray) -> float:
    """Attacker i: V(W) = mean over subcarriers of the variance over time."""
    return float(np.var(window, axis=0).mean())


def min_consecutive_change(window: np.ndarray) -> float:
    """Attacker ii: D(W) = min_t || |h_{t+1}| - |h_t| ||_2.

    Targets intervals in which the surface did not change between packets.
    """
    if window.shape[0] < 2:
        raise ValueError("window needs at least two frames")
    diffs = np.linalg.norm(np.diff(window, axis=0), axis=1)
    return float(diffs.min())


def consecutive_change_quantiles(window: np.ndarray,
                                 qs=(0.05, 0.25, 0.5)) -> np.ndarray:
    """Distribution features of consecutive changes (used alongside D(W))."""
    diffs = np.linalg.norm(np.diff(window, axis=0), axis=1)
    return np.quantile(diffs, qs)


class ThresholdDetector:
    """Calibrate a threshold on empty-room (no-motion) training windows."""

    def __init__(self, statistic, quantile: float = 0.95):
        self.statistic, self.quantile, self.threshold = statistic, quantile, None

    def fit(self, empty_windows) -> "ThresholdDetector":
        scores = [self.statistic(w) for w in empty_windows]
        self.threshold = float(np.quantile(scores, self.quantile))
        return self

    def predict(self, windows) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("call fit() first")
        return np.array([self.statistic(w) > self.threshold for w in windows], int)
