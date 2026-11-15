"""CSI preprocessing (paper Sec. V-C).

Input: complex CSI array of shape (T, K) - T frames, K usable subcarriers
(null and pilot subcarriers already removed).
"""
from __future__ import annotations

import numpy as np


def normalize_amplitude(h: np.ndarray) -> np.ndarray:
    """Per-frame amplitude normalization to reduce AGC effects."""
    amp = np.abs(h)
    norm = np.linalg.norm(amp, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return amp / norm


def sanitize_phase(h: np.ndarray) -> np.ndarray:
    """Remove the linear phase trend across subcarriers for each frame."""
    phase = np.unwrap(np.angle(h), axis=1)
    k = np.arange(h.shape[1])
    out = np.empty_like(phase)
    for t in range(h.shape[0]):
        slope, intercept = np.polyfit(k, phase[t], 1)
        out[t] = phase[t] - (slope * k + intercept)
    return out


def sliding_windows(x: np.ndarray, length: int, step: int) -> np.ndarray:
    """Return windows of shape (n_windows, length, ...) over the first axis."""
    if length <= 0 or step <= 0:
        raise ValueError("length and step must be positive")
    starts = range(0, x.shape[0] - length + 1, step)
    return np.stack([x[s:s + length] for s in starts]) if x.shape[0] >= length \
        else np.empty((0, length) + x.shape[1:])
