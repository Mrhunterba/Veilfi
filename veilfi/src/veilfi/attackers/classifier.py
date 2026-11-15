"""ML attackers (paper Sec. IV-F, attackers iii and iv).

Baseline: Random Forest on simple window statistics. A CNN-GRU-attention model
following Mekruksavanich et al. (2023) is the planned stronger attacker and
will be added under ``attackers/deep.py``.

Attacker iii = fit on unshielded sessions; attacker iv = the same pipeline
re-fit on shielded sessions.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .threshold import consecutive_change_quantiles, windowed_variance


def window_features(window: np.ndarray) -> np.ndarray:
    per_sc_std = window.std(axis=0)
    return np.concatenate([
        [windowed_variance(window)],
        consecutive_change_quantiles(window),
        np.quantile(per_sc_std, [0.1, 0.5, 0.9]),
    ])


def make_attacker(seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=seed,
                                  class_weight="balanced", n_jobs=-1)


def featurize(windows) -> np.ndarray:
    return np.stack([window_features(w) for w in windows])
