"""Session-level train/test separation (no window shares a session across sets)."""
from __future__ import annotations

import numpy as np


def session_split(sessions, test_sessions):
    """Return boolean masks (train, test) given the held-out session ids."""
    sessions = np.asarray(sessions)
    test = np.isin(sessions, list(test_sessions))
    return ~test, test


def assert_no_leakage(train_sessions, test_sessions) -> None:
    overlap = set(np.unique(train_sessions)) & set(np.unique(test_sessions))
    if overlap:
        raise AssertionError(f"sessions in both train and test: {sorted(overlap)}")
