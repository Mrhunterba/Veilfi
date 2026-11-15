"""Traffic-monitor features from frame METADATA only (paper Sec. IV-D).

A frame record is a dict with keys:
    t (float, seconds), src (str), dst (str), ftype (str: 'data'|'mgmt'|'ctrl'),
    subtype (str, e.g. 'ack', 'beacon', 'null'), associated (bool, src is a
    member of the protected network).
No payload field exists by design.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def packet_rate(frames, bssid_members: set[str], window_s: float) -> float:
    """Packets/s from the protected network - drives the shield update rate."""
    n = sum(1 for f in frames if f["src"] in bssid_members)
    return n / window_s if window_s > 0 else 0.0


def window_features(frames, window_s: float) -> dict:
    """Features for the active-sensing classifier for one time window."""
    if not frames:
        return {"rate": 0.0, "iat_mean": 0.0, "iat_cv": 0.0, "unassoc_frac": 0.0,
                "ack_frac": 0.0, "top_unassoc_rate": 0.0, "burst_max": 0}
    t = np.sort([f["t"] for f in frames])
    iat = np.diff(t) if len(t) > 1 else np.array([0.0])
    types = Counter(f["subtype"] for f in frames)
    unassoc = [f for f in frames if not f["associated"]]
    by_sender = Counter(f["src"] for f in unassoc)
    bursts = np.split(t, np.flatnonzero(iat > 0.01) + 1)  # gaps > 10 ms split bursts
    return {
        "rate": len(frames) / window_s,
        "iat_mean": float(iat.mean()),
        "iat_cv": float(iat.std() / iat.mean()) if iat.mean() > 0 else 0.0,
        "unassoc_frac": len(unassoc) / len(frames),
        "ack_frac": types.get("ack", 0) / len(frames),
        # steady stream from one unassociated sender (Polite-WiFi style probing)
        "top_unassoc_rate": (max(by_sender.values()) / window_s) if by_sender else 0.0,
        "burst_max": int(max(len(b) for b in bursts)),
    }
