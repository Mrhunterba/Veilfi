"""Reconfiguration schedule for the VeilFi privacy shield (paper Sec. IV-B).

Rules implemented here:
  * No holds      - a configuration is never kept over consecutive updates.
  * Large changes - every update draws a fresh random state for ALL elements.
  * Rate matching - the update rate is at least the observed packet rate,
                    capped by the measured controller limit.

Randomness comes from the operating system CSPRNG (``secrets``), which on
Linux is seeded from hardware entropy sources. Sequences are never stored or
logged, so an eavesdropper cannot replay or predict them.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Optional

N_ELEMENTS = 256  # open-source 5 GHz RIS [Heinrichs et al., ISAP 2023]


def random_configuration(n_elements: int = N_ELEMENTS) -> int:
    """Return a fresh random binary configuration as an n-bit integer."""
    return secrets.randbits(n_elements)


def next_configuration(previous: Optional[int], n_elements: int = N_ELEMENTS) -> int:
    """Draw a new full-array configuration that differs from ``previous``.

    A collision with the previous state has probability 2**-n_elements, but
    the "no holds" rule is enforced explicitly anyway.
    """
    while True:
        cfg = random_configuration(n_elements)
        if cfg != previous:
            return cfg


def to_bytes(cfg: int, n_elements: int = N_ELEMENTS) -> bytes:
    """Pack a configuration into bytes (MSB first) for the serial controller."""
    return cfg.to_bytes((n_elements + 7) // 8, "big")


@dataclass
class RateDecision:
    update_rate_hz: float
    meets_packet_rate: bool  # False => possible protection failure (report it)


def rate_matched_update_rate(
    packet_rate_hz: float,
    controller_limit_hz: float,
    margin: float = 1.2,
) -> RateDecision:
    """Free-running mode: choose an update rate >= packet rate (with margin).

    If the required rate exceeds the controller limit, the fastest available
    rate is used and the condition is flagged so it can be reported.
    """
    if packet_rate_hz < 0 or controller_limit_hz <= 0:
        raise ValueError("rates must be non-negative and limit positive")
    target = packet_rate_hz * margin
    if target <= controller_limit_hz:
        return RateDecision(update_rate_hz=max(target, 1e-3), meets_packet_rate=True)
    return RateDecision(update_rate_hz=controller_limit_hz,
                        meets_packet_rate=packet_rate_hz <= controller_limit_hz)


class PacketTriggeredScheduler:
    """Packet-triggered mode: update once after each observed packet,
    unless the controller is still busy with the previous command."""

    def __init__(self, controller_limit_hz: float, n_elements: int = N_ELEMENTS):
        self.min_interval = 1.0 / controller_limit_hz
        self.n_elements = n_elements
        self._last_time: Optional[float] = None
        self._current: Optional[int] = None
        self.skipped = 0  # packets that arrived while the controller was busy

    def on_packet(self, t: float) -> Optional[int]:
        """Return a new configuration to apply at time ``t``, or None if skipped."""
        if self._last_time is not None and t - self._last_time < self.min_interval:
            self.skipped += 1
            return None
        self._current = next_configuration(self._current, self.n_elements)
        self._last_time = t
        return self._current
