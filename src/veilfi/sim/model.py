"""Simulated WiFi CSI with a reconfigurable surface (SIMULATION ONLY).

Narrowband-per-subcarrier geometric model at 5.5 GHz:

    H_t(f) = H_static(f) + H_human,t(f) + H_RIS,t(f) + H_rot,t(f) + n_t(f)

* H_static : L static multipath components (walls, furniture), random per session.
* H_human  : one reflection whose path length d(t) follows the activity
             (walking, arm motion, breathing, or absent).
* H_RIS    : sum over N binary elements, b_n(t) in {+1,-1} (pi phase contrast),
             scaled so that its RMS amplitude is ``ris_ratio`` x human amplitude.
* H_rot    : optional rotating reflector, continuously changing path length.
* n_t      : complex AWGN.

All numeric parameters are ASSUMPTIONS chosen to be physically plausible; they
are not measured on VeilFi hardware. Results from this model indicate trends
only and must be reported as simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0 = 3e8
FC = 5.5e9
K = 56                                     # usable subcarriers, 20 MHz channel
FREQS = FC + (np.arange(K) - K / 2 + 0.5) * 312.5e3
N_RIS = 256

LABEL = {"empty": 0, "stationary": 0, "walking": 1, "arm": 1}


@dataclass
class SimParams:
    snr_db: float = 30.0
    human_amp: float = 0.2            # relative to unit-power static channel
    ris_ratio: float = 1.0            # RIS RMS amplitude / human amplitude
    rot_ratio: float = 1.0            # rotating reflector amplitude / human amplitude
    packet_rate: float = 100.0        # protected-network packets/s seen by eavesdropper
    controller_limit: float = 200.0   # max RIS updates/s (ASSUMED; to be measured)
    hold_range_s: tuple = (0.02, 0.2) # hold-based (IRShield-like) dwell times (ASSUMED)
    block_s: float = 30.0             # seconds per activity block
    window_s: float = 1.0


@dataclass
class Environment:
    """One recording session: room multipath, geometry, RIS element responses."""
    rng: np.random.Generator
    static: np.ndarray = field(init=False)
    tau_h: float = field(init=False)
    ph_h: float = field(init=False)
    ris_coef: np.ndarray = field(init=False)     # (N_RIS, K)
    tau_rot: float = field(init=False)

    def __post_init__(self):
        r = self.rng
        L = 8
        a = (r.normal(size=L) + 1j * r.normal(size=L)) * np.exp(-np.arange(L) / 3)
        tau = np.sort(r.uniform(5e-9, 80e-9, L))
        h = (a[:, None] * np.exp(-2j * np.pi * FREQS[None] * tau[:, None])).sum(0)
        self.static = h / np.sqrt(np.mean(np.abs(h) ** 2))
        self.tau_h = r.uniform(10e-9, 40e-9)
        self.ph_h = r.uniform(0, 2 * np.pi)
        tau_n = r.uniform(8e-9, 12e-9) + r.uniform(0, 1e-9, N_RIS)
        psi = r.uniform(0, 2 * np.pi, N_RIS)
        self.ris_coef = np.exp(-2j * np.pi * FREQS[None] * tau_n[:, None] + 1j * psi[:, None]) \
            / np.sqrt(N_RIS)
        self.tau_rot = r.uniform(8e-9, 15e-9)


def _ou(rng, t, std, tau_c, dt=1e-3):
    """Ornstein-Uhlenbeck velocity integrated to displacement, sampled at t."""
    n = int(np.ceil(t[-1] / dt)) + 2 if len(t) else 2
    a = np.exp(-dt / tau_c)
    w = rng.normal(0, std * np.sqrt(1 - a * a), n)
    v = np.empty(n)
    v[0] = rng.normal(0, std)
    for i in range(1, n):
        v[i] = a * v[i - 1] + w[i]
    d = np.concatenate([[0], np.cumsum(v[:-1] * dt)])
    return np.interp(t, np.arange(n) * dt, d)


def human_term(env: Environment, activity: str, t: np.ndarray, p: SimParams):
    """Complex human reflection for each packet time; shape (T, K)."""
    rng = env.rng
    if activity == "empty":
        return np.zeros((len(t), K), complex)
    if activity == "walking":
        d, amp = 2 * _ou(rng, t, 0.8, 1.0), p.human_amp           # x2: reflection path
    elif activity == "arm":
        d, amp = 2 * _ou(rng, t, 0.25, 0.5), 0.5 * p.human_amp
    elif activity == "stationary":
        d = 2 * 0.005 * np.sin(2 * np.pi * rng.uniform(0.2, 0.33) * t + rng.uniform(0, 6.3))
        amp = p.human_amp
    else:
        raise ValueError(activity)
    phase = -2j * np.pi * FREQS[None] * (env.tau_h + d[:, None] / C0)
    return amp * np.exp(1j * env.ph_h) * np.exp(phase)


def rot_term(env: Environment, t: np.ndarray, p: SimParams, rng):
    """Rotating plate: path length 0.2 m * cos(theta(t)), slowly varying speed."""
    omega = 2 * np.pi * (1.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6.3)))
    theta = np.concatenate([[0], np.cumsum(omega[:-1] * np.diff(t))]) if len(t) > 1 else t
    d = 0.2 * np.cos(theta)
    amp = p.rot_ratio * p.human_amp
    return amp * np.exp(-2j * np.pi * FREQS[None] * (env.tau_rot + d[:, None] / C0))


# ---------------------------------------------------------------- schedules
def schedule(setting: str, t: np.ndarray, p: SimParams, rng):
    """Return (config index per packet, configs as +/-1 array, n_skipped)."""
    T = len(t)
    if setting in ("none", "static"):
        return np.zeros(T, int), rng.choice([-1, 1], (1, N_RIS)), 0
    if setting == "hold":
        ups, s = [], 0.0
        while s < t[-1]:
            ups.append(s)
            s += rng.uniform(*p.hold_range_s)
        ups = np.array(ups)
    elif setting.startswith("veilfi_free"):
        rate = min(1.2 * p.packet_rate, p.controller_limit)
        ups = np.arange(rng.uniform(0, 1 / rate), t[-1] + 1 / rate, 1 / rate)
        ups = np.concatenate([[0.0], ups])
    elif setting.startswith("veilfi_pt"):
        ups, last, skipped = [0.0], -np.inf, 0
        min_iv = 1.0 / p.controller_limit
        for tp in t:                       # update triggered right after each packet
            if tp - last >= min_iv:
                ups.append(tp + 1e-6)
                last = tp
            else:
                skipped += 1
        ups = np.array(ups)
        idx = np.searchsorted(ups, t, side="right") - 1
        return idx, rng.choice([-1, 1], (len(ups), N_RIS)), skipped
    else:
        raise ValueError(setting)
    idx = np.searchsorted(ups, t, side="right") - 1
    return idx, rng.choice([-1, 1], (len(ups), N_RIS)), 0


def simulate_block(env: Environment, activity: str, setting: str, p: SimParams,
                   rng, legit_env: Environment | None = None):
    """Simulate one block. Returns times, eavesdropper CSI, legit capacity, config index."""
    n = rng.poisson(p.packet_rate * p.block_s)
    t = np.sort(rng.uniform(0, p.block_s, max(n, 2)))
    H = env.static[None] + human_term(env, activity, t, p)
    cap, idx = None, None
    if setting != "none":
        idx, B, _ = schedule(setting, t, p, rng)
        alpha = p.ris_ratio * p.human_amp
        H = H + alpha * (B @ env.ris_coef)[idx]
        if setting.endswith("_rot"):
            H = H + rot_term(env, t, p, rng)
        if legit_env is not None:
            G = legit_env.static[None] + alpha * (B @ legit_env.ris_coef)[idx]
            if setting.endswith("_rot"):
                G = G + rot_term(legit_env, t, p, rng)
            cap = capacity(G, p.snr_db)
    elif legit_env is not None:
        cap = capacity(np.repeat(legit_env.static[None], len(t), 0), p.snr_db)
    sigma = np.sqrt(10 ** (-p.snr_db / 10) / 2)
    H = H + sigma * (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape))
    return t, H, cap, idx


def capacity(G, snr_db):
    """Per-packet Shannon capacity proxy (bit/s/Hz), averaged over subcarriers."""
    snr = 10 ** (snr_db / 10)
    return np.log2(1 + snr * np.abs(G) ** 2).mean(axis=1)
