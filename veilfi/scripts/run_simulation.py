"""Run the VeilFi SIMULATION study and write results to results/sim/.

    python scripts/run_simulation.py            # full study
    python scripts/run_simulation.py --quick    # smoke run

Experiments
  E1  main table: 4 attackers x shield settings at default parameters
  E2  packet-rate sweep (controller limit fixed) - hypotheses H1/H3
  E3  RIS-strength sweep - effect of reflection strength
  E4  communication proxy (Shannon capacity at a legitimate receiver)
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from veilfi.attackers.classifier import make_attacker, window_features
from veilfi.attackers.threshold import (ThresholdDetector, min_consecutive_change,
                                        windowed_variance)
from veilfi.csi.preprocess import normalize_amplitude
from veilfi.eval.metrics import rates, session_bootstrap_ci
from veilfi.sim.model import LABEL, Environment, SimParams, simulate_block

ACTIVITIES = ["empty", "stationary", "walking", "arm"]
SETTINGS = ["none", "static", "hold", "veilfi_free", "veilfi_pt", "veilfi_pt_rot"]


def session_data(args):
    """Simulate one session under one setting; return per-window records."""
    sess, setting, p, seed = args
    env_rng = np.random.default_rng([seed, sess])          # same room per session
    env = Environment(env_rng)
    legit = Environment(np.random.default_rng([seed, sess, 999]))
    rng = np.random.default_rng([seed, sess, zlib.crc32(setting.encode())])
    feats, dmin, labels, caps, acts, same = [], [], [], [], [], []
    for act in rng.permutation(ACTIVITIES):
        t, H, cap, idx = simulate_block(env, act, setting, p, rng, legit)
        if idx is not None:
            same.append(np.mean(idx[1:] == idx[:-1]))
        if cap is not None:
            caps.append(cap)
        A = normalize_amplitude(H)
        edges = np.arange(0, p.block_s + 1e-9, p.window_s)
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t < b)
            if m.sum() < 3:
                continue
            w = A[m]
            feats.append(window_features(w))
            dmin.append(min_consecutive_change(w))
            labels.append(LABEL[act])
            acts.append(ACTIVITIES.index(act))
    return dict(session=sess, setting=setting, feats=np.array(feats),
                dmin=np.array(dmin), y=np.array(labels), act=np.array(acts),
                same_frac=float(np.mean(same)) if same else 1.0,
                cap=np.concatenate(caps) if caps else None)


def stack(recs):
    X = np.concatenate([r["feats"] for r in recs])
    D = np.concatenate([r["dmin"] for r in recs])
    y = np.concatenate([r["y"] for r in recs])
    s = np.concatenate([np.full(len(r["y"]), r["session"]) for r in recs])
    act = np.concatenate([r["act"] for r in recs])
    return X, D, y, s, act


def evaluate(recs_by_setting, train_ids, test_ids, ref_setting="none", n_boot=1000):
    """Return {setting: {attacker: (adv, lo, hi, tpr, fpr)}}."""
    out = {}
    ref = [r for r in recs_by_setting[ref_setting] if r["session"] in train_ids]
    Xr, _, yr, _, _ = stack(ref)
    clf_unshielded = make_attacker(0).fit(Xr, yr)                    # attacker iii
    for setting, recs in recs_by_setting.items():
        tr = [r for r in recs if r["session"] in train_ids]
        te = [r for r in recs if r["session"] in test_ids]
        Xtr, Dtr, ytr, _, _ = stack(tr)
        Xte, Dte, yte, ste, ate = stack(te)
        preds = {}
        # i: windowed variance, threshold = 95th pct of no-motion training windows
        thr = np.quantile(Xtr[ytr == 0, 0], 0.95)
        preds["i_variance"] = (Xte[:, 0] > thr).astype(int)
        # ii: smallest consecutive change
        thr = np.quantile(Dtr[ytr == 0], 0.95)
        preds["ii_min_change"] = (Dte > thr).astype(int)
        preds["iii_ml_unshielded"] = clf_unshielded.predict(Xte)
        preds["iv_ml_retrained"] = make_attacker(1).fit(Xtr, ytr).predict(Xte)
        out[setting] = {"unchanged_pair_frac": float(np.mean([r["same_frac"] for r in recs]))}
        for k, yp in preds.items():
            adv, lo, hi = session_bootstrap_ci(yte, yp, ste, n_boot=n_boot)
            tpr, fpr = rates(yte, yp)
            per_act = {a: float(yp[ate == i].mean()) for i, a in enumerate(ACTIVITIES)}
            out[setting][k] = dict(adv=adv, lo=lo, hi=hi, tpr=tpr, fpr=fpr,
                                   positive_rate_by_activity=per_act)
    return out


def run(settings, p, n_sessions, seed, pool):
    jobs = [(s, st, p, seed) for st in settings for s in range(n_sessions)]
    recs = list(pool.map(session_data, jobs, chunksize=1))
    by = {st: [r for r in recs if r["setting"] == st] for st in settings}
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/sim")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_sess = 6 if a.quick else 24
    n_train = n_sess * 5 // 8
    train_ids, test_ids = set(range(n_train)), set(range(n_train, n_sess))
    base = SimParams(block_s=10 if a.quick else 30)
    results = {"params": asdict(base), "n_sessions": n_sess,
               "train_sessions": sorted(train_ids), "test_sessions": sorted(test_ids)}

    with ProcessPoolExecutor() as pool:
        # E1 main table + E4 comm proxy
        by = run(SETTINGS, base, n_sess, a.seed, pool)
        results["E1"] = evaluate(by, train_ids, test_ids)
        cap_ref = np.concatenate([r["cap"] for r in by["none"]])
        results["E4"] = {}
        for st in SETTINGS:
            c = np.concatenate([r["cap"] for r in by[st]])
            results["E4"][st] = dict(
                mean_rel=float(c.mean() / cap_ref.mean()),
                p5_rel=float(np.percentile(c, 5) / np.percentile(cap_ref, 5)),
                std=float(c.std()))
        print("E1/E4 done", flush=True)

        # E2 packet-rate sweep
        results["E2"] = {}
        for lam in ([10, 100, 400] if a.quick else [10, 50, 100, 200, 400, 800]):
            p = replace(base, packet_rate=float(lam))
            by = run(["none", "hold", "veilfi_pt", "veilfi_pt_rot"], p, n_sess, a.seed, pool)
            results["E2"][str(lam)] = evaluate(by, train_ids, test_ids, n_boot=500)
            print("E2", lam, flush=True)

        # E3 RIS-strength sweep
        results["E3"] = {}
        for rr in ([0.5, 2] if a.quick else [0.25, 0.5, 1, 2, 4]):
            p = replace(base, ris_ratio=float(rr))
            by = run(["none", "hold", "veilfi_pt"], p, n_sess, a.seed, pool)
            results["E3"][str(rr)] = evaluate(by, train_ids, test_ids, n_boot=500)
            print("E3", rr, flush=True)

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved", os.path.join(a.out, "results.json"))


if __name__ == "__main__":
    main()
