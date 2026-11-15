"""Figures and LaTeX table for the SIMULATION results (results/sim/results.json)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/sim/results.json"
OUT = os.path.dirname(SRC)
r = json.load(open(SRC))

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

NAMES = {"none": "No surface", "static": "Static config.", "hold": "Hold-based",
         "veilfi_free": "VeilFi free-run", "veilfi_pt": "VeilFi pkt-trig.",
         "veilfi_pt_rot": "VeilFi pkt-trig. + rot."}
ATT = [("i_variance", "(i) variance threshold", BLUE, "o"),
       ("iii_ml_unshielded", "(iii) ML, trained unshielded", ORANGE, "s"),
       ("iv_ml_retrained", "(iv) ML, retrained on shielded", AQUA, "D")]

# ---------- Fig. A: main comparison (E1) ----------
fig, ax = plt.subplots(figsize=(3.5, 2.9))
settings = list(r["E1"])
for j, (k, lab, col, mk) in enumerate(ATT):
    for i, s in enumerate(settings):
        d = r["E1"][s][k]
        y = i + (j - 1) * 0.24
        ax.errorbar(d["adv"], y, xerr=[[d["adv"] - d["lo"]], [d["hi"] - d["adv"]]],
                    fmt=mk, ms=4.5, color=col, ecolor=col, elinewidth=1.2, capsize=0,
                    label=lab if i == 0 else None)
ax.axvline(0.2, color=MUTED, lw=1, ls="--")
ax.text(0.215, -0.55, "target", color=MUTED, fontsize=6.5)
ax.set_yticks(range(len(settings)), [NAMES[s] for s in settings])
ax.set_xlim(-0.05, 1.0)
ax.set_xlabel("Attacker advantage (TPR − FPR) ± 95% CI")
ax.set_ylim(len(settings) - 0.5, -0.8)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sim_fig_main.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "sim_fig_main.png"), dpi=200, bbox_inches="tight")

# ---------- Fig. B: packet-rate sweep (E2), two panels, one axis each ----------
lams = sorted(map(int, r["E2"]))
SER = [("none", BLUE, "o"), ("hold", ORANGE, "s"),
       ("veilfi_pt", AQUA, "D"), ("veilfi_pt_rot", YELLOW, "^")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
for s, col, mk in SER:
    y1 = [r["E2"][str(l)][s]["iv_ml_retrained"]["adv"] for l in lams]
    y2 = [r["E2"][str(l)][s]["unchanged_pair_frac"] for l in lams]
    a1.plot(lams, y1, color=col, marker=mk, ms=4.5, lw=2, label=NAMES[s])
    a1.annotate(NAMES[s], (lams[-1], y1[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=6.5, color=INK)
    if s != "none":
        a2.plot(lams, y2, color=col, marker=mk, ms=4.5, lw=2, label=NAMES[s])
for a in (a1, a2):
    a.set_xscale("log")
    a.set_xticks(lams, [str(l) for l in lams])
    a.minorticks_off()
    a.axvline(r["params"]["controller_limit"], color=MUTED, lw=1, ls="--")
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_xlabel("Protected-network packet rate (packets/s)")
a1.set_ylabel("Adv, retrained attacker (iv)")
a1.set_ylim(0, 1)
a1.set_xlim(right=lams[-1] * 3.2)
a1.set_title("(a) Attacker advantage", fontsize=8, color=INK)
a2.set_ylabel("Consecutive packets, same config.")
a2.set_ylim(0, 1.05)
a2.set_title("(b) Unchanged packet pairs", fontsize=8, color=INK)
a2.text(r["params"]["controller_limit"] * 1.08, 0.05, "controller\nlimit", color=MUTED, fontsize=6.5)
a1.text(r["params"]["controller_limit"] * 1.08, 0.05, "controller\nlimit", color=MUTED, fontsize=6.5)
a2.legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sim_fig_rate.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "sim_fig_rate.png"), dpi=200, bbox_inches="tight")

# ---------- Fig. C: RIS-strength sweep (E3) ----------
rrs = sorted(map(float, r["E3"]))
fig, ax = plt.subplots(figsize=(3.5, 2.7))
for s, att, col, mk, ls in [("hold", "i_variance", ORANGE, "s", "--"),
                            ("hold", "iv_ml_retrained", ORANGE, "s", "-"),
                            ("veilfi_pt", "i_variance", AQUA, "D", "--"),
                            ("veilfi_pt", "iv_ml_retrained", AQUA, "D", "-")]:
    y = [r["E3"][str(x) if str(x) in r["E3"] else str(int(x))][s][att]["adv"] for x in rrs]
    ax.plot(rrs, y, color=col, marker=mk, ms=4.5, lw=2, ls=ls,
            label=f"{NAMES[s]}, {'(i) threshold' if att.startswith('i_') else '(iv) retrained ML'}")
ax.set_xscale("log", base=2)
ax.set_xticks(rrs, [f"{x:g}" for x in rrs])
ax.minorticks_off()
ax.set_ylim(-0.05, 1)
ax.axhline(0.2, color=MUTED, lw=1, ls=":")
ax.set_xlabel("Surface / human reflection amplitude ratio")
ax.set_ylabel("Attacker advantage")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sim_fig_strength.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "sim_fig_strength.png"), dpi=200, bbox_inches="tight")

# ---------- LaTeX table (E1 + E4) ----------
rows = []
for s in settings:
    d = r["E1"][s]
    cells = [f"{d[k]['adv']:.2f} [{d[k]['lo']:.2f}, {d[k]['hi']:.2f}]"
             for k in ("i_variance", "ii_min_change", "iii_ml_unshielded", "iv_ml_retrained")]
    e4 = r["E4"][s]
    rows.append(f"{NAMES[s]} & " + " & ".join(cells) +
                f" & {d['unchanged_pair_frac']:.2f} & {e4['std']:.2f} \\\\")
with open(os.path.join(OUT, "sim_table.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("figures and table written to", OUT)
