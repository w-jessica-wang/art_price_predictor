"""Decile-lift chart: does the RAW ensemble ranking signal sort auction over/under-performers?

Bins the held-out **2025 H2 test set** (never used to fit calibration, Stage-C
shrink, or CQR — unlike 2025 H1 validation) by the model's *unshrunk* belief
(`pred_logratio_raw`) about whether a lot will exceed its presale **estimate
midpoint**, and plots the share of lots in each decile that actually did. A
monotonic staircase = real ranking skill the MedAPE hides.

Every bar carries a 95% Wilson confidence interval and its sample count, the
headline AUC carries a bootstrap CI, and a Spearman rank test makes the
monotonicity claim formal rather than visual.

Run:  python figures/decile_lift.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "..", "models_eval", "eval_predictions.parquet")

d = pd.read_parquet(PRED)
# Test-only: 2025 H2. The 2025 H1 validation rows were used to FIT calibration /
# Stage-C / CQR, so including them would be in-sample. Fall back gracefully if the
# split tag is absent (older artifact).
if "split" in d.columns:
    d = d[d["split"] == "test"].copy()
PERIOD = "2025 H2 holdout (test)"

y, e = d.sold_price.values, d.estimate_mid.values
beat = (y > e).astype(int)                 # did the hammer exceed the estimate MIDPOINT?
score = d.pred_logratio_raw.values         # model's unshrunk residual belief
base = beat.mean() * 100
auc = roc_auc_score(beat, score)

# Artist-CLUSTERED bootstrap CI for AUC: lots by the same artist are correlated, so
# we resample whole artists (not individual lots) — a wider, honest interval.
rng = np.random.default_rng(0)
artist = d.artist.astype(str).values
groups = {a: np.where(artist == a)[0] for a in np.unique(artist)}
gkeys = list(groups.keys())
boot = []
for _ in range(2000):
    pick = rng.choice(len(gkeys), len(gkeys), replace=True)
    idx = np.concatenate([groups[gkeys[k]] for k in pick])
    if beat[idx].min() != beat[idx].max():
        boot.append(roc_auc_score(beat[idx], score[idx]))
auc_lo, auc_hi = np.percentile(boot, [2.5, 97.5])

dec = pd.qcut(score, 10, labels=False, duplicates="drop")
g = pd.DataFrame({"dec": dec, "beat": beat})
grp = g.groupby("dec")["beat"]
pct = grp.mean().values * 100
cnt = grp.count().values


def _wilson(k, nn, z=1.96):
    if nn == 0:
        return 0.0, 0.0
    p = k / nn
    den = 1 + z**2 / nn
    c = (p + z**2 / (2 * nn)) / den
    half = z * np.sqrt(p * (1 - p) / nn + z**2 / (4 * nn**2)) / den
    return (c - half) * 100, (c + half) * 100


ci = [_wilson(int(round(p / 100 * c)), int(c)) for p, c in zip(pct, cnt)]
err_lo = [p - lo for p, (lo, hi) in zip(pct, ci)]
err_hi = [hi - p for p, (lo, hi) in zip(pct, ci)]

# Formal monotonicity: Spearman rank correlation between decile index and beat-rate
rho, rho_p = spearmanr(np.arange(len(pct)), pct)

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=140)
colors = plt.cm.RdYlGn(np.linspace(0.15, 0.90, len(pct)))
bars = ax.bar(range(1, len(pct) + 1), pct, color=colors, edgecolor="white",
              linewidth=0.8, zorder=3,
              yerr=[err_lo, err_hi], capsize=3, ecolor="#555")

ax.axhline(base, ls="--", lw=1.4, color="#444", zorder=2)
ax.text(len(pct) + 0.4, base, f" base rate {base:.0f}%", va="center", ha="left",
        fontsize=9, color="#444")

# Label EVERY bar with its rate and n (auditable)
for i, (b, p, c) in enumerate(zip(bars, pct, cnt), start=1):
    ax.text(b.get_x() + b.get_width() / 2, p + max(err_hi) + 1.5, f"{p:.0f}%",
            ha="center", va="bottom", fontsize=9.5,
            fontweight="bold" if i in (1, len(pct)) else "normal")
    ax.text(b.get_x() + b.get_width() / 2, 2, f"n={c}", ha="center", va="bottom",
            fontsize=7.5, color="#333", rotation=90)

ax.set_xticks(range(1, len(pct) + 1))
ax.set_xticklabels(["1\n(bearish)"] + [str(i) for i in range(2, len(pct))] + [f"{len(pct)}\n(bullish)"])
ax.set_xlabel("Raw ensemble over-performance signal (decile of pred_logratio_raw)", fontsize=11)
ax.set_ylabel("% of lots whose hammer exceeded the estimate midpoint", fontsize=10.5)
ax.set_ylim(0, 100)
ax.set_title("Raw ensemble ranking signal sorts auction over-performers\n"
             f"{PERIOD} (n={len(d):,})  ·  AUC = {auc:.2f} [{auc_lo:.2f}–{auc_hi:.2f}]  ·  "
             f"top {pct[-1]:.0f}% vs bottom {pct[0]:.0f}%  ·  Spearman ρ={rho:.2f} (p={rho_p:.1e})",
             fontsize=11.5, fontweight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()

OUT = os.path.join(HERE, "decile_lift.png")
fig.savefig(OUT, bbox_inches="tight")
print(f"period={PERIOD}  n={len(d):,}")
print(f"AUC={auc:.3f}  95% CI [{auc_lo:.3f}, {auc_hi:.3f}]  base={base:.1f}%")
print(f"bottom decile={pct[0]:.1f}% (n={cnt[0]})  top decile={pct[-1]:.1f}% (n={cnt[-1]})")
print(f"Spearman monotonicity: rho={rho:.3f}  p={rho_p:.2e}")
print(f"saved -> {OUT}")
