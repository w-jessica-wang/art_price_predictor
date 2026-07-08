# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Walk-forward (rolling-origin) cross-validation for the art-price model.

Why: the single 2025 H2 test split contains NO Christie's (their data caps at
2025-03-27, so all Christie's sits in 2025 H1). It also rests every headline number
on one window. Walk-forward fixes both: several temporal folds, each training on the
past and testing on the next quarter, refitting the FULL calibration chain (isotonic
→ group-bias → Stage-C shrink → CQR) per fold via the same train() pipeline. Christie's
gets clean out-of-sample scoring in the earlier folds; pooling folds gives stable,
across-time metrics and per-fold confidence.

Each fold uses (train_end, val_start, val_end, test_end) with a 1-month embargo gap
between train_end and val_start (lookback features span up to 12m; merge_asof already
blocks within-window leakage, the embargo guards the train/eval seam).

Run:  python -m ml.walk_forward_eval --data <clean.csv>
Outputs: models_eval/walk_forward.json (summary), figures/walk_forward.png,
         models_eval/walk_forward_preds.parquet (pooled OOS test predictions).
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from ml.train import train

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _quarter_start(year, q):
    return pd.Timestamp(year=year, month=(q - 1) * 3 + 1, day=1)


def make_folds(embargo_months=1):
    """Test each quarter from 2024 Q4 through 2025 Q4. For test quarter starting Qs:
       test=[Qs, Qs+3mo), val=[Qs-3mo, Qs), train< Qs-3mo-embargo."""
    test_quarters = [(2024, 4), (2025, 1), (2025, 2), (2025, 3), (2025, 4)]
    folds = []
    for (yr, q) in test_quarters:
        qs = _quarter_start(yr, q)
        val_start = qs - pd.DateOffset(months=3)
        train_end = val_start - pd.DateOffset(months=embargo_months)
        test_end  = qs + pd.DateOffset(months=3)
        folds.append({
            "label": f"{yr} Q{q}",
            "fold_dates": (train_end, val_start, qs, test_end),
        })
    return folds


# ── metric helpers (dollar space) ──────────────────────────────────────────
def _ape(p, y):       return np.abs(p - y) / np.clip(y, 1, None)
def medape(p, y):     return float(np.median(_ape(p, y)) * 100)
def within20(p, y):   return float((_ape(p, y) <= 0.20).mean() * 100)
def r2log(p, y):
    a, b = np.log1p(y), np.log1p(np.clip(p, 1, None))
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


def block_metrics(d):
    y = d.sold_price.values.astype(float)
    est = d.estimate_mid.values.astype(float)
    mp = d.predicted_hammer.values.astype(float)
    ok = (y > 0) & (est > 0) & (mp > 0) & np.isfinite(mp)
    y, est, mp, d = y[ok], est[ok], mp[ok], d[ok].reset_index(drop=True)
    cov = float(((y >= d.conformal_p10.values) & (y <= d.conformal_p90.values)).mean() * 100)
    beat = (y > est).astype(int)
    score = d.pred_logratio_raw.values
    auc = float(roc_auc_score(beat, score)) if beat.min() != beat.max() else float("nan")
    return {
        "n": int(len(y)),
        "medape_model": round(medape(mp, y), 1), "medape_est": round(medape(est, y), 1),
        "within20_model": round(within20(mp, y), 1), "within20_est": round(within20(est, y), 1),
        "r2_model": round(r2log(mp, y), 3), "r2_est": round(r2log(est, y), 3),
        "coverage": round(cov, 1), "ranking_auc": round(auc, 3),
        "base_rate_beat": round(float(beat.mean() * 100), 1),
    }


def paired_medape_delta_ci(d, n_boot=3000, seed=0):
    """Artist-clustered paired bootstrap of Δ = MedAPE(model) − MedAPE(estimate).
    Returns (delta_point, lo, hi, frac_model_better). Resamples whole artists so
    correlated lots don't shrink the interval."""
    y = d.sold_price.values.astype(float); est = d.estimate_mid.values.astype(float)
    mp = d.predicted_hammer.values.astype(float)
    ok = (y > 0) & (est > 0) & (mp > 0) & np.isfinite(mp)
    y, est, mp = y[ok], est[ok], mp[ok]
    artist = d.artist.astype(str).values[ok]
    groups = {a: np.where(artist == a)[0] for a in np.unique(artist)}
    keys = list(groups.keys()); rng = np.random.default_rng(seed)
    def _delta(idx):
        am = np.abs(mp[idx]-y[idx])/np.clip(y[idx],1,None)
        ae = np.abs(est[idx]-y[idx])/np.clip(y[idx],1,None)
        return (np.median(am) - np.median(ae)) * 100
    point = _delta(np.arange(len(y)))
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[k]] for k in pick])
        boot.append(_delta(idx))
    boot = np.array(boot)
    return (round(float(point), 2), round(float(np.percentile(boot, 2.5)), 2),
            round(float(np.percentile(boot, 97.5)), 2), round(float((boot < 0).mean()), 3))


def cluster_auc_ci(d, cluster_col="artist", n_boot=2000, seed=0):
    """Cluster-resampled bootstrap CI for pooled ranking AUC. cluster_col chooses
    the correlation unit (artist, or auction_id for sale-level). Resamples whole
    clusters so correlated lots don't shrink the interval. Returns (lo, hi)."""
    y = d.sold_price.values.astype(float); est = d.estimate_mid.values.astype(float)
    beat = (y > est).astype(int); score = d.pred_logratio_raw.values
    if cluster_col not in d.columns:
        return None, None
    cl = d[cluster_col].astype(str).values
    groups = {a: np.where(cl == a)[0] for a in np.unique(cl)}
    keys = list(groups.keys()); rng = np.random.default_rng(seed); boot = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[k]] for k in pick])
        if beat[idx].min() != beat[idx].max():
            boot.append(roc_auc_score(beat[idx], score[idx]))
    return round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/art_auction_clean.csv")
    ap.add_argument("--embargo-months", type=int, default=1)
    args = ap.parse_args()

    folds = make_folds(args.embargo_months)
    tmp = tempfile.mkdtemp(prefix="wf_")
    pooled, per_fold = [], []
    try:
        for f in folds:
            print(f"\n{'='*70}\n  WALK-FORWARD FOLD — test {f['label']}\n{'='*70}")
            try:
                res = train(data_path=args.data, models_dir=tmp, force_recompute=True,
                            fold_dates=f["fold_dates"], return_fold_preds=True)
            except Exception as e:
                print(f"  ⚠ fold {f['label']} failed: {type(e).__name__}: {e}")
                continue
            tp = res["test_preds"]
            if tp is None or len(tp) < 50:
                print(f"  ⚠ fold {f['label']} too small (n={0 if tp is None else len(tp)}) — skipped")
                continue
            tp = tp.copy(); tp["fold"] = f["label"]
            pooled.append(tp)
            fm = block_metrics(tp); fm["fold"] = f["label"]
            # per-house within this fold (esp. Christie's coverage)
            fm["houses"] = {h: int((tp.house.str.lower() == h).sum())
                            for h in tp.house.str.lower().unique()}
            per_fold.append(fm)
            print(f"  fold {f['label']}: n={fm['n']}  MedAPE {fm['medape_model']}/{fm['medape_est']}  "
                  f"AUC {fm['ranking_auc']}  Christie's={fm['houses'].get('christies',0)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not pooled:
        print("No folds succeeded."); return

    allp = pd.concat(pooled, ignore_index=True)
    agg = block_metrics(allp)
    auc_lo, auc_hi = cluster_auc_ci(allp, "artist")
    sale_lo, sale_hi = cluster_auc_ci(allp, "auction_id")   # sale-level clustering sensitivity
    md_pt, md_lo, md_hi, md_frac = paired_medape_delta_ci(allp)

    # pooled per-house (the payoff: Christie's now has clean OOS coverage)
    per_house = {}
    for h, g in allp.groupby(allp.house.str.lower()):
        if len(g) < 50:
            continue
        per_house[h] = block_metrics(g)

    # decile lift on pooled raw signal + monotonicity
    y = allp.sold_price.values.astype(float); est = allp.estimate_mid.values.astype(float)
    beat = (y > est).astype(int); score = allp.pred_logratio_raw.values
    dec = pd.qcut(score, 10, labels=False, duplicates="drop")
    drate = pd.Series(beat).groupby(dec).mean().values * 100
    rho, rho_p = spearmanr(np.arange(len(drate)), drate)

    summary = {
        "method": "purged walk-forward CV (rolling-origin), embargo "
                  f"{args.embargo_months}mo, full calibration chain refit per fold",
        "n_folds": len(per_fold),
        "pooled": {**agg, "ranking_auc_ci": [auc_lo, auc_hi],
                   "ranking_auc_ci_artist_clustered": [auc_lo, auc_hi],
                   "ranking_auc_ci_sale_clustered": [sale_lo, sale_hi],
                   "medape_delta_pp": md_pt,                 # model − estimate (negative = model better)
                   "medape_delta_ci": [md_lo, md_hi],        # artist-clustered paired bootstrap
                   "medape_delta_p_model_better": md_frac,
                   "decile_bottom": round(float(drate[0]), 1),
                   "decile_top": round(float(drate[-1]), 1),
                   "decile_spearman_rho": round(float(rho), 3),
                   "decile_spearman_p": float(rho_p)},
        "per_fold": per_fold,
        "per_house_pooled": per_house,
        "christies_test_lots": int((allp.house.str.lower() == "christies").sum()),
    }
    os.makedirs(os.path.join(ROOT, "models_eval"), exist_ok=True)
    with open(os.path.join(ROOT, "models_eval", "walk_forward.json"), "w") as fjson:
        json.dump(summary, fjson, indent=2)
    allp.to_parquet(os.path.join(ROOT, "models_eval", "walk_forward_preds.parquet"), index=False)

    # figure
    try:
        _make_figure(per_fold, agg, per_house, summary)
    except Exception as e:
        print(f"  (figure skipped: {e})")

    print("\n" + "=" * 70)
    print(f"  POOLED ({agg['n']:,} OOS lots across {len(per_fold)} folds, "
          f"Christie's test lots = {summary['christies_test_lots']:,})")
    print(f"  MedAPE  model/est : {agg['medape_model']}% / {agg['medape_est']}%  "
          f"(paired Δ {md_pt}pp, 95% CI [{md_lo}, {md_hi}], P(model better)={md_frac})")
    print(f"  Within20 model/est: {agg['within20_model']}% / {agg['within20_est']}%")
    print(f"  R2(log) model/est : {agg['r2_model']} / {agg['r2_est']}")
    print(f"  Ranking AUC       : {agg['ranking_auc']}  CI artist-clustered [{auc_lo}, {auc_hi}]  sale-clustered [{sale_lo}, {sale_hi}]")
    print(f"  Coverage          : {agg['coverage']}%")
    print(f"  Decile {summary['pooled']['decile_bottom']}% → {summary['pooled']['decile_top']}%  "
          f"(Spearman ρ={summary['pooled']['decile_spearman_rho']})")
    print("\n  Per-house (pooled):")
    for h, m in per_house.items():
        print(f"    {h:10s} n={m['n']:5d}  MedAPE {m['medape_model']}/{m['medape_est']}  AUC {m['ranking_auc']}")
    print(f"\n  Saved → models_eval/walk_forward.json + walk_forward_preds.parquet + figures/walk_forward.png")


def _make_figure(per_fold, agg, per_house, summary):
    import matplotlib.pyplot as plt
    labels = [f["fold"] for f in per_fold]
    mm = [f["medape_model"] for f in per_fold]
    me = [f["medape_est"] for f in per_fold]
    auc = [f["ranking_auc"] for f in per_fold]
    chr_n = [f["houses"].get("christies", 0) for f in per_fold]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    x = np.arange(len(labels))
    ax1.plot(x, me, "o--", color="#888", label="Specialist estimate")
    ax1.plot(x, mm, "o-", color="#42a5f5", label="Model")
    for i, c in enumerate(chr_n):
        if c > 0:
            ax1.annotate(f"Christie's✓\n({c})", (x[i], min(mm[i], me[i])),
                         textcoords="offset points", xytext=(0, -28), ha="center",
                         fontsize=7.5, color="#ef5350")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("MedAPE (%)"); ax1.set_title("Walk-forward MedAPE by test quarter\n"
                  f"pooled model {agg['medape_model']}% vs est {agg['medape_est']}%", fontsize=11, loc="left")
    ax1.legend(fontsize=9); ax1.grid(axis="y", ls=":", alpha=0.4)

    ax2.bar(x, auc, color="#66bb6a", edgecolor="white")
    ax2.axhline(0.5, ls="--", color="#444"); ax2.text(len(x)-0.5, 0.505, " random", fontsize=8, color="#444")
    for i, a in enumerate(auc):
        ax2.text(x[i], a + 0.005, f"{a:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0.45, max(0.7, max(auc) + 0.05)); ax2.set_ylabel("Ranking AUC")
    ax2.set_title(f"Walk-forward ranking AUC by quarter\npooled {agg['ranking_auc']} "
                  f"CI {summary['pooled']['ranking_auc_ci']}", fontsize=11, loc="left")
    ax2.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "walk_forward.png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
