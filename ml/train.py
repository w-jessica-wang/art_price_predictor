# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Training Pipeline — Art Auction Price Prediction (research-grade evaluation)
======================================================
Speed engineering:
  - Base models (LightGBM + CatBoost) run in PARALLEL threads
    Each gets N_JOBS//2 cores so they don't compete
  - Quantile models (P10/P90) run in PARALLEL threads
  - CV folds run in PARALLEL threads
  - LR=0.05 → ~40% fewer iterations, same accuracy
  - Early stopping patience = 40

Pipeline:
  1. CatBoost with artist as native categorical (internal target encoding)
  2. Exponential time-weighting (recent data weighted higher)
  3. Separate low-value model for <$10k lots
  4. LightGBM quantile regression (honest P10/P90 intervals)
  5. Three-stage calibration: A tiered isotonic, B medium/house group-bias,
     C estimate-anchored shrink (provably no worse than estimate baseline on validation)
  6. 3-fold temporal CV for robust metric reporting
  7. OOF meta-stacking with NNLS meta-learner over 3 base models
  8. Two-model base ensemble (LightGBM + CatBoost); neural net dropped after
     ablation showed it added nothing (<0.06pp) — also removes the torch dependency
  9. Train/val/test diagnostics for overfitting detection
 10. eval_predictions.parquet saved for app Trends tab
 11. CQR conformal prediction intervals (empirical coverage guarantee)
 12. Sell-through probability model (binary LightGBM)
 13. Ablation study (feature group contribution analysis)
"""

import json, os, pickle, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

from ml.features import (
    load_data, compute_artist_features,
    temporal_split, FEATURE_COLS,
)

N_JOBS  = os.cpu_count() or 4
N_EACH  = max(1, N_JOBS // 3)   # cores per parallel model
LOW_VAL = 10_000


# ── metrics ────────────────────────────────────────────────────────────────

def eval_metrics(y_true, y_pred, est_arr=None, label="", verbose=True):
    """
    Compute evaluation metrics.
    est_arr: estimate_mid values — required when target is log_ratio.
             If provided: hammer = exp(pred) * est_mid.
             If None: hammer = expm1(pred) (legacy log_hammer target).
    """
    if est_arr is not None:
        h_true = np.exp(y_true) * est_arr
        h_pred = np.exp(np.clip(y_pred, -5, 5)) * est_arr
    else:
        h_true = np.expm1(y_true)
        h_pred = np.expm1(np.clip(y_pred, 0, 20))
    ape    = np.abs(h_true - h_pred) / h_true.clip(1)
    r2     = r2_score(y_true, y_pred)
    mape   = ape.mean() * 100
    medape = np.median(ape) * 100
    w20    = (ape < .20).mean() * 100
    w50    = (ape < .50).mean() * 100
    bias   = float((y_pred - y_true).mean())
    if verbose:
        tag = f"[{label}]" if label else ""
        print(f"  {tag:<24} R²={r2:.4f}  MedAPE={medape:.1f}%  "
              f"MAPE={mape:.1f}%  w20={w20:.1f}%  bias={bias:+.3f}")
    return dict(r2=r2, mape=mape, medape=medape, within_20=w20, within_50=w50, bias=bias)


def time_weights(dates: pd.Series, half_life_years=2.0) -> np.ndarray:
    days_old  = (dates.max() - dates).dt.days.fillna(0).values
    half_life = half_life_years * 365.25
    return np.exp(-days_old * np.log(2) / half_life).astype(np.float32)


def prep(sub: pd.DataFrame, use_weights=False, exclude_low=False, trim_outliers=False):
    """
    Prepare training data.
    TARGET: log_ratio = log(hammer / estimate_mid)  ← switched from log_hammer.
    exclude_low:    remove <$10k lots from this subset (train main model on mid/hi only).
    trim_outliers:  clip top/bottom 0.5% of log_ratio (removes data errors from training).
    """
    s = (sub[sub["sold"] == 1]
         .dropna(subset=["log_ratio", "log_estimate_mid"])
         .copy()
         .reset_index(drop=True))
    X       = s[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y       = s["log_ratio"].values.astype(np.float32)      # ← log_ratio target
    est     = s["estimate_mid"].values.astype(np.float64)
    art     = s["artist"].values
    art_med = (s["artist_medium"].values
               if "artist_medium" in s.columns
               else np.array([f"{a} | unknown" for a in art]))
    is_lo   = (est < LOW_VAL)
    w       = time_weights(s["sale_date"]) if use_weights else np.ones(len(s), np.float32)

    if exclude_low:
        mask = ~is_lo
        X, y, est, art, art_med, is_lo, w = (X[mask], y[mask], est[mask],
                                               art[mask], art_med[mask], is_lo[mask], w[mask])
        s = s[mask].reset_index(drop=True)

    if trim_outliers and len(y) > 200:
        q_lo, q_hi = np.quantile(y, [0.005, 0.995])
        mask = (y >= q_lo) & (y <= q_hi)
        X, y, est, art, art_med, is_lo, w = (X[mask], y[mask], est[mask],
                                               art[mask], art_med[mask], is_lo[mask], w[mask])
        s = s[mask].reset_index(drop=True)

    return X, y, est, art, art_med, is_lo, w, s


# ── model builders ─────────────────────────────────────────────────────────

def _lgbm_base_params(n_jobs, n_leaves=127):
    return dict(
        objective="regression_l1", metric="mae",
        num_leaves=n_leaves, learning_rate=0.05,
        feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
        min_child_samples=30, lambda_l1=0.3, lambda_l2=0.3,
        n_jobs=n_jobs, verbose=-1, seed=42, max_bin=255,
    )


def _fit_lgbm(X_tr, y_tr, X_val, y_val, w_tr, tag, n_jobs, n_leaves=127, alpha=None,
              extra_params=None, feature_cols=None, early_stopping_rounds=40):
    """
    early_stopping_rounds=None → fixed 800 rounds, no early stopping.
    Used for quantile models where val quantile loss diverges immediately due
    to train/val distribution shift (2018-2024 train vs 2025 val lower tail),
    causing early stopping to fire at iter=1 regardless of learning rate.
    """
    t0  = time.time()
    params = _lgbm_base_params(n_jobs, n_leaves)
    if alpha is not None:
        params["objective"] = "quantile"
        params["alpha"]     = alpha
    if extra_params:
        params.update(extra_params)
    feat_names = feature_cols if feature_cols is not None else FEATURE_COLS
    dtr  = lgb.Dataset(X_tr,  label=y_tr,  weight=w_tr, feature_name=feat_names, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, reference=dtr, free_raw_data=True)
    if early_stopping_rounds is not None:
        m = lgb.train(
            params, dtr, num_boost_round=2000, valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                       lgb.log_evaluation(9999)],
        )
    else:
        # No early stopping — train for fixed rounds
        m = lgb.train(params, dtr, num_boost_round=800,
                      callbacks=[lgb.log_evaluation(9999)])
    iters = m.best_iteration if (early_stopping_rounds is not None and m.best_iteration > 0) else m.num_trees()
    print(f"  ✓ {tag:<30} iter={iters:>4}  t={time.time()-t0:.1f}s")
    return m


def _fit_catboost(X_tr, y_tr, X_val, y_val, art_tr, art_val,
                  art_med_tr, art_med_val, w_tr, n_jobs, feat_cols=None):
    t0  = time.time()
    Xc_tr  = np.hstack([art_tr.reshape(-1,1),  art_med_tr.reshape(-1,1),  X_tr.astype(object)])
    Xc_val = np.hstack([art_val.reshape(-1,1), art_med_val.reshape(-1,1), X_val.astype(object)])
    names  = ["artist", "artist_medium"] + (feat_cols or FEATURE_COLS)
    pool_tr  = Pool(Xc_tr,  label=y_tr,  cat_features=[0, 1], weight=w_tr,  feature_names=names)
    pool_val = Pool(Xc_val, label=y_val, cat_features=[0, 1], feature_names=names)
    m = CatBoostRegressor(
        loss_function="MAE", eval_metric="MAE",
        iterations=2000, learning_rate=0.05, depth=7,
        l2_leaf_reg=8, min_data_in_leaf=65, rsm=0.75,
        subsample=0.75, bootstrap_type="Bernoulli",
        early_stopping_rounds=40, use_best_model=True,
        thread_count=n_jobs, random_seed=42, verbose=False,
        # Don't litter a catboost_info/ dir. Its early-stopping eval_set is the
        # VALIDATION fold (not the held-out test set) — suppress to avoid confusion.
        allow_writing_files=False,
    )
    m.fit(pool_tr, eval_set=pool_val, silent=True)
    print(f"  ✓ {'CatBoost (artist + medium cats)':<30} iter={m.best_iteration_:>4}  t={time.time()-t0:.1f}s")
    return m, names


def cb_predict(model, X, art, art_med=None):
    if art_med is not None:
        Xc = np.hstack([art.reshape(-1,1), art_med.reshape(-1,1), X.astype(object)])
    else:
        Xc = np.hstack([art.reshape(-1,1), X.astype(object)])
    return np.array(model.predict(Xc), dtype=np.float32)


# ── OOF meta-stacking ──────────────────────────────────────────────────────

def build_oof_predictions(X_tr, y_tr, art_tr, art_med_tr, w_tr, X_val, y_val, art_val, art_med_val, n_folds=5, feat_cols=None):
    """
    Temporal cross-validation to generate out-of-fold predictions
    from each base model (LGBM, CatBoost).

    Folds are temporal (no shuffling) — each fold uses progressively more
    training data so the OOF distribution matches deployment.

    ⚠️ DISCLOSURE: each fold's base learner currently uses the held-out OOF block
    itself as the early-stopping eval_set, so the *number of boosting iterations*
    is selected with knowledge of those rows (the model is not fitted on their
    labels, but the stopping point is informed by them). These are therefore
    "out-of-training-fold predictions with fold-informed early stopping," not
    perfectly untouched OOF. The practical effect on the downstream NNLS weights is
    small (final metrics are measured on a separate val/test timeline), but a fully
    rigorous version would early-stop on an inner tail of each fold's own training
    data instead. The NNLS stack is only used to set 2 ensemble weights.

    Returns:
      oof_preds: np.ndarray of shape (len(X_tr), 2) — train OOF predictions
    """
    n = len(X_tr)
    fold_size = n // n_folds
    oof_preds = np.full((n, 2), np.nan, dtype=np.float32)

    print(f"  Building OOF predictions ({n_folds} temporal folds)...")
    t_oof = time.time()

    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end   = val_start + fold_size if fold_idx < n_folds - 1 else n

        # Use all data before the validation fold as training
        if val_start == 0:
            # First fold has no prior data to train on → no valid OOF prediction.
            # Leave it as NaN so fit_meta_learner excludes these rows from the NNLS
            # fit (previously zero-filled, which is inert for NNLS but misleading).
            continue

        tr_idx  = np.arange(0, val_start)
        val_idx = np.arange(val_start, val_end)

        Xf_tr,  yf_tr  = X_tr[tr_idx],  y_tr[tr_idx]
        Xf_val, yf_val = X_tr[val_idx], y_tr[val_idx]
        artf_tr     = art_tr[tr_idx]
        artf_val    = art_tr[val_idx]
        art_medf_tr  = art_med_tr[tr_idx]
        art_medf_val = art_med_tr[val_idx]
        wf_tr        = w_tr[tr_idx]

        # Train each base model on the fold
        fold_results = {}

        def _lgbm_fold():
            m = _fit_lgbm(Xf_tr, yf_tr, Xf_val, yf_val, wf_tr,
                          f"LGBM fold-{fold_idx}", max(1, N_EACH//2),
                          feature_cols=feat_cols)
            return "lgbm", m.predict(Xf_val)

        def _cat_fold():
            m, _ = _fit_catboost(Xf_tr, yf_tr, Xf_val, yf_val,
                                  artf_tr, artf_val, art_medf_tr, art_medf_val,
                                  wf_tr, max(1, N_EACH//2), feat_cols=feat_cols)
            return "cat", cb_predict(m, Xf_val, artf_val, art_medf_val)

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(_lgbm_fold), ex.submit(_cat_fold)]
            for fut in as_completed(futs):
                k, v = fut.result()
                fold_results[k] = v

        oof_preds[val_idx, 0] = fold_results["lgbm"].astype(np.float32)
        oof_preds[val_idx, 1] = fold_results["cat"].astype(np.float32)

        fold_medape = np.median(np.abs(
            np.expm1(yf_val) - np.expm1(
                np.nanmean(oof_preds[val_idx], axis=1)
            )
        ) / np.expm1(yf_val).clip(1)) * 100
        print(f"    fold-{fold_idx} done  val_idx=[{val_start}:{val_end}]  "
              f"mean-MedAPE≈{fold_medape:.1f}%")

    # First-fold rows stay NaN (no valid OOF) and are dropped by the meta-learner.
    print(f"  OOF done in {time.time()-t_oof:.1f}s  "
          f"({int((~np.isnan(oof_preds).any(axis=1)).sum()):,}/{n:,} rows usable)")
    return oof_preds


def fit_meta_learner(oof_preds, y_tr):
    """
    Non-negative least squares meta-learner.
    Solves: min ||oof_preds @ w - y_tr||^2  s.t. w >= 0
    Then normalises weights to sum to 1.
    NNLS avoids the negative-coefficient instability of Ridge when
    the base-model predictions are highly correlated.
    """
    from scipy.optimize import nnls
    # Drop rows with no valid OOF prediction (the first temporal fold) so the
    # meta-learner is fit only on genuine out-of-fold predictions.
    valid = ~np.isnan(oof_preds).any(axis=1)
    w, _ = nnls(oof_preds[valid], y_tr[valid])
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]
    names = ["lgbm", "cat"]
    print("  NNLS meta-learner weights: " +
          "  ".join(f"{n}={v:.3f}" for n, v in zip(names, w)))
    return w   # plain ndarray — no sklearn object needed


# ── isotonic calibration ───────────────────────────────────────────────────

def fit_tiered_calibrator(y_true, y_pred, est):
    """
    Fit per-price-tier isotonic calibrators.
    Each price band gets its own IsotonicRegression so systematic biases
    that differ by value tier (e.g. <$10k lots vs >$1M lots) are corrected
    independently.

    Returns dict: tier_name → IsotonicRegression  (with "global" fallback).
    """
    tier_defs = [
        ("lt_10k",    est < 10_000),
        ("10k_50k",  (est >= 10_000)  & (est < 50_000)),
        ("50k_200k", (est >= 50_000)  & (est < 200_000)),
        ("200k_1m",  (est >= 200_000) & (est < 1_000_000)),
        ("gt_1m",    est >= 1_000_000),
    ]
    # Global fallback (used for any tier with insufficient data)
    global_cal = IsotonicRegression(out_of_bounds="clip", increasing=True)
    global_cal.fit(y_pred, y_true)
    calibrators: dict = {"global": global_cal}

    for tier_name, mask in tier_defs:
        if mask.sum() >= 50:
            cal = IsotonicRegression(out_of_bounds="clip", increasing=True)
            cal.fit(y_pred[mask], y_true[mask])
            calibrators[tier_name] = cal
        else:
            calibrators[tier_name] = global_cal

    return calibrators


def apply_tiered_calibration(calibrators: dict, y_pred: np.ndarray,
                              est: np.ndarray) -> np.ndarray:
    """Apply per-tier isotonic calibration to prediction array."""
    y_cal = y_pred.copy().astype(np.float64)
    tier_defs = [
        ("lt_10k",    est < 10_000),
        ("10k_50k",  (est >= 10_000)  & (est < 50_000)),
        ("50k_200k", (est >= 50_000)  & (est < 200_000)),
        ("200k_1m",  (est >= 200_000) & (est < 1_000_000)),
        ("gt_1m",    est >= 1_000_000),
    ]
    for tier_name, mask in tier_defs:
        if mask.any():
            cal = calibrators.get(tier_name, calibrators["global"])
            y_cal[mask] = cal.predict(y_pred[mask])
    return y_cal.astype(np.float32)


# ── group-conditional bias correction (medium / house) ─────────────────────
#
# Tiered isotonic calibration above corrects bias as a function of price
# tier only. Residual analysis (2026-06-10) showed systematic bias that's
# ORTHOGONAL to price tier:
#   - by medium:  photographs/prints/watercolors under-predicted by ~15-22pp
#                  median residual vs ~0-5pp for paintings/screenprints
#   - by house:   Christie's under-predicted by ~11pp median vs ~3pp for
#                  Phillips/Bonhams
# This stage applies a small additive shift per medium/house group, fit
# AFTER tiered calibration, with empirical-Bayes shrinkage so small groups
# (e.g. sotheby's, n~tiny) don't get noisy corrections.

def fit_group_bias_correction(y_true: np.ndarray, y_pred_cal: np.ndarray,
                               groups: np.ndarray, min_n: int = 20,
                               k: float = 30.0) -> dict:
    """
    Fit a shrunk additive bias correction per group value (e.g. medium or
    house), in log_ratio space, applied AFTER tiered isotonic calibration.

    For each group g with n_g samples:
        correction[g] = (n_g / (n_g + k)) * mean(y_true - y_pred_cal | group==g)

    Groups with n_g < min_n get no correction (handled naturally by the
    shrinkage formula too, but min_n avoids near-zero-sample groups).

    Returns dict: group value (str, lowercased) -> additive correction
    (float), plus "_default": 0.0 for unseen groups at inference time.
    """
    resid  = (y_true.astype(np.float64) - y_pred_cal.astype(np.float64))
    # Lowercase/strip so keys match predict.py's normalized medium/house
    # strings (clean_data.py's medium fallback can preserve mixed case for
    # media that don't match any pattern in _MEDIUM_MAP).
    groups = np.array([str(g).lower().strip() for g in groups])
    out: dict = {"_default": 0.0}
    for g in np.unique(groups):
        mask = groups == g
        n = int(mask.sum())
        if n < min_n:
            continue
        r_mean = float(resid[mask].mean())
        out[g] = (n / (n + k)) * r_mean
    return out


def apply_group_bias_correction(corrections: dict, y_pred: np.ndarray,
                                 groups: np.ndarray) -> np.ndarray:
    """Apply additive group-bias correction from fit_group_bias_correction."""
    default = corrections.get("_default", 0.0)
    groups  = np.array([str(g).lower().strip() for g in groups])
    corr    = np.array([corrections.get(g, default) for g in groups], dtype=np.float64)
    return (y_pred.astype(np.float64) + corr).astype(np.float32)


# ── conformal prediction helpers ───────────────────────────────────────────

_TIER_NAMES = ["lt_10k", "10k_50k", "50k_200k", "200k_1m", "gt_1m"]

def _tier_mask(tier_name: str, est: np.ndarray) -> np.ndarray:
    """Return boolean mask selecting the given price tier."""
    if tier_name == "lt_10k":    return est < 10_000
    if tier_name == "10k_50k":   return (est >= 10_000)  & (est < 50_000)
    if tier_name == "50k_200k":  return (est >= 50_000)  & (est < 200_000)
    if tier_name == "200k_1m":   return (est >= 200_000) & (est < 1_000_000)
    if tier_name == "gt_1m":     return est >= 1_000_000
    return np.ones(len(est), dtype=bool)


# ── Stage-C: estimate-anchored shrink ───────────────────────────────────────
# The specialist's low–high estimate is a near-efficient baseline. The ensemble's
# deviation from it (the predicted log_ratio) helps squared-error/variance on
# high-value lots, but can hurt the *typical* lot's MedAPE. So we blend the
# calibrated prediction back toward the estimate (log_ratio = 0) by a factor
# λ ∈ [0, 1], chosen on VALIDATION to minimise dollar MedAPE:
#       final_log_ratio = λ · calibrated_log_ratio
# Because λ = 0 (pure estimate) is always in the search grid, the result is
# *provably no worse than the estimate baseline on val* — and keeps the model's
# lift only where it actually pays off. A global λ is always fit; a per-tier λ
# overrides only when that tier has enough val support AND beats the global λ on
# that tier by a margin (guards against overfitting thin tiers).

def _dollar_ape(p_lr, y_lr, est):
    """Absolute percentage error in dollar space (per-lot array) from log_ratio preds."""
    h_true = np.exp(np.clip(np.asarray(y_lr, np.float64), -5, 5)) * np.asarray(est, np.float64)
    h_pred = np.exp(np.clip(np.asarray(p_lr, np.float64), -5, 5)) * np.asarray(est, np.float64)
    return np.abs(h_true - h_pred) / np.clip(h_true, 1, None)


def _dollar_medape(p_lr, y_lr, est) -> float:
    """Median absolute percentage error in dollar space from log_ratio preds."""
    return float(np.median(_dollar_ape(p_lr, y_lr, est)) * 100)


def _dollar_within20(p_lr, y_lr, est) -> float:
    """Fraction (%) of lots predicted within 20% of the true hammer."""
    return float((_dollar_ape(p_lr, y_lr, est) <= 0.20).mean() * 100)


def fit_estimate_shrink(y_lr, p_lr, est, k_shrink: int = 200,
                        min_fit_n: int = 40, min_gain: float = 0.5,
                        w20_tol: float = 0.25) -> dict:
    """Fit Stage-C shrink factor(s) on validation — confidence-adaptive by tier.

    Grid-searches λ ∈ [0,1] to minimise dollar MedAPE (λ=0 = pure estimate is
    in-grid, so each tier is ≤ its estimate-only MedAPE). A tier only deviates
    from the estimate (λ>0) when there's real evidence to:

      1. EVIDENCE GATE — the best λ must beat λ=0 (the estimate) on that val tier
         by at least `min_gain` pp; otherwise λ=0 (defer). This stops the model
         from deviating on a noise-level val edge that won't hold out-of-sample —
         the val/test non-stationarity that made some tiers slip below the estimate.
      2. CONFIDENCE WEIGHT — the surviving λ is shrunk toward 0 by an
         empirical-Bayes weight w = n/(n+k_shrink); thin tiers trust the estimate
         more (this also fixes the >$1M leak — sparse tiers can't inherit a large λ).

    MULTI-OBJECTIVE: MedAPE is primary (it carries the "no worse than the estimate"
    guarantee). Within the set of λ whose val MedAPE is within `w20_tol` pp of the
    best — and never worse than the estimate (λ=0) — we pick the λ that maximises
    within-20% hit-rate. So we never trade away MedAPE beyond a tiny tolerance, but
    we recover hit-rate where it's free.

    Returns {"global": λ, <tier>: λ, ...} with an explicit λ for every tier."""
    y_lr = np.asarray(y_lr, np.float64)
    p_lr = np.asarray(p_lr, np.float64)
    est  = np.asarray(est,  np.float64)
    grid = np.linspace(0.0, 1.0, 41)

    def best_lambda(mask):
        yy, pp, ee = y_lr[mask], p_lr[mask], est[mask]
        med = np.array([_dollar_medape(lam * pp, yy, ee) for lam in grid])
        w20 = np.array([_dollar_within20(lam * pp, yy, ee) for lam in grid])
        best_med = float(med.min())
        med0 = float(med[0])                        # λ=0 = the estimate baseline
        # λ candidates: within tol of the best MedAPE AND no worse than the estimate.
        cand = np.where((med <= best_med + w20_tol) & (med <= med0 + 1e-9))[0]
        if len(cand) == 0:
            cand = np.array([int(np.argmin(med))])
        # Among MedAPE-acceptable λ, take the one with the best within-20% hit-rate.
        j = int(cand[int(np.argmax(w20[cand]))])
        # val MedAPE gain of the chosen λ over λ=0 (the estimate) on this tier
        gain = float(med0 - med[j])
        return float(grid[j]), gain

    g_lam, _ = best_lambda(np.ones(len(y_lr), dtype=bool))
    shrink = {"global": g_lam}
    for t in _TIER_NAMES:
        m = _tier_mask(t, est)
        n = int(m.sum())
        lam, gain = best_lambda(m) if n >= min_fit_n else (0.0, 0.0)
        if n < min_fit_n or gain < min_gain:
            shrink[t] = 0.0                    # thin or no real edge → defer to estimate
        else:
            w = n / (n + k_shrink)             # empirical-Bayes confidence weight
            shrink[t] = round(w * lam, 4)
    return shrink


def apply_estimate_shrink(shrink: dict, p_lr, est):
    """Apply Stage-C shrink to a calibrated log_ratio array (global λ, with
    per-tier overrides where present)."""
    if not shrink:
        return p_lr
    p_lr = np.asarray(p_lr, np.float64)
    est  = np.asarray(est)
    out  = p_lr * shrink.get("global", 1.0)
    for t, lam in shrink.items():
        if t == "global":
            continue
        m = _tier_mask(t, est)
        out[m] = p_lr[m] * lam
    return out.astype(np.float32)


def fit_cross_conformal(ensemble_fn, X, y, est, art, art_med, lo,
                        groups_medium=None, groups_house=None,
                        estimate_shrink=None,
                        K: int = 10, alpha_lo: float = 0.10,
                        alpha_hi: float = 0.90, seed: int = 42):
    """
    K-fold stratified cross-conformal prediction intervals.

    Algorithm
    ---------
    For each of K stratified folds:
      1. Split val into 90 % cal fold / 10 % conf fold,
         stratified by price tier → balanced coverage across tiers.
      2. Fit per-tier isotonic calibrators on the cal fold.
      2b. Fit medium/house group-bias corrections on the cal fold
          (if groups_medium / groups_house provided) — mirrors the
          two-stage calibration used for point predictions, so conformal
          residuals are computed against the SAME calibrated prediction
          that's reported as the headline point estimate.
      3. Compute calibrated residuals  r = y − ŷ_cal  on the conf fold.
      4. Extract P10 / P90 quantiles (global + per-tier).
    Final offsets = mean over K folds.  Stability = std over K folds.

    Properties
    ----------
    • Satisfies exchangeability (i.i.d. within each fold) — theoretically valid.
    • Uses ≈ 90 % of val for calibration per fold → near-maximal accuracy.
    • Averaging over K folds removes single-split randomness.
    • Mondrian property: per-tier offsets achieve conditional coverage.

    Reference: Vovk et al. (2005); Angelopoulos & Bates (2023 tutorial).

    Parameters
    ----------
    ensemble_fn : callable  (X, art, art_med, lo) → raw predictions
    K           : number of folds (default 10)
    """
    from sklearn.model_selection import StratifiedKFold

    # Stratify by price tier label so every fold is tier-balanced
    tier_labels = np.digitize(est, [10_000, 50_000, 200_000, 1_000_000])

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)

    global_lo_k: list[float] = []
    global_hi_k: list[float] = []
    tier_lo_k = {t: [] for t in _TIER_NAMES}
    tier_hi_k = {t: [] for t in _TIER_NAMES}

    for cal_idx, conf_idx in skf.split(X, tier_labels):
        # ── fit fold calibrators on 90 % ─────────────────────────────
        p_ens_cal = ensemble_fn(X[cal_idx], art[cal_idx],
                                art_med[cal_idx], lo[cal_idx])
        cal_k = fit_tiered_calibrator(y[cal_idx], p_ens_cal, est[cal_idx])
        p_cal_cal = apply_tiered_calibration(cal_k, p_ens_cal, est[cal_idx])

        # ── 2b. medium/house group-bias correction, fit on cal fold ──
        med_corr_k = house_corr_k = None
        if groups_medium is not None:
            med_corr_k = fit_group_bias_correction(
                y[cal_idx], p_cal_cal, groups_medium[cal_idx])
            p_cal_cal = apply_group_bias_correction(
                med_corr_k, p_cal_cal, groups_medium[cal_idx])
        if groups_house is not None:
            house_corr_k = fit_group_bias_correction(
                y[cal_idx], p_cal_cal, groups_house[cal_idx])

        # ── calibrated residuals on 10 % holdout ─────────────────────
        p_ens_conf = ensemble_fn(X[conf_idx], art[conf_idx],
                                 art_med[conf_idx], lo[conf_idx])
        p_cal_conf = apply_tiered_calibration(cal_k, p_ens_conf, est[conf_idx])
        if med_corr_k is not None:
            p_cal_conf = apply_group_bias_correction(
                med_corr_k, p_cal_conf, groups_medium[conf_idx])
        if house_corr_k is not None:
            p_cal_conf = apply_group_bias_correction(
                house_corr_k, p_cal_conf, groups_house[conf_idx])
        # Stage-C shrink: conformal residuals must center on the SAME shrunk
        # prediction reported as the point estimate, or coverage drifts.
        if estimate_shrink:
            p_cal_conf = apply_estimate_shrink(
                estimate_shrink, p_cal_conf, est[conf_idx])
        resid = (y[conf_idx] - p_cal_conf).astype(np.float64)

        # Global quantiles
        global_lo_k.append(float(np.quantile(resid, alpha_lo)))
        global_hi_k.append(float(np.quantile(resid, alpha_hi)))

        # Per-tier quantiles
        for t in _TIER_NAMES:
            m = _tier_mask(t, est[conf_idx])
            if m.sum() >= 20:
                tier_lo_k[t].append(float(np.quantile(resid[m], alpha_lo)))
                tier_hi_k[t].append(float(np.quantile(resid[m], alpha_hi)))

    # ── aggregate ────────────────────────────────────────────────────
    lo_q    = float(np.mean(global_lo_k))
    hi_q    = float(np.mean(global_hi_k))
    lo_std  = float(np.std(global_lo_k))
    hi_std  = float(np.std(global_hi_k))

    print(f"  Cross-conformal (K={K}, stratified by tier, α=0.10/0.90):")
    print(f"  Global:  P10={lo_q:+.3f} ± {lo_std:.3f}  "
          f"P90={hi_q:+.3f} ± {hi_std:.3f}  (mean ± std, {K} folds)")

    tier_conformal: dict = {}
    print("  Tier-specific conformal intervals:")
    for t in _TIER_NAMES:
        if tier_lo_k[t]:
            t_lo     = float(np.mean(tier_lo_k[t]))
            t_hi     = float(np.mean(tier_hi_k[t]))
            t_lo_std = float(np.std(tier_lo_k[t]))
            t_hi_std = float(np.std(tier_hi_k[t]))
            n_folds  = len(tier_lo_k[t])
            tier_conformal[t] = {
                "lo": t_lo, "hi": t_hi,
                "lo_std": t_lo_std, "hi_std": t_hi_std,
            }
            print(f"    {t:<12}  "
                  f"P10={t_lo:+.3f} ± {t_lo_std:.3f}  "
                  f"P90={t_hi:+.3f} ± {t_hi_std:.3f}  "
                  f"(K={n_folds})")

    return lo_q, hi_q, tier_conformal, lo_std, hi_std


# ── sell-through probability model ─────────────────────────────────────────

def train_sell_through(df_tr, df_val, n_jobs):
    """
    Binary classifier: predicts P(lot sells) from pre-sale features.
    Uses all FEATURE_COLS except log_estimate_mid (available pre-sale).
    """
    # Exclude features that require knowing the hammer price (post-sale leakage)
    feat_cols = [c for c in FEATURE_COLS if c not in ["log_estimate_mid"]]
    X_tr  = df_tr[feat_cols].fillna(0).values.astype(np.float32)
    y_tr  = df_tr["sold"].values.astype(np.float32)
    X_val = df_val[feat_cols].fillna(0).values.astype(np.float32)
    y_val = df_val["sold"].values.astype(np.float32)

    dtr  = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feat_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtr)
    m = lgb.train(
        dict(objective="binary", metric="auc", num_leaves=63,
             learning_rate=0.05, feature_fraction=0.75, n_jobs=n_jobs,
             verbose=-1, seed=42),
        dtr, num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(9999)],
    )
    preds_val = m.predict(X_val)
    from sklearn.metrics import roc_auc_score
    auc      = roc_auc_score(y_val, preds_val)
    accuracy = ((preds_val > 0.5) == y_val).mean()
    print(f"  Sell-through model: AUC={auc:.4f}  Accuracy={accuracy*100:.1f}%")
    return m


# ── ablation study ──────────────────────────────────────────────────────────

def run_ablation(X_tr, y_tr, X_val, y_val, w_tr, full_medape):
    """
    Compare val MedAPE when feature groups / individual features are zeroed out.

    BASELINE FIX (was comparing apples to oranges): every ablated variant here
    is scored with a quick single-LightGBM fit (`medape_fast`), which lands
    ~11pp worse than the calibrated LGBM+CatBoost ensemble simply
    because it skips ensembling and tiered calibration — not because of any
    feature. The old code diffed `medape_fast(ablated)` against `full_medape`
    (the *calibrated ensemble's* val MedAPE), which bakes that ~11pp
    architecture/calibration gap into every single delta — swamping the true
    per-feature effect (typically a few tenths of a pp) and making every
    feature look like "KEEP, +10pp" regardless of whether it does anything.
    Fix: first measure a quick-fit baseline on the FULL (unablated) matrix
    (`base_medape`) and diff every ablated variant against THAT — same
    architecture, same (lack of) calibration, genuinely apples-to-apples.
    `full_medape` is retained only to print contextually (how the quick-fit
    proxy relates to the real deployed system).

    Part i : 4 group ablations  (artist features, provenance, dimensions, estimate-only).
    Part ii: 3 watch-feature ablations (is_evening_sale, is_online_sale, house_cat).
    """
    def medape_fast(X_t, y_t, X_v, y_v, w_t, tag):
        m = _fit_lgbm(X_t, y_t, X_v, y_v, w_t, tag, max(1, N_EACH), n_leaves=127)
        p = m.predict(X_v)
        return float(np.median(np.abs(np.expm1(y_v) - np.expm1(p))
                               / np.expm1(y_v).clip(1)) * 100)

    # Consistent baseline: quick single-LGBM fit on the *unablated* matrix.
    # Every delta below is measured against this number (not full_medape).
    base_medape = medape_fast(X_tr, y_tr, X_val, y_val, w_tr, "Full (ablation baseline)")
    print(f"\n  [baseline check]  quick-fit full model = {base_medape:.1f}%   "
          f"calibrated ensemble = {full_medape:.1f}%   "
          f"(Δ={base_medape - full_medape:+.1f}pp is architecture/calibration gap, "
          f"NOT a feature effect — every delta below is vs. {base_medape:.1f}%)")

    # ── Part i: Group ablation ────────────────────────────────────────────
    groups = {
        "No artist features":  [i for i, c in enumerate(FEATURE_COLS)
                                 if c.startswith("artist_")],
        "No provenance":       [i for i, c in enumerate(FEATURE_COLS)
                                 if c in ("signed","exhibited","literature",
                                          "has_provenance","provenance_len",
                                          "prov_has_museum","prov_auction_count")],
        "No dimensions":       [i for i, c in enumerate(FEATURE_COLS)
                                 if c in ("log_area","log_width","log_height",
                                          "has_dims","aspect_ratio")],
        "Estimate only":       [i for i, c in enumerate(FEATURE_COLS)
                                 if c != "log_estimate_mid"],
    }

    print("\n── Step 10b-i: Ablation study (feature groups) ──────────────────────")
    print(f"  {'Feature group':<22}  {'Val MedAPE':>10}  {'Δ from full':>12}")
    print(f"  {'Full model':<22}  {base_medape:>9.1f}%  {'—':>12}")

    results = {}

    def _run_group(name, idxs):
        Xa_tr  = X_tr.copy();  Xa_tr[:, idxs]  = 0.0
        Xa_val = X_val.copy(); Xa_val[:, idxs] = 0.0
        medape = medape_fast(Xa_tr, y_tr, Xa_val, y_val, w_tr,
                             f"Ablation: {name[:20]}")
        return name, medape

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_run_group, name, idxs): name
                for name, idxs in groups.items()}
        for fut in as_completed(futs):
            name, medape = fut.result()
            results[name] = medape

    for name in groups:   # print in consistent order
        medape = results[name]
        delta  = medape - base_medape
        print(f"  {name:<22}  {medape:>9.1f}%  {delta:>+11.1f}pp")

    # ── Part ii: Watch-feature ablation (individual) ─────────────────────
    # Zero out each borderline feature individually to isolate its contribution.
    # Interpretation:
    #   delta > +0.1pp  → feature carries real signal (removing it hurts)  → KEEP
    #   delta < -0.1pp  → feature adds noise (removing it helps)            → DROP
    #   |delta| ≤ 0.1pp → neutral / within noise floor                     → BORDERLINE
    watch_features = ["house_cat", "is_evening_sale", "is_online_sale"]
    watch_idxs = {f: FEATURE_COLS.index(f) for f in watch_features if f in FEATURE_COLS}

    print("\n── Step 10b-ii: Watch-feature ablation (individual) ────────────────")
    print(f"  {'Feature':<22}  {'Val MedAPE':>10}  {'Δ from full':>12}  {'Verdict':>10}")
    print(f"  {'Full model':<22}  {base_medape:>9.1f}%  {'—':>12}  {'—':>10}")

    watch_results = {}

    def _run_single(name, idx):
        Xa_tr  = X_tr.copy();  Xa_tr[:, idx]  = 0.0
        Xa_val = X_val.copy(); Xa_val[:, idx] = 0.0
        medape = medape_fast(Xa_tr, y_tr, Xa_val, y_val, w_tr,
                             f"Ablation: {name[:20]}")
        return name, medape

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_run_single, name, idx): name
                for name, idx in watch_idxs.items()}
        for fut in as_completed(futs):
            name, medape = fut.result()
            watch_results[name] = medape

    for name in watch_features:
        if name not in watch_results:
            print(f"  {name:<22}  {'(not in FEATURE_COLS)':>34}")
            continue
        medape = watch_results[name]
        delta  = medape - base_medape
        verdict = ("KEEP" if delta > 0.10
                   else "DROP" if delta < -0.10
                   else "borderline")
        print(f"  {name:<22}  {medape:>9.1f}%  {delta:>+11.2f}pp  {verdict:>10}")


# ── main ───────────────────────────────────────────────────────────────────

def train(data_path="data/art_auction_clean.csv",
          models_dir="models", force_recompute=False,
          deploy=False, deploy_cal_months=3,
          fold_dates=None, return_fold_preds=False):
    """
    fold_dates: optional (train_end, val_start, val_end, test_end) timestamps for
        walk-forward CV. When given, overrides the default temporal_split with these
        windows (train < train_end; val in [val_start, val_end); test in [val_end,
        test_end)) and uses train_end as the leakage-safe feature cutoff. The gap
        [train_end, val_start) is an embargo. Mutually exclusive with deploy.
    return_fold_preds: when True (walk-forward use), return the test-fold predictions
        DataFrame + metrics and skip all artifact/meta writing.
    """

    os.makedirs(models_dir, exist_ok=True)
    T0 = time.time()

    mode = ("WALK-FORWARD fold" if fold_dates is not None else
            "DEPLOY (refit-all)" if deploy else "EVAL (held-out test)")
    print("\n═══════════════════════════════════════════════════════════")
    print("  Art Auction — Training Pipeline  (parallel, fast)")
    print(f"  CPUs={N_JOBS}  per-model={N_EACH}  LowValThreshold=${LOW_VAL:,}")
    print(f"  Mode: {mode}")
    print("═══════════════════════════════════════════════════════════\n")

    # In deploy mode, preserve the honest held-out metrics from a prior EVAL run
    # (models/meta.json) so the served artifact still carries its forward estimate.
    prior_holdout = None
    prior_display = None
    if deploy:
        _meta_path = os.path.join(models_dir, "meta.json")
        if os.path.exists(_meta_path):
            try:
                with open(_meta_path) as _f:
                    _prev = json.load(_f)
                # Only carry forward genuine held-out numbers (not another deploy run's)
                if not _prev.get("deploy", False) and _prev.get("test_metrics"):
                    prior_holdout = _prev["test_metrics"]
                    # The honest model-vs-estimate display block from the eval run —
                    # carried so the served app shows held-out numbers, not the
                    # in-sample cal-tail it would otherwise recompute.
                    prior_display = _prev.get("display")
            except Exception:
                pass
        if prior_holdout is None:
            print("  ⚠ DEPLOY: no prior EVAL meta.json with held-out test_metrics found.")
            print("    Run `python -m ml.train` (eval mode) FIRST so the deploy model can")
            print("    carry its honest forward-performance estimate. Continuing anyway.\n")

    # ── 1. Data ───────────────────────────────────────────────────────────
    # Leakage-safe boundary for global fallbacks + medium imputation: in EVAL mode
    # the held-out period starts 2025-01-01 (temporal_split trains on ≤2024), so
    # those constants are computed only from earlier lots. In DEPLOY mode there is
    # no held-out test → use all data (train_end=None).
    if fold_dates is not None:
        feat_train_end = pd.Timestamp(fold_dates[0])
    else:
        feat_train_end = None if deploy else pd.Timestamp("2025-01-01")
    cache = os.path.join(models_dir, "features_cache.parquet")
    if not force_recompute and os.path.exists(cache):
        print("── Step 1: Load cached features ──────────────────────────────")
        df = pd.read_parquet(cache)
        print(f"  {len(df):,} rows  ({time.time()-T0:.1f}s)")
    else:
        print("── Step 1: Compute features ──────────────────────────────────")
        df = compute_artist_features(
            load_data(data_path, train_end=feat_train_end), train_end=feat_train_end)
        df.to_parquet(cache, index=False)

    # ── 2. Split + prep ───────────────────────────────────────────────────
    print("\n── Step 2: Split ─────────────────────────────────────────────")
    train_df, val_df, test_df = temporal_split(df)

    if fold_dates is not None:
        # Walk-forward fold: train < train_end, embargo gap, val in [val_start,
        # val_end), test in [val_end, test_end). Same downstream pipeline as EVAL.
        _tre, _vs, _ve, _tte = [pd.Timestamp(x) for x in fold_dates]
        train_df = df[df["sale_date"] < _tre]
        val_df   = df[(df["sale_date"] >= _vs) & (df["sale_date"] < _ve)]
        test_df  = df[(df["sale_date"] >= _ve) & (df["sale_date"] < _tte)]
        print(f"  [WF] train(<{_tre.date()})={len(train_df):,}  "
              f"val[{_vs.date()},{_ve.date()})={len(val_df):,}  "
              f"test[{_ve.date()},{_tte.date()})={len(test_df):,}")
    elif deploy:
        # Deploy artifact: train the GBM ensemble on ALL labelled data except a
        # thin recent tail, which is held out SOLELY to fit calibration + conformal
        # (those must never be fit on data the model trained on). No honest
        # forward test set survives — that's the EVAL run's job — so we alias
        # test_df = val_df: the whole downstream pipeline runs unchanged on the
        # tail, but the resulting "test" numbers are IN-SAMPLE and are NOT
        # published (see meta: test_metrics=null, holdout_eval_metrics carried
        # forward from the prior eval run).
        _sold = df[(df["sold"] == 1) & df["sale_date"].notna()]
        max_date = _sold["sale_date"].max()
        cutoff   = max_date - pd.DateOffset(months=deploy_cal_months)
        train_df = df[df["sale_date"] <= cutoff]
        val_df   = df[df["sale_date"] >  cutoff]
        test_df  = val_df                      # alias: in-sample tail, metrics suppressed
        print(f"  [DEPLOY] trained_through={max_date.date()}  cal-tail cutoff={cutoff.date()} "
              f"({deploy_cal_months}mo)")
        print(f"  [DEPLOY] train(≤cutoff)={len(train_df):,}  cal-tail(>cutoff)={len(val_df):,}  "
              f"test=cal-tail (in-sample, NOT published)")

    # Full prep (includes <$10k) — used for low-value specialist model
    X_full, y_full, _, _, _, lo_full, w_full, s_full = prep(
        train_df, use_weights=True)

    # Main training: exclude <$10k + trim 0.5% outlier log_ratios
    X_tr, y_tr, est_tr, art_tr, art_med_tr, lo_tr, w_tr, s_tr = prep(
        train_df, use_weights=True, exclude_low=True, trim_outliers=True)

    # Low-value training subset (from full prep, no trimming)
    lo_tr_X   = X_full[lo_full];  lo_tr_y = y_full[lo_full]
    w_lo      = w_full[lo_full]

    X_val, y_val, est_val, art_val, art_med_val, lo_val, _, s_val = prep(val_df)
    X_te,  y_te,  est_te,  art_te,  art_med_te,  lo_te,  _, s_te  = prep(test_df)

    lo_val_X = X_val[lo_val]; lo_val_y = y_val[lo_val]

    print(f"  main train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}")
    print(f"  low-val specialist:  train={lo_full.sum():,}  val={lo_val.sum():,}")

    # ── 2.5. Sell-through as feature + augment all X matrices ─────────────────
    # The sell_through model is trained on FEATURE_COLS minus log_estimate_mid (40 feats).
    # Its output (sell_through_prob) is appended as feature #42.
    # No circular dependency: sell_through model never sees sell_through_prob.
    #
    # LEAKAGE FIX: sell_through model was trained on train_df.  Predicting on train rows
    # with the same model creates mild self-leakage (overfit binary signal → lower train
    # MedAPE without matching val gain → wider gap).
    # Fix: temporal OOF for training rows — each row's sell_through_prob comes from a model
    # trained on strictly earlier data.  Val/test remain genuine OOS (no fix needed).
    print("\n── Step 2.5: Sell-through model + augment features ─────────")
    t25 = time.time()
    sell_through_model = train_sell_through(train_df, val_df, N_EACH)
    print(f"  Sell-through model done in {time.time()-t25:.1f}s")

    FEAT_NAMES     = FEATURE_COLS + ["sell_through_prob"]
    _st_feat_cols  = [c for c in FEATURE_COLS if c != "log_estimate_mid"]
    _st_drop       = FEATURE_COLS.index("log_estimate_mid")

    def _st_vec(X41):
        """OOS sell-through from a 41-feature matrix → (n,1)."""
        return sell_through_model.predict(
            np.delete(X41, _st_drop, axis=1).astype(np.float32)
        ).astype(np.float32).reshape(-1, 1)

    def _oof_st(sold_df, n_folds=5):
        """
        Temporal OOF sell-through for a filtered sold DataFrame.
        Fold k trains on all train_df rows with dates before fold k's start date,
        then predicts for rows of sold_df that fall in that date window.
        Fold 0 (insufficient prior data) falls back to the full model.
        """
        tr = train_df.sort_values("sale_date").reset_index(drop=True)
        n  = len(tr)
        fold_ends = [(k + 1) * (n // n_folds) for k in range(n_folds)]
        fold_ends[-1] = n
        # Date at which each fold starts (for assignment)
        fold_sdates = [tr["sale_date"].iloc[0]]
        for k in range(1, n_folds):
            fold_sdates.append(tr["sale_date"].iloc[fold_ends[k - 1]])

        sdf = sold_df.reset_index(drop=True)
        # Assign each row to the highest fold whose start date it exceeds
        fold_idx = np.zeros(len(sdf), dtype=int)
        for k in range(1, n_folds):
            fold_idx[sdf["sale_date"].values >= fold_sdates[k]] = k

        oof = np.full(len(sdf), np.nan, dtype=np.float32)
        for k in range(n_folds):
            mask = fold_idx == k
            if not mask.any():
                continue
            X_fold = sdf[mask][_st_feat_cols].fillna(0).values.astype(np.float32)
            if k == 0:
                # No prior data — use full model as fallback
                oof[mask] = sell_through_model.predict(X_fold).astype(np.float32)
                continue
            prior = tr.iloc[:fold_ends[k - 1]]
            m_k = lgb.train(
                dict(objective="binary", metric="auc", num_leaves=63,
                     learning_rate=0.05, feature_fraction=0.75,
                     n_jobs=max(1, N_EACH // 2), verbose=-1, seed=42),
                lgb.Dataset(
                    prior[_st_feat_cols].fillna(0).values.astype(np.float32),
                    label=prior["sold"].values.astype(np.float32),
                ),
                num_boost_round=200, callbacks=[lgb.log_evaluation(9999)],
            )
            oof[mask] = m_k.predict(X_fold).astype(np.float32)

        nan_mask = np.isnan(oof)
        if nan_mask.any():
            oof[nan_mask] = sell_through_model.predict(
                sdf[nan_mask][_st_feat_cols].fillna(0).values.astype(np.float32)
            ).astype(np.float32)
        return oof.reshape(-1, 1)

    t_oof_st = time.time()
    st_oof_tr   = _oof_st(s_tr)    # OOF → no leakage for main train rows
    st_oof_full = _oof_st(s_full)  # OOF → no leakage for lo-specialist rows
    print(f"  OOF sell-through done in {time.time()-t_oof_st:.1f}s")

    # Augment: training uses OOF predictions; val/test are already OOS
    X_tr   = np.hstack([X_tr,   st_oof_tr])
    X_full = np.hstack([X_full, st_oof_full])
    X_val  = np.hstack([X_val,  _st_vec(X_val)])
    X_te   = np.hstack([X_te,   _st_vec(X_te)])

    # Rebuild lo subsets from augmented arrays (labels/weights unchanged)
    lo_tr_X  = X_full[lo_full]
    lo_val_X = X_val[lo_val]

    print(f"  All matrices augmented → {len(FEAT_NAMES)} features total")

    # ── 3. Parallel base model training ───────────────────────────────────
    print("\n── Step 3: Train base models (parallel) ──────────────────────")
    t3 = time.time()
    results = {}

    def run_lgbm():
        return "lgbm", _fit_lgbm(X_tr, y_tr, X_val, y_val, w_tr, "LightGBM", N_EACH,
                                  feature_cols=FEAT_NAMES)
    def run_cat():
        return "cat", _fit_catboost(X_tr, y_tr, X_val, y_val,
                                    art_tr, art_val, art_med_tr, art_med_val, w_tr, N_EACH,
                                    feat_cols=FEAT_NAMES)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(run_lgbm), ex.submit(run_cat)]
        for fut in as_completed(futures):
            k, v = fut.result()
            results[k] = v

    lgbm_model                   = results["lgbm"]
    cat_model, cat_names         = results["cat"]
    print(f"  Base models done in {time.time()-t3:.1f}s (parallel)")

    # ── 4. Parallel quantile + low-value models ────────────────────────────
    print("\n── Step 4: Quantile + low-value models (parallel) ────────────")
    t4 = time.time()

    def run_q10():
        # Disable early stopping: val quantile loss at α=0.10 diverges immediately
        # due to train(2018-2024)/val(2025) lower-tail distribution shift — any split
        # that improves training q10 makes val q10 worse → iter=1 regardless of lr.
        # Fixed 800 rounds; CQR conformalization calibrates the output anyway.
        return "q10", _fit_lgbm(
            X_tr, y_tr, X_val, y_val, w_tr, "LGBM q10", N_EACH,
            n_leaves=63, alpha=0.10,
            extra_params={"learning_rate": 0.02, "min_child_samples": 50},
            early_stopping_rounds=None,
            feature_cols=FEAT_NAMES,
        )
    def run_q90():
        return "q90", _fit_lgbm(X_tr,y_tr,X_val,y_val,w_tr,"LGBM q90",N_EACH,
                                 n_leaves=63,alpha=0.90, feature_cols=FEAT_NAMES)
    def run_lo():
        if len(lo_tr_X) < 500:
            return "lo", None
        # Stronger regularisation for noisy low-value data:
        #   fewer leaves (simpler trees), heavier L1/L2, larger min-leaf.
        return "lo", _fit_lgbm(
            lo_tr_X, lo_tr_y, lo_val_X, lo_val_y,
            w_lo, "LGBM low-value", N_EACH, n_leaves=31,
            feature_cols=FEAT_NAMES,
            extra_params=dict(
                min_child_samples=100,
                lambda_l1=1.0, lambda_l2=1.0,
                feature_fraction=0.70,
            ),
        )
    aux = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        for fut in as_completed([ex.submit(run_q10), ex.submit(run_q90),
                                  ex.submit(run_lo)]):
            k, v = fut.result()
            aux[k] = v

    lgbm_q10  = aux["q10"]
    lgbm_q90  = aux["q90"]
    lo_model  = aux["lo"]
    print(f"  Auxiliary models done in {time.time()-t4:.1f}s (parallel)")

    # ── 4.25: CQR (Conformalized Quantile Regression) ────────────────────
    # Romano et al. (2019): conformalize the q10/q90 quantile models on val
    # to guarantee 90% marginal coverage with per-lot adaptive interval width.
    #
    # Score definitions:
    #   E_lo_i = q10_pred_i - y_i  (> 0 means q10 too high → lower bound down)
    #   E_hi_i = y_i - q90_pred_i  (> 0 means q90 too low  → upper bound up)
    # Correction at level 1-α = 0.90 → apply to every test prediction as:
    #   q10_corrected(x) = q10_pred(x) - cqr_lo_correction
    #   q90_corrected(x) = q90_pred(x) + cqr_hi_correction
    q10_preds_val = lgbm_q10.predict(X_val)
    q90_preds_val = lgbm_q90.predict(X_val)
    n_cqr   = len(y_val)
    q_level = min(1.0, np.ceil((n_cqr + 1) * 0.90) / n_cqr)   # finite-sample guarantee
    cqr_lo_correction = float(np.quantile(q10_preds_val - y_val, q_level))
    cqr_hi_correction = float(np.quantile(y_val - q90_preds_val, q_level))
    # Compute corrected val coverage for reporting
    cqr_q10_val = q10_preds_val - cqr_lo_correction
    cqr_q90_val = q90_preds_val + cqr_hi_correction
    cqr_cov_val = float(((y_val >= cqr_q10_val) & (y_val <= cqr_q90_val)).mean() * 100)
    print("\n── Step 4.25: CQR conformal correction ──────────────────────────")
    print(f"  lo_correction={cqr_lo_correction:+.4f}  hi_correction={cqr_hi_correction:+.4f}")
    print(f"  CQR val coverage: {cqr_cov_val:.1f}%  (asymmetric CQR, expected≈80%)")

    # ── 5. OOF meta-stacking ──────────────────────────────────────────────
    print("\n── Step 5: OOF meta-stacking (10-fold temporal) ──────────────")
    t5 = time.time()

    oof_preds = build_oof_predictions(
        X_tr, y_tr, art_tr, art_med_tr, w_tr,
        X_val, y_val, art_val, art_med_val,
        n_folds=10,
        feat_cols=FEAT_NAMES,
    )
    meta_learner = fit_meta_learner(oof_preds, y_tr)
    print(f"  Meta-stacking done in {time.time()-t5:.1f}s")

    # Build full-train predictions for val/test (used by meta-learner)
    def base_preds_matrix(X, art, art_med):
        pl  = lgbm_model.predict(X).astype(np.float32)
        pc  = cb_predict(cat_model, X, art, art_med)
        return np.column_stack([pl, pc])

    w_l, w_c = (float(meta_learner[i]) for i in range(2))
    print(f"  Final weights → lgbm={w_l:.3f} cat={w_c:.3f}")

    def ensemble(X, art, art_med, lo_mask=None):
        """Weighted ensemble using NNLS weights. Target: log_ratio."""
        pmat = base_preds_matrix(X, art, art_med)
        p = (pmat @ meta_learner).astype(np.float32)
        # Lo-value specialist overrides main ensemble for <$10k lots
        if lo_model is not None and lo_mask is not None and lo_mask.any():
            p[lo_mask] = lo_model.predict(X[lo_mask])
        return p

    # ── 6. Calibration + cross-conformal intervals ────────────────────
    print("\n── Step 6: Calibration + conformal intervals ─────────────────")

    # Fit final calibrators on ALL val — maximum calibration accuracy.
    # These are used for every downstream prediction.
    p_ens_val = ensemble(X_val, art_val, art_med_val, lo_val)
    calibrators = fit_tiered_calibrator(y_val, p_ens_val, est_val)
    p_cal_val_a = apply_tiered_calibration(calibrators, p_ens_val, est_val)
    print(f"  Tiered calibration: all {len(X_val):,} val points (5 price tiers)")
    print(f"  bias  before={np.mean(p_ens_val-y_val):+.4f}  "
          f"after tier-cal={np.mean(p_cal_val_a-y_val):+.4f}")

    # ── 6b. Medium / house group-bias correction (Stage B) ──────────────
    # Tiered (price-only) calibration above doesn't account for systematic
    # bias that varies by medium or house (see fit_group_bias_correction
    # docstring). Fit small additive corrections on the same val population,
    # applied sequentially: medium first, then house on the residual.
    medium_val = s_val["medium"].values
    house_val  = s_val["house"].values

    medium_corr = fit_group_bias_correction(y_val, p_cal_val_a, medium_val)
    p_cal_val_b = apply_group_bias_correction(medium_corr, p_cal_val_a, medium_val)

    house_corr  = fit_group_bias_correction(y_val, p_cal_val_b, house_val)
    p_cal_val   = apply_group_bias_correction(house_corr, p_cal_val_b, house_val)

    _med_shown = sorted(((g, v) for g, v in medium_corr.items() if g != "_default"),
                         key=lambda x: -abs(x[1]))
    print(f"  Medium-bias correction ({len(_med_shown)} groups): " +
          ", ".join(f"{g}={v:+.3f}" for g, v in _med_shown[:6]))
    _house_shown = sorted(((g, v) for g, v in house_corr.items() if g != "_default"),
                           key=lambda x: -abs(x[1]))
    print(f"  House-bias correction ({len(_house_shown)} groups):  " +
          ", ".join(f"{g}={v:+.3f}" for g, v in _house_shown))
    print(f"  bias  after tier-cal={np.mean(p_cal_val_a-y_val):+.4f}  "
          f"after medium-corr={np.mean(p_cal_val_b-y_val):+.4f}  "
          f"after house-corr={np.mean(p_cal_val-y_val):+.4f}")

    # ── 6c. Stage-C estimate-anchored shrink (fit on val) ───────────────
    # Blend the calibrated prediction toward the estimate so the model is
    # provably no worse than the estimate baseline on validation (λ=0 is in-grid). Fit BEFORE
    # conformal so interval residuals center on the shrunk prediction.
    estimate_shrink = fit_estimate_shrink(y_val, p_cal_val, est_val)
    med_est_v   = _dollar_medape(np.zeros_like(y_val), y_val, est_val)
    med_full_v  = _dollar_medape(p_cal_val, y_val, est_val)
    p_cal_val   = apply_estimate_shrink(estimate_shrink, p_cal_val, est_val)
    med_shrunk_v = _dollar_medape(p_cal_val, y_val, est_val)
    print("\n── Step 6c: Estimate-anchored shrink (Stage C, fit on val) ───")
    _tier_lam_str = ", ".join(f"{t}={l:.2f}" for t, l in estimate_shrink.items()
                              if t != "global")
    print(f"  λ global={estimate_shrink['global']:.3f}" +
          (f"   per-tier: {_tier_lam_str}" if _tier_lam_str else "   (no per-tier override)"))
    print(f"  val MedAPE → estimate-only={med_est_v:.1f}%  "
          f"full-model={med_full_v:.1f}%  shrunk={med_shrunk_v:.1f}%  "
          f"(shrunk ≤ estimate by construction: "
          f"{'✓' if med_shrunk_v <= med_est_v + 1e-6 else '✗ CHECK'})")

    print("")
    (conformal_lo, conformal_hi,
     tier_conformal,
     cc_lo_std, cc_hi_std) = fit_cross_conformal(
        ensemble, X_val, y_val, est_val, art_val, art_med_val, lo_val,
        groups_medium=medium_val, groups_house=house_val,
        estimate_shrink=estimate_shrink, K=10
    )

    # ── 7. Evaluation ──────────────────────────────────────────────────────
    print("\n── Step 7: Evaluation ────────────────────────────────────────")

    def _apply_full_calibration(p_ens, est, s_df, apply_shrink=True):
        """Stage A (tiered isotonic) + Stage B (medium then house bias corr)
        + Stage C (estimate-anchored shrink). Set apply_shrink=False to get the
        pre-shrink calibrated log-ratio — the model's *unshrunk* residual belief,
        used for ranking-skill analysis (the shrink optimises MedAPE but compresses
        ranking signal, zeroing it entirely in λ=0 tiers)."""
        p = apply_tiered_calibration(calibrators, p_ens, est)
        p = apply_group_bias_correction(medium_corr, p, s_df["medium"].values)
        p = apply_group_bias_correction(house_corr,  p, s_df["house"].values)
        if apply_shrink:
            p = apply_estimate_shrink(estimate_shrink, p, est)
        return p

    print("  Train (2018–2024, excl. <$10k + outlier trim):")
    p_ens_tr = ensemble(X_tr, art_tr, art_med_tr, lo_tr)
    p_cal_tr = _apply_full_calibration(p_ens_tr, est_tr, s_tr)
    eval_metrics(y_tr, p_ens_tr, est_arr=est_tr, label="raw train")
    train_m = eval_metrics(y_tr, p_cal_tr, est_arr=est_tr, label="calibrated train")

    # Primary metrics: ≥$10k lots only (apples-to-apples with train which also excludes <$10k).
    hi_val = est_val >= LOW_VAL
    hi_te  = est_te  >= LOW_VAL

    print("  Val (2025 H1, ≥$10k — primary):")
    eval_metrics(y_val, p_ens_val, est_arr=est_val, label="raw val")
    val_m_all = eval_metrics(y_val, p_cal_val, est_arr=est_val, label="calibrated val (all)", verbose=False)
    val_m = eval_metrics(y_val[hi_val], p_cal_val[hi_val], est_arr=est_val[hi_val],
                         label="calibrated val ≥$10k")
    print(f"  (lo-val <$10k footnote: N={(~hi_val).sum():,}  "
          f"MedAPE={val_m_all['medape']:.1f}%→excluded from headline)")

    p_ens_te = ensemble(X_te, art_te, art_med_te, lo_te)
    p_cal_te = _apply_full_calibration(p_ens_te, est_te, s_te)
    print("  Test (2025 H2, ≥$10k — primary):")
    eval_metrics(y_te, p_ens_te, est_arr=est_te, label="raw test")
    test_m_all = eval_metrics(y_te, p_cal_te, est_arr=est_te, label="calibrated test (all)", verbose=False)
    test_m = eval_metrics(y_te[hi_te], p_cal_te[hi_te], est_arr=est_te[hi_te],
                          label="calibrated test ≥$10k")
    print(f"  (lo-val <$10k footnote: N={(~hi_te).sum():,}  "
          f"MedAPE={test_m_all['medape']:.1f}%→excluded from headline)")

    print(f"\n  Overfitting gaps (≥$10k) → "
          f"train-val: {train_m['medape']:.1f}% → {val_m['medape']:.1f}%  "
          f"(+{val_m['medape']-train_m['medape']:.1f}pp)  |  "
          f"val-test: {val_m['medape']:.1f}% → {test_m['medape']:.1f}%  "
          f"({test_m['medape']-val_m['medape']:+.1f}pp)")

    # Conformal coverage on test set (residuals in log_ratio space)
    te_resids = y_te - p_cal_te
    te_coverage = float(((te_resids >= conformal_lo) & (te_resids <= conformal_hi)).mean() * 100)
    print(f"\n  Test conformal coverage: P10-P90 band covers {te_coverage:.1f}% of test lots "
          f"(target: 80%)")

    # ── 8. Price-bucket breakdown ──────────────────────────────────────────
    print("\n── Step 8: Price-bucket breakdown (calibrated test) ──────────")
    h_true = np.exp(y_te) * est_te          # hammer = exp(log_ratio) * estimate_mid
    h_pred = np.exp(np.clip(p_cal_te, -5, 5)) * est_te
    ape    = np.abs(h_true - h_pred) / h_true.clip(1)
    bins   = [0, 10e3, 50e3, 200e3, 1e6, 1e10]
    lbls   = ["<$10k", "$10-50k", "$50-200k", "$200k-1M", ">$1M"]
    bkt    = pd.cut(h_true, bins=bins, labels=lbls)
    print(f"  {'Bucket':<14}  {'N':>6}  {'MedAPE':>8}  {'within20%':>10}")
    # Primary tiers first (≥$10k)
    for lbl in ["$10-50k", "$50-200k", "$200k-1M", ">$1M"]:
        m = bkt == lbl
        if not m.any(): continue
        a = ape[m]
        print(f"  {lbl:<14}  {m.sum():>6,}  {np.median(a)*100:>7.1f}%  "
              f"{(a<.2).mean()*100:>9.1f}%")
    # Low-value specialist footnote
    print(f"  {'─'*50}")
    m_lo = bkt == "<$10k"
    if m_lo.any():
        a_lo = ape[m_lo]
        print(f"  {'[lo] <$10k':<14}  {m_lo.sum():>6,}  {np.median(a_lo)*100:>7.1f}%  "
              f"{(a_lo<.2).mean()*100:>9.1f}%  ← lo-value specialist")

    # ── 8b. Residual breakdown by medium / house (calibrated test, ≥$10k) ──
    # Verifies the Stage-B group-bias correction actually flattened the
    # systematic medium/house bias found in the 2026-06-10 residual analysis
    # (resid = pred - actual; large |resid| ⇒ remaining structure).
    print("\n── Step 8b: Residual breakdown — medium / house (calibrated test ≥$10k) ──")
    resid_te = (p_cal_te.astype(np.float64) - y_te.astype(np.float64))[hi_te]
    med_te   = s_te["medium"].values[hi_te]
    house_te_arr = s_te["house"].values[hi_te]
    print(f"  {'Medium':<22} {'N':>6}  {'mean resid':>11}  {'median resid':>13}")
    for m in sorted(pd.unique(med_te), key=lambda g: -(med_te == g).sum()):
        mask = med_te == m
        if mask.sum() < 30:
            continue
        print(f"  {m:<22} {mask.sum():>6,}  {resid_te[mask].mean():>+10.3f}   {np.median(resid_te[mask]):>+12.3f}")
    print(f"  {'House':<22} {'N':>6}  {'mean resid':>11}  {'median resid':>13}")
    for h in sorted(pd.unique(house_te_arr)):
        mask = house_te_arr == h
        if mask.sum() < 10:
            continue
        print(f"  {h:<22} {mask.sum():>6,}  {resid_te[mask].mean():>+10.3f}   {np.median(resid_te[mask]):>+12.3f}")

    # ── 9. Parallel temporal CV (3 folds) ─────────────────────────────────
    print("\n── Step 9: Temporal CV — 3 folds (parallel) ──────────────────")
    t9 = time.time()
    sold = df[df["sold"]==1].dropna(subset=["log_ratio","log_estimate_mid"]).copy()
    fold_pairs = [(2021, 2022), (2022, 2023), (2023, 2024)]

    def run_fold(tr_yr, val_yr):
        f_tr  = sold[sold["sale_year"] <= tr_yr]
        f_val = sold[sold["sale_year"] == val_yr]
        Xf_tr  = f_tr[FEATURE_COLS].fillna(0).values.astype(np.float32)
        yf_tr  = f_tr["log_ratio"].values.astype(np.float32)
        Xf_val = f_val[FEATURE_COLS].fillna(0).values.astype(np.float32)
        yf_val = f_val["log_ratio"].values.astype(np.float32)
        ef_val = f_val["estimate_mid"].values
        wf     = time_weights(f_tr["sale_date"])
        m = _fit_lgbm(Xf_tr, yf_tr, Xf_val, yf_val, wf,
                      f"CV fold (val={val_yr})", N_EACH//3 or 1)
        pf = m.predict(Xf_val)
        # Report CV MedAPE on ≥$10k only (consistent with primary eval metric)
        hi = ef_val >= LOW_VAL
        h_true_f = np.exp(yf_val[hi]) * ef_val[hi]
        h_pred_f = np.exp(np.clip(pf[hi], -5, 5)) * ef_val[hi]
        medape = np.median(np.abs(h_true_f - h_pred_f) / h_true_f.clip(1)) * 100
        return val_yr, medape

    cv_results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(run_fold, tr, vl) for tr, vl in fold_pairs]
        for fut in as_completed(futs):
            yr, medape = fut.result()
            cv_results[yr] = medape
            print(f"  Fold val={yr}  MedAPE={medape:.1f}%")

    cv_medapes = list(cv_results.values())
    print(f"  CV MedAPE: {np.mean(cv_medapes):.1f}% ± {np.std(cv_medapes):.1f}%"
          f"  ({time.time()-t9:.1f}s parallel)")

    # ── 10. Feature importance ─────────────────────────────────────────────
    print("\n── Step 10: Feature importance (LightGBM gain) ───────────────")
    imp = lgbm_model.feature_importance(importance_type="gain")
    top = sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1])
    mx  = top[0][1] or 1
    for name, score in top[:15]:
        print(f"  {name:<35} {'█'*int(score/mx*35)}")

    # ── 10b. Ablation study ───────────────────────────────────────────────
    # Pass first 41 features (FEATURE_COLS) only — ablation trains fresh LGBM
    # models and doesn't need sell_through_prob (which would require a separate
    # sell-through inference pass per ablation run).
    _nfc = len(FEATURE_COLS)
    run_ablation(X_tr[:, :_nfc], y_tr, X_val[:, :_nfc], y_val, w_tr, val_m["medape"])

    # ── 11. Save ──────────────────────────────────────────────────────────
    print("\n── Step 11: Save ─────────────────────────────────────────────")
    lgbm_model.save_model(os.path.join(models_dir, "lgbm.txt"))
    lgbm_q10.save_model(  os.path.join(models_dir, "lgbm_q10.txt"))
    lgbm_q90.save_model(  os.path.join(models_dir, "lgbm_q90.txt"))
    cat_model.save_model( os.path.join(models_dir, "catboost.cbm"))
    if lo_model:
        lo_model.save_model(os.path.join(models_dir, "lgbm_low.txt"))
    # Bundle Stage-B group corrections into the same dict under keys that
    # can't collide with tier names ("global", "lt_10k", ...) so this stays
    # backward-compatible: old predict.py just ignores the extra keys, and
    # `isinstance(calibrator, dict)` is unchanged.
    calibrators["_medium_corr"] = medium_corr
    calibrators["_house_corr"]  = house_corr
    calibrators["_estimate_shrink"] = estimate_shrink
    with open(os.path.join(models_dir, "calibrator.pkl"),   "wb") as f: pickle.dump(calibrators, f)
    with open(os.path.join(models_dir, "meta_learner.pkl"), "wb") as f: pickle.dump(meta_learner, f)
    with open(os.path.join(models_dir, "sell_through.pkl"), "wb") as f: pickle.dump(sell_through_model, f)

    # artist_stats — add last_sale_date and rolling sell_thru_12m
    sold_for_stats = df[df["sold"]==1].dropna(subset=["log_ratio","log_estimate_mid"])
    artist_stats = (sold_for_stats.groupby("artist")
                    .agg(n_sales=("sold_price","count"),
                         median_ratio=("log_ratio","median"),
                         median_log_hammer=("log_hammer","median"),
                         sell_through_rate=("sold","mean"),
                         log_price_std=("log_hammer","std"))
                    .reset_index())
    # Last sale date per artist
    last_dates = (sold_for_stats.sort_values("sale_date")
                  .groupby("artist")["sale_date"].last()
                  .rename("last_sale_date"))
    artist_stats = artist_stats.join(last_dates, on="artist", how="left")
    # Last known 12m sell-through rate
    if "artist_sell_thru_12m" in df.columns:
        st12m = (df.sort_values("sale_date")
                   .groupby("artist")["artist_sell_thru_12m"].last()
                   .rename("sell_thru_12m"))
        artist_stats = artist_stats.join(st12m, on="artist", how="left")
        global_st = float(len(sold_for_stats) / max(len(df), 1))
        artist_stats["sell_thru_12m"] = artist_stats["sell_thru_12m"].fillna(global_st)
    # Last known momentum per artist — so inference uses the most recent trailing
    # momentum (as of the training cutoff) instead of a hardcoded 0.0, matching how
    # the feature was computed at train time (train/inference parity).
    _df_sorted = df.sort_values("sale_date")
    for _mcol in ("artist_momentum_6m", "artist_momentum_12m", "artist_momentum_accel"):
        if _mcol in df.columns:
            _last = _df_sorted.groupby("artist")[_mcol].last().rename(_mcol)
            artist_stats = artist_stats.join(_last, on="artist", how="left")
            artist_stats[_mcol] = artist_stats[_mcol].fillna(0.0)
    artist_stats.to_parquet(os.path.join(models_dir, "artist_stats.parquet"), index=False)

    # Global latest market-regime values (as of the most recent training month) —
    # saved so inference defaults match training instead of a flat g_med_hammer.
    market_latest = {}
    for _kcol in ("market_index_3m", "market_index_12m"):
        if _kcol in df.columns:
            _v = _df_sorted[_kcol].dropna()
            if len(_v):
                market_latest[_kcol] = float(_v.iloc[-1])

    # Save artist×house ratio stats for inference-time lookup of house-specific track records
    sold_for_ah = df[df["sold"]==1].dropna(subset=["log_ratio"])
    if "house_cat" in df.columns:
        ah_stats = (sold_for_ah.groupby(["artist", "house_cat"])["log_ratio"]
                    .median()
                    .reset_index()
                    .rename(columns={"log_ratio": "median_ratio"}))
        ah_stats["house_cat"] = ah_stats["house_cat"].astype(int)
        ah_stats.to_parquet(os.path.join(models_dir, "artist_house_stats.parquet"), index=False)

    # Save eval predictions (log_ratio → hammer conversion)
    raw_q10_val = lgbm_q10.predict(X_val)
    raw_q90_val = lgbm_q90.predict(X_val)
    raw_q10_te  = lgbm_q10.predict(X_te)
    raw_q90_te  = lgbm_q90.predict(X_te)

    q10_val = np.exp(raw_q10_val) * est_val
    q90_val = np.exp(raw_q90_val) * est_val
    q10_te  = np.exp(raw_q10_te)  * est_te
    q90_te  = np.exp(raw_q90_te)  * est_te

    # CQR-corrected quantile intervals (adaptive per lot)
    cqr_q10_pred_val = np.exp(raw_q10_val - cqr_lo_correction) * est_val
    cqr_q90_pred_val = np.exp(raw_q90_val + cqr_hi_correction) * est_val
    cqr_q10_pred_te  = np.exp(raw_q10_te  - cqr_lo_correction) * est_te
    cqr_q90_pred_te  = np.exp(raw_q90_te  + cqr_hi_correction) * est_te

    conf_p10_val = np.exp(np.clip(p_cal_val + conformal_lo, -5, 5)) * est_val
    conf_p90_val = np.exp(np.clip(p_cal_val + conformal_hi, -5, 5)) * est_val
    conf_p10_te  = np.exp(np.clip(p_cal_te  + conformal_lo, -5, 5)) * est_te
    conf_p90_te  = np.exp(np.clip(p_cal_te  + conformal_hi, -5, 5)) * est_te

    # Pre-shrink calibrated log-ratio — the model's unshrunk residual belief
    # (for ranking-skill analysis; the Stage-C shrink compresses ranking signal).
    p_raw_val = _apply_full_calibration(p_ens_val, est_val, s_val, apply_shrink=False)
    p_raw_te  = _apply_full_calibration(p_ens_te,  est_te,  s_te,  apply_shrink=False)

    # Carry sale_month + a split tag so downstream consumers can separate the
    # calibration/shrink-fitting VALIDATION set (2025 H1) from the clean held-out
    # TEST set (2025 H2). The headline must be test-only — val is in-sample for
    # Stage-C, calibration, and CQR.
    _evcols = ["sale_year", "sale_month", "sold_price", "estimate_mid",
               "artist", "medium", "house", "auction_id"]   # auction_id → sale-level clustering
    _evcols = [c for c in _evcols if c in s_val.columns]
    eval_val = s_val[_evcols].copy()
    eval_val["split"] = "val"
    eval_val["predicted_hammer"] = np.exp(np.clip(p_cal_val, -5, 5)) * est_val
    eval_val["pred_logratio_raw"] = p_raw_val
    eval_val["p10"] = q10_val
    eval_val["p90"] = q90_val
    eval_val["cqr_p10"] = cqr_q10_pred_val
    eval_val["cqr_p90"] = cqr_q90_pred_val
    eval_val["conformal_p10"] = conf_p10_val
    eval_val["conformal_p90"] = conf_p90_val

    _evcols_te = [c for c in _evcols if c in s_te.columns]
    eval_te = s_te[_evcols_te].copy()
    eval_te["split"] = "test"
    eval_te["predicted_hammer"] = np.exp(np.clip(p_cal_te, -5, 5)) * est_te
    eval_te["pred_logratio_raw"] = p_raw_te
    eval_te["p10"] = q10_te
    eval_te["p90"] = q90_te
    eval_te["cqr_p10"] = cqr_q10_pred_te
    eval_te["cqr_p90"] = cqr_q90_pred_te
    eval_te["conformal_p10"] = conf_p10_te
    eval_te["conformal_p90"] = conf_p90_te

    # Walk-forward fold: return this fold's TEST predictions (+ headline metrics) and
    # skip all artifact/meta writing. The caller pools folds for aggregate reporting.
    if return_fold_preds:
        if force_recompute and os.path.exists(cache):
            os.remove(cache)
        return {
            "test_preds": eval_te.reset_index(drop=True),
            "val_preds":  eval_val.reset_index(drop=True),
            "test_metrics_all": test_m_all,
            "nnls_weights": {"lgbm": float(w_l), "cat": float(w_c)},
            "estimate_shrink": estimate_shrink,
        }

    # In deploy mode test_df aliases val_df, so eval_te duplicates eval_val — save once.
    eval_preds = eval_val.copy() if deploy else pd.concat([eval_val, eval_te], ignore_index=True)
    _eval_path = os.path.join(models_dir, "eval_predictions.parquet")
    # In deploy mode eval_preds is the in-sample cal tail. If a prior eval run left
    # honest held-out predictions, keep those for the app's benchmark/trends instead
    # of overwriting them with in-sample numbers.
    if deploy and prior_display is not None and os.path.exists(_eval_path):
        print(f"  Kept prior eval_predictions.parquet (honest held-out) — not overwriting with in-sample tail")
    else:
        eval_preds.to_parquet(_eval_path, index=False)
        print(f"  Saved eval_predictions.parquet  ({len(eval_preds):,} rows)")

    # ── Display metrics — single source of truth for the app + docs ─────────
    # Computed from eval_preds so the UI never hardcodes (and never drifts).
    # CRITICAL: the headline is computed on the TEST split only (2025 H2). The
    # validation split (2025 H1) is in-sample for Stage-C, tiered calibration, and
    # CQR, so including it would inflate the headline. In deploy mode eval_preds is
    # the in-sample cal tail, so the honest display block is carried from the eval run.
    _hd_src = eval_preds[eval_preds["split"] == "test"] if "split" in eval_preds.columns else eval_preds
    if len(_hd_src) == 0:
        _hd_src = eval_preds
    _y  = _hd_src["sold_price"].values
    _mp = _hd_src["predicted_hammer"].values
    _e  = _hd_src["estimate_mid"].values
    _ape_m = np.abs(_mp - _y) / np.clip(_y, 1, None)
    _ape_e = np.abs(_e  - _y) / np.clip(_y, 1, None)
    _cov   = float(((_y >= _hd_src["conformal_p10"]) &
                    (_y <= _hd_src["conformal_p90"])).mean() * 100)
    # Ranking AUC on the unshrunk signal: does it sort lots that beat the midpoint?
    try:
        from sklearn.metrics import roc_auc_score as _auc
        _beat = (_y > _e).astype(int)
        _score = _hd_src["pred_logratio_raw"].values
        _rank_auc = round(float(_auc(_beat, _score)), 3) if _beat.min() != _beat.max() else None
    except Exception:
        _rank_auc = None
    display_metrics = dict(
        medape_model=round(float(np.median(_ape_m) * 100), 1),
        medape_est=round(float(np.median(_ape_e) * 100), 1),
        within20_model=round(float((_ape_m <= 0.20).mean() * 100), 1),
        within20_est=round(float((_ape_e <= 0.20).mean() * 100), 1),
        r2_raw_model=round(float(r2_score(np.log1p(_y), np.log1p(_mp))), 3),
        r2_raw_est=round(float(r2_score(np.log1p(_y), np.log1p(_e))), 3),
        interval_coverage=round(_cov, 1),
        ranking_auc=_rank_auc,                # test-only AUC, beats-vs-misses midpoint
        n_holdout=int(len(_hd_src)),          # test-only (2025 H2) lots in the headline
        holdout_period="2025 H2 (test)",
        n_priced=int((df["sold"] == 1).sum()),
        n_total=int(len(df)),
        houses=sorted(df["house"].dropna().str.lower().unique().tolist()),
    )

    from datetime import datetime
    meta = dict(
        w_lgbm=w_l, w_cat=w_c, display=display_metrics,
        feature_cols=FEATURE_COLS, cat_feat_names=cat_names,
        target="log_ratio", low_value_threshold=LOW_VAL,
        has_low_model=lo_model is not None,
        has_meta_learner=True,
        has_sell_through=True,
        has_sell_through_feature=True,   # sell_through_prob appended as feature #42
        has_artist_medium_cat=True,
        has_tiered_calibrator=True,
        has_cqr=True,
        cqr_lo_correction=round(cqr_lo_correction, 6),
        cqr_hi_correction=round(cqr_hi_correction, 6),
        cqr_val_coverage=round(cqr_cov_val, 2),
        cross_conformal_K=10,
        cross_conformal_stability={
            "global_lo_std": round(cc_lo_std, 4),
            "global_hi_std": round(cc_hi_std, 4),
            "tier_lo_std": {t: round(tier_conformal[t]["lo_std"], 4)
                            for t in tier_conformal if "lo_std" in tier_conformal[t]},
            "tier_hi_std": {t: round(tier_conformal[t]["hi_std"], 4)
                            for t in tier_conformal if "hi_std" in tier_conformal[t]},
        },
        trained_at=datetime.now().strftime("%Y-%m-%d"),
        conformal_lo=conformal_lo,
        conformal_hi=conformal_hi,
        conformal_tiers=tier_conformal,
        conformal_test_coverage=round(te_coverage, 2),
        train_rows=int(len(X_tr)),
        cv_medape_mean=float(np.mean(cv_medapes)),
        cv_medape_std=float(np.std(cv_medapes)),
        train_metrics={k: round(float(v),4) for k,v in train_m.items()},
        val_metrics  ={k: round(float(v),4) for k,v in val_m.items()},
        test_metrics ={k: round(float(v),4) for k,v in test_m.items()},
        market_latest=market_latest,   # latest trailing market index → inference default
    )
    if deploy:
        # The served artifact: no honest held-out test of its own (it trained on
        # everything but the calibration tail). Suppress the in-sample tail
        # "test" numbers and carry forward the prior EVAL run's held-out estimate.
        meta["deploy"]          = True
        meta["trained_through"] = str(max_date.date())
        meta["deploy_cal_cutoff"] = str(cutoff.date())
        meta["deploy_cal_months"] = deploy_cal_months
        meta["test_metrics"]    = None          # in-sample tail — not an honest test
        meta["holdout_eval_metrics"] = prior_holdout   # honest forward estimate (or None)
        # Show the honest held-out display block (from the eval run) in the served
        # app, not the in-sample cal-tail numbers. Keep the in-sample ones visible
        # under a separate key for transparency.
        if prior_display:
            meta["deploy_insample_display"] = display_metrics
            # Preserve current dataset coverage counts (full data through deploy cutoff)
            _carried = dict(prior_display)
            _carried["n_priced"] = display_metrics["n_priced"]
            _carried["n_total"]  = display_metrics["n_total"]
            _carried["houses"]   = display_metrics["houses"]
            meta["display"] = _carried
    else:
        meta["deploy"] = False
    with open(os.path.join(models_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Eval manifest — publication-grade provenance so every reported number is
    # traceable to one frozen artifact (full hashes, commit + dirty state, dataset
    # hash, environment).
    import hashlib, subprocess, platform, sys, re
    def _sha(p):
        try:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()                       # full 64-char SHA-256
        except Exception:
            return None
    def _git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  cwd=os.path.dirname(__file__)).stdout.strip() or None
        except Exception:
            return None
    def _pkg_ver(name):
        try:
            import importlib.metadata as _im
            return _im.version(name)
        except Exception:
            return None
    _commit = _git("rev-parse", "--verify", "HEAD")
    if _commit is not None and not re.fullmatch(r"[0-9a-f]{40}", _commit):
        _commit = None                                 # unborn repo (no commits yet)
    _dirty = bool(_git("status", "--porcelain"))       # uncommitted changes present?
    # Reference the eval manifest these carried metrics came from (deploy mode)
    _eval_manifest_ref = None
    if deploy:
        try:
            _prev_man = os.path.join(models_dir, "eval_manifest.json")
            if os.path.exists(_prev_man):
                _eval_manifest_ref = json.load(open(_prev_man)).get("artifact_sha256", {}).get("meta.json")
        except Exception:
            pass
    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "code_commit": _commit,                        # None until the repo is committed
        "working_tree_dirty": _dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "key_packages": {p: _pkg_ver(p) for p in ("lightgbm", "catboost", "scikit-learn",
                                                   "numpy", "pandas", "scipy")},
        "data_path": os.path.basename(data_path),
        "data_sha256": _sha(data_path),
        "feature_cache_sha256": _sha(cache),
        "n_features": len(FEATURE_COLS) + (1 if meta.get("has_sell_through_feature") else 0),
        "feature_cols_count": len(FEATURE_COLS),
        "nnls_weights": {"lgbm": round(float(w_l), 4), "cat": round(float(w_c), 4)},
        "split": {"train": "≤2024", "val": "2025 H1", "test": "2025 H2"},
        "headline_split": "test (2025 H2)" if not deploy else "carried from eval",
        "carried_from_eval_meta_sha256": _eval_manifest_ref,   # deploy: which eval run the display came from
        "score_column_for_ranking": "pred_logratio_raw (unshrunk)",
        "display": meta.get("display"),
        "artifact_sha256": {
            f: _sha(os.path.join(models_dir, f)) for f in
            ("lgbm.txt", "lgbm_low.txt", "lgbm_q10.txt", "lgbm_q90.txt", "catboost.cbm",
             "calibrator.pkl", "meta_learner.pkl", "sell_through.pkl",
             "eval_predictions.parquet", "artist_stats.parquet", "artist_house_stats.parquet",
             "meta.json")
        },
    }
    with open(os.path.join(models_dir, "eval_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote eval_manifest.json (commit={_commit}, dirty={_dirty}, headline={manifest['headline_split']})")

    # Only delete features cache when --recompute was explicitly passed.
    # Normal runs keep the cache to speed up re-training.
    if force_recompute and os.path.exists(cache):
        os.remove(cache)
        print("  Deleted features_cache.parquet (--recompute flag set)")

    elapsed = time.time() - T0
    print(f"\n✓ Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if deploy:
        print(f"  DEPLOY artifact saved → trained through {max_date.date()} "
              f"(GBM ≤ {cutoff.date()}, calibrated on {deploy_cal_months}-month tail).")
        print("  ⚠ 'test' numbers above are IN-SAMPLE (cal tail) — NOT a held-out test.")
        if prior_holdout:
            print(f"  Honest forward estimate (carried from eval run): "
                  f"MedAPE={prior_holdout.get('medape')}%  "
                  f"within20%={prior_holdout.get('within_20')}%  "
                  f"R²={prior_holdout.get('r2')}")
        else:
            print("  ⚠ No held-out estimate carried — run eval mode first.")
        print(f"  CV MedAPE ≥$10k: {np.mean(cv_medapes):.1f}% ± {np.std(cv_medapes):.1f}%")
    else:
        print(f"  Calibrated test ≥$10k →  MedAPE={test_m['medape']:.1f}%  "
              f"R²={test_m['r2']:.4f}  within20%={test_m['within_20']:.1f}%")
        print(f"  (all lots incl. <$10k: MedAPE={test_m_all['medape']:.1f}%)")
        print(f"  CV MedAPE ≥$10k: {np.mean(cv_medapes):.1f}% ± {np.std(cv_medapes):.1f}%")
        print(f"  Conformal coverage (test): {te_coverage:.1f}%  "
              f"[P10 offset={conformal_lo:+.3f}  P90 offset={conformal_hi:+.3f}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="data/art_auction_clean.csv")
    ap.add_argument("--models",    default="models")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--deploy", action="store_true",
                    help="Refit-all DEPLOY artifact: train through the most recent data "
                         "(minus a thin calibration tail) for serving. Run eval mode first "
                         "so the honest held-out estimate is carried into meta.json.")
    ap.add_argument("--deploy-cal-months", type=int, default=3, dest="deploy_cal_months",
                    help="Months of recent data held out (from the GBM) to fit "
                         "calibration + conformal in --deploy mode (default 3).")
    args = ap.parse_args()
    train(args.data, args.models, args.recompute,
          deploy=args.deploy, deploy_cal_months=args.deploy_cal_months)
