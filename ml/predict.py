# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Inference — leakage-controlled deployed ensemble with calibration + quantile intervals.

Two-model ensemble (LightGBM + CatBoost) combined by the NNLS meta-learner
(meta_learner.pkl), with tiered isotonic calibration, group-bias and
estimate-anchored shrink corrections, plus a sell-through model and
split-conformal prediction intervals from meta.json.
Falls back gracefully when any optional component is missing.
"""
import json, math, os, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

from ml.features import MEDIUM_CATEGORIES, HOUSE_MAP, FEATURE_COLS, SMOOTH_K

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class ArtPricePredictor:
    def __init__(self, models_dir=None):
        self.md = models_dir or MODELS_DIR
        self._load()

    def _load(self):
        with open(os.path.join(self.md, "meta.json")) as f:
            self.meta = json.load(f)

        self.w_l = self.meta["w_lgbm"]
        self.w_c = self.meta["w_cat"]
        self.low_thr = self.meta.get("low_value_threshold", 10_000)

        # Conformal offsets (guaranteed coverage intervals)
        self.conformal_lo = self.meta.get("conformal_lo", None)
        self.conformal_hi = self.meta.get("conformal_hi", None)
        # CQR corrections (adaptive per-lot quantile intervals)
        self.cqr_lo_correction = self.meta.get("cqr_lo_correction", None)
        self.cqr_hi_correction = self.meta.get("cqr_hi_correction", None)

        # Use feature columns from training run (backward compatible with old models)
        self.feature_cols = self.meta.get("feature_cols", FEATURE_COLS)
        # Target space: "log_ratio" (new) or "log_hammer" (old models)
        self.use_log_ratio = (self.meta.get("target", "log_hammer") == "log_ratio")
        # Whether CatBoost was trained with artist_medium as 2nd categorical
        self.has_art_med = self.meta.get("has_artist_medium_cat", False)

        self.lgbm     = lgb.Booster(model_file=os.path.join(self.md, "lgbm.txt"))
        self.lgbm_q10 = lgb.Booster(model_file=os.path.join(self.md, "lgbm_q10.txt"))
        self.lgbm_q90 = lgb.Booster(model_file=os.path.join(self.md, "lgbm_q90.txt"))
        self.lgbm_low = None
        if self.meta.get("has_low_model") and os.path.exists(os.path.join(self.md, "lgbm_low.txt")):
            self.lgbm_low = lgb.Booster(model_file=os.path.join(self.md, "lgbm_low.txt"))

        with open(os.path.join(self.md, "calibrator.pkl"), "rb") as f:
            self.calibrator = pickle.load(f)
        # Tiered calibrator is a dict (new models); single IsotonicRegression = legacy
        self.has_tiered_cal = isinstance(self.calibrator, dict)

        self.cat = CatBoostRegressor()
        self.cat.load_model(os.path.join(self.md, "catboost.cbm"))

        # NNLS meta-learner (preferred) — fallback to weighted average
        self.meta_learner = None
        meta_learner_path = os.path.join(self.md, "meta_learner.pkl")
        if (self.meta.get("has_meta_learner", False)
                and os.path.exists(meta_learner_path)):
            with open(meta_learner_path, "rb") as f:
                self.meta_learner = pickle.load(f)

        # Sell-through probability model (optional)
        self.sell_through_model = None
        st_path = os.path.join(self.md, "sell_through.pkl")
        if self.meta.get("has_sell_through") and os.path.exists(st_path):
            with open(st_path, "rb") as f:
                self.sell_through_model = pickle.load(f)

        self.artist_stats = pd.read_parquet(
            os.path.join(self.md, "artist_stats.parquet")).set_index("artist")
        self.g_med_ratio  = float(self.artist_stats["median_ratio"].median())
        self.g_med_hammer = float(self.artist_stats["median_log_hammer"].median())
        self.g_sell_thru  = float(self.artist_stats["sell_through_rate"].median())
        self.g_log_std    = float(self.artist_stats["log_price_std"].median())

        # Artist×house ratio lookup: (artist, house_cat_int) → median log_ratio
        # Falls back to artist_smoothed_ratio for unknown pairs.
        self._artist_house_stats: dict = {}
        ah_path = os.path.join(self.md, "artist_house_stats.parquet")
        if os.path.exists(ah_path):
            try:
                ah = pd.read_parquet(ah_path)
                for _, r in ah.iterrows():
                    self._artist_house_stats[(str(r["artist"]), int(r["house_cat"]))] = float(r["median_ratio"])
            except Exception:
                pass

        # Latest trailing market-regime index (as of the training cutoff) — used as
        # the inference default for market_index_3m/12m so deployment matches training
        # instead of a flat g_med_hammer. Falls back to g_med_hammer if absent.
        _ml = (self.meta.get("market_latest") or {}) if isinstance(self.meta, dict) else {}
        self._market_latest = {
            "market_index_3m":  float(_ml.get("market_index_3m",  self.g_med_hammer)),
            "market_index_12m": float(_ml.get("market_index_12m", self.g_med_hammer)),
        }

        # Latest S&P 500 macro values — used as inference-time defaults for sp500_* features
        self._sp500_latest = {"sp500_ret_3m": 0.02, "sp500_ret_12m": 0.10, "sp500_level": 0.50}
        sp500_path = os.path.join(self.md, "..", "data", "macro_sp500.parquet")
        if os.path.exists(sp500_path):
            try:
                sp = pd.read_parquet(sp500_path).dropna(subset=["sp500_ret_3m"])
                if len(sp):
                    row = sp.iloc[-1]
                    self._sp500_latest = {
                        "sp500_ret_3m":  float(row["sp500_ret_3m"]),
                        "sp500_ret_12m": float(row["sp500_ret_12m"]),
                        "sp500_level":   float(row["sp500_level"]),
                    }
            except Exception:
                pass

    # ── artist lookup ──────────────────────────────────────────────────────
    def _artist_feats(self, artist: str, sale_date: "pd.Timestamp | None" = None) -> dict:
        import pandas as _pd
        if artist in self.artist_stats.index:
            row = self.artist_stats.loc[artist]
            n   = float(row["n_sales"])
            w   = n / (n + SMOOTH_K)
            log_std    = float(row.get("log_price_std") or self.g_log_std)
            med_hammer = float(row["median_log_hammer"])
            med_ratio  = float(row["median_ratio"])
            raw_prestige = math.log1p(n) * med_hammer
            sell_thru_12m = float(row.get("sell_thru_12m", self.g_sell_thru))
            # Days since last sale — compute from last_sale_date if available
            days_since = 365  # neutral fallback
            if sale_date is not None and "last_sale_date" in row.index:
                ld = row["last_sale_date"]
                if _pd.notna(ld):
                    try:
                        days_since = max(0, min(3650, (sale_date - _pd.Timestamp(ld)).days))
                    except Exception:
                        pass
            return dict(
                artist_n_past_sales      = n,
                artist_median_log_hammer = med_hammer,
                artist_median_ratio      = med_ratio,
                artist_sell_through_rate = float(row["sell_through_rate"]),
                artist_log_price_std     = log_std,
                artist_trend_12m_ratio   = med_ratio,
                artist_last_ratio        = med_ratio,
                artist_has_history       = int(n >= 5),
                artist_smoothed_ratio    = w * med_ratio + (1 - w) * self.g_med_ratio,
                artist_momentum_6m       = float(row.get("artist_momentum_6m") or 0.0),
                artist_momentum_12m      = float(row.get("artist_momentum_12m") or 0.0),
                artist_prestige_score    = raw_prestige,
                artist_sell_thru_12m     = sell_thru_12m,
                days_since_last_sale     = float(days_since),
                artist_momentum_accel    = float(row.get("artist_momentum_accel") or 0.0),
            ), True, log_std
        g_st12m = float(getattr(self, "g_sell_thru", 0.8))
        return dict(
            artist_n_past_sales=0, artist_median_log_hammer=self.g_med_hammer,
            artist_median_ratio=self.g_med_ratio, artist_sell_through_rate=self.g_sell_thru,
            artist_log_price_std=self.g_log_std, artist_trend_12m_ratio=self.g_med_ratio,
            artist_last_ratio=self.g_med_ratio, artist_has_history=0,
            artist_smoothed_ratio=self.g_med_ratio,
            artist_momentum_6m=0.0, artist_momentum_12m=0.0,
            artist_prestige_score=0.0,
            artist_sell_thru_12m=g_st12m,
            days_since_last_sale=float(365),
            artist_momentum_accel=0.0,
        ), False, self.g_log_std

    # ── feature vector ────────────────────────────────────────────────────
    def _fvec(self, **kw):
        import re as _re
        import pandas as _pd
        est_lo = float(kw.get("estimate_low") or 0)
        est_hi = float(kw.get("estimate_high") or 0)
        est_mid= (est_lo + est_hi) / 2
        med    = str(kw.get("medium","unknown")).lower().strip()
        med_c  = MEDIUM_CATEGORIES.get(med, 26)
        house  = HOUSE_MAP.get(str(kw.get("house","christies")).lower(), 0)
        w_cm   = float(kw.get("width_cm") or 0)
        h_cm   = float(kw.get("height_cm") or 0)
        # PARITY: training leaves aspect_ratio NaN for missing dims and prep() fills
        # NaN→0.0, so 0.0 is the model's learned "dims missing" marker. Emit the same
        # here (previously 1.0, which the model reads as a square artwork).
        asp    = max(0.1, min(10, w_cm / max(h_cm, 0.01))) if (w_cm and h_cm) else 0.0
        prov   = str(kw.get("provenance") or "")
        pl     = min(len(prov), 500)
        pac    = len(_re.findall(r"christie|sotheby|phillips|bonham|auction",prov,_re.I))
        # Mirror features.py: "private collection/collector" mention (regex, case-insensitive)
        ppc    = int(bool(_re.search(r"private collection|private collector", prov, _re.I)))
        sy     = int(kw.get("sale_year", 2025))
        sm     = int(kw.get("sale_month", 6))
        sq     = (sm-1)//3+1
        ym     = kw.get("year_made")
        age    = max(0, min(500, sy - int(ym))) if ym else 0
        title  = str(kw.get("title") or "")
        artist = str(kw.get("artist",""))
        lot_num = float(kw.get("lot_number") or 0)

        # Pass sale_date to artist lookup so days_since_last_sale can be computed
        try:
            sale_date = _pd.Timestamp(f"{sy}-{sm:02d}-01")
        except Exception:
            sale_date = None

        af, known, log_std = self._artist_feats(artist, sale_date=sale_date)

        # artist_medium key for CatBoost (returned but not in feat dict)
        art_med = f"{artist} | {med}"

        # Artist nationality lookup (same dict as features.py)
        from ml.features import ARTIST_NATIONALITY as _AN
        nat_region = int(_AN.get(artist, 0))

        feat = dict(
            log_estimate_mid    = np.log1p(est_mid),
            estimate_range_pct  = (est_hi-est_lo)/max(est_lo,1),
            **af,
            medium_cat          = med_c,
            log_width           = np.log1p(w_cm),
            log_height          = np.log1p(h_cm),
            has_dims            = int(bool(w_cm and h_cm)),
            aspect_ratio        = asp,
            provenance_len      = pl,
            prov_auction_count  = pac,
            prov_private_collection = ppc,
            artist_nationality_region = nat_region,
            # Evening/prestige sales vs day/online sales.
            # Pass is_evening_sale=1 for Christie's/Sotheby's/Phillips Evening Sales,
            # single-owner collections, and other prestige sales (≤80 lots, high estimate).
            is_evening_sale     = int(bool(kw.get("is_evening_sale", False))),
            is_online_sale      = int(bool(kw.get("is_online_sale",  False))),
            sale_year           = sy,
            sale_month          = sm,
            sale_quarter        = sq,
            artwork_age         = age,
            house_cat           = house,
            market_index_3m     = float(kw.get("market_index_3m",  self._market_latest["market_index_3m"])),
            market_index_12m    = float(kw.get("market_index_12m", self._market_latest["market_index_12m"])),
            market_momentum     = float(kw.get("market_momentum",  1.0)),
            title_is_untitled   = int("untitled" in title.lower()),
            estimate_in_millions= int(est_mid >= 1_000_000),
            log_lot_number      = np.log1p(lot_num),
            sale_month_sin      = float(np.sin(2 * np.pi * sm / 12)),
            sale_month_cos      = float(np.cos(2 * np.pi * sm / 12)),
            log_estimate_spread = float(np.log1p(max(0, est_hi - est_lo))),
            estimate_is_round   = int(est_lo > 0 and est_lo % 10_000 == 0),
            log_title_length    = float(np.log1p(len(title))),
            # artist_momentum_accel supplied via **af (persisted latest per-artist value)
            # Relative positioning
            lot_vs_artist_median = float(
                np.log1p(est_mid) - af.get("artist_median_log_hammer", np.log1p(est_mid))
            ),
            estimate_tightness  = float(np.clip(
                1.0 - (est_hi - est_lo) / max(est_mid, 1), -1.0, 1.0
            )),
            # Artist×house ratio — house-specific track record
            # Falls back to artist_smoothed_ratio when no house-specific history exists
            artist_house_ratio  = self._artist_house_stats.get(
                (artist, house),
                af.get("artist_smoothed_ratio", self.g_med_ratio),
            ),
            # Lot position within the sale (0=first, 1=last); 0.5 = neutral when unknown
            lot_position_pct    = float(kw.get("lot_position_pct", 0.5)),
            # S&P 500 macro features — use latest known values as inference defaults
            sp500_ret_3m        = float(kw.get("sp500_ret_3m",  self._sp500_latest["sp500_ret_3m"])),
            sp500_ret_12m       = float(kw.get("sp500_ret_12m", self._sp500_latest["sp500_ret_12m"])),
            sp500_level         = float(kw.get("sp500_level",   self._sp500_latest["sp500_level"])),
        )
        X = np.array([[feat[c] for c in self.feature_cols]], dtype=np.float32)
        return X, est_mid, known, log_std, art_med

    # ── base predictions ──────────────────────────────────────────────────
    def _base_preds(self, X, artist: str, art_med: str = "") -> np.ndarray:
        """Returns shape (1, 2) prediction matrix for meta-learner.
        Column order: [lgbm, cat] — matches 2-element NNLS weight vector.
        """
        p_l = self.lgbm.predict(X)[0]
        if self.has_art_med:
            Xc = np.hstack([np.array([[artist]]), np.array([[art_med]]), X.astype(object)])
        else:
            Xc = np.hstack([np.array([[artist]]), X.astype(object)])
        p_c = float(self.cat.predict(Xc)[0])
        return np.array([[p_l, p_c]], dtype=np.float32)

    # ── calibration ───────────────────────────────────────────────────────
    def _apply_cal(self, p_raw: float, est_mid: float,
                    medium: str = "", house: str = "") -> float:
        """Apply calibration. Routes to per-tier calibrator for new models,
        falls back to global IsotonicRegression for legacy models.

        New models also carry "_medium_corr" / "_house_corr" — small additive
        bias corrections (fit 2026-06-10) for systematic bias that the
        price-tier-only isotonic calibration above doesn't capture (e.g.
        photographs/prints under-predicted vs paintings; Christie's
        under-predicted vs Phillips/Bonhams). Missing on legacy models
        (.get returns {}), in which case this is a no-op.
        """
        if self.has_tiered_cal:
            tier = (
                "lt_10k"   if est_mid < 10_000    else
                "10k_50k"  if est_mid < 50_000    else
                "50k_200k" if est_mid < 200_000   else
                "200k_1m"  if est_mid < 1_000_000 else
                "gt_1m"
            )
            cal = self.calibrator.get(tier, self.calibrator["global"])
            p_cal = float(cal.predict([p_raw])[0])

            medium_corr = self.calibrator.get("_medium_corr", {})
            house_corr  = self.calibrator.get("_house_corr", {})
            p_cal += medium_corr.get(medium, medium_corr.get("_default", 0.0))
            p_cal += house_corr.get(house, house_corr.get("_default", 0.0))

            # Stage C: estimate-anchored shrink — blend toward the estimate
            # (log_ratio=0) by a val-fit λ so the model is never worse than the
            # specialist baseline. Per-tier λ overrides global where present.
            shrink = self.calibrator.get("_estimate_shrink", {})
            if shrink:
                lam = shrink.get(tier, shrink.get("global", 1.0))
                p_cal *= lam
            return p_cal
        return float(self.calibrator.predict([p_raw])[0])

    # ── sell-through probability ───────────────────────────────────────────
    def _sell_through_prob(self, X) -> float:
        """Predict P(lot sells). Returns 0.5 if model not available."""
        if self.sell_through_model is None:
            return 0.5
        # Sell-through model was trained on feature_cols minus log_estimate_mid
        try:
            drop_idx = self.feature_cols.index("log_estimate_mid")
            X_st = np.delete(X, drop_idx, axis=1).astype(np.float32)
        except (ValueError, IndexError):
            X_st = X.astype(np.float32)
        return float(self.sell_through_model.predict(X_st)[0])

    # ── predict ───────────────────────────────────────────────────────────
    def predict(self, **kw) -> dict:
        # Validate estimates up front — fail loudly instead of silently predicting
        # nonsense from est_mid=0 (e.g. estimate_low=None coerces to 0 downstream).
        try:
            _lo = float(kw.get("estimate_low") or 0)
            _hi = float(kw.get("estimate_high") or 0)
        except (TypeError, ValueError):
            raise ValueError(f"estimate_low/estimate_high must be numeric, got "
                             f"{kw.get('estimate_low')!r} / {kw.get('estimate_high')!r}")
        if _lo <= 0 or _hi <= 0:
            raise ValueError(f"estimates must be positive, got {_lo} / {_hi}")
        if _hi < _lo:
            raise ValueError(f"estimate_high ({_hi}) < estimate_low ({_lo})")

        X, est_mid, known, log_std, art_med = self._fvec(**kw)
        artist = str(kw.get("artist",""))
        is_low = est_mid < self.low_thr

        # Compute sell_through_prob from the original 42-feature X (before any augmentation).
        # If models were trained with sell_through_prob as feature #43, append it now so
        # all downstream model calls (lgbm, catboost, q10, q90, lo) see 43 cols.
        sell_through_prob = self._sell_through_prob(X)
        if self.meta.get("has_sell_through_feature", False):
            X = np.hstack([X, np.array([[sell_through_prob]], dtype=np.float32)])

        pmat = self._base_preds(X, artist, art_med)

        # Use NNLS weight vector if available, else weighted fallback
        if self.meta_learner is not None:
            w = np.asarray(self.meta_learner, dtype=np.float32)
            p_raw = float(pmat[0] @ w)
        else:
            # pmat columns: [lgbm, cat]
            p_l, p_c = pmat[0, 0], pmat[0, 1]
            total_w = self.w_l + self.w_c
            p_raw = (self.w_l*p_l + self.w_c*p_c) / max(total_w, 1e-6)

        # Low-value specialist override (<$10k)
        if is_low and self.lgbm_low is not None:
            p_raw = self.lgbm_low.predict(X)[0]

        # Isotonic calibration (per-tier for new models, global for legacy)
        # + medium/house group-bias correction (Stage B, see _apply_cal docstring)
        med_key   = str(kw.get("medium", "unknown")).lower().strip()
        house_key = str(kw.get("house", "christies")).lower().strip()
        p_cal = self._apply_cal(p_raw, est_mid, medium=med_key, house=house_key)

        # Convert from log_ratio or log_hammer space → dollar hammer
        if self.use_log_ratio:
            hammer = max(0, math.exp(p_cal) * est_mid)
            q10    = max(0, math.exp(self.lgbm_q10.predict(X)[0]) * est_mid)
            q90    = max(0, math.exp(self.lgbm_q90.predict(X)[0]) * est_mid)
        else:
            hammer = max(0, math.expm1(p_cal))
            q10    = max(0, math.expm1(self.lgbm_q10.predict(X)[0]))
            q90    = max(0, math.expm1(self.lgbm_q90.predict(X)[0]))

        # Tier-specific conformal prediction intervals (better coverage per price band)
        tier_offsets = self.meta.get("conformal_tiers", {})
        tier_key = (
            "lt_10k"   if est_mid < 10_000    else
            "10k_50k"  if est_mid < 50_000    else
            "50k_200k" if est_mid < 200_000   else
            "200k_1m"  if est_mid < 1_000_000 else
            "gt_1m"
        )
        t_lo = tier_offsets.get(tier_key, {}).get("lo", self.conformal_lo)
        t_hi = tier_offsets.get(tier_key, {}).get("hi", self.conformal_hi)
        if t_lo is not None and t_hi is not None:
            if self.use_log_ratio:
                conformal_p10 = max(0, math.exp(p_cal + t_lo) * est_mid)
                conformal_p90 = max(0, math.exp(p_cal + t_hi) * est_mid)
            else:
                conformal_p10 = max(0, math.expm1(max(0, p_cal + t_lo)))
                conformal_p90 = max(0, math.expm1(max(0, p_cal + t_hi)))
        else:
            conformal_p10 = q10
            conformal_p90 = q90

        # CQR intervals — adaptive per-lot widths from conformalized quantile models
        cqr_p10 = cqr_p90 = None
        if (self.cqr_lo_correction is not None and
                self.cqr_hi_correction is not None):
            raw_q10_lr = self.lgbm_q10.predict(X)[0]
            raw_q90_lr = self.lgbm_q90.predict(X)[0]
            cqr_q10_lr = raw_q10_lr - self.cqr_lo_correction
            cqr_q90_lr = raw_q90_lr + self.cqr_hi_correction
            if self.use_log_ratio:
                cqr_p10 = max(0, math.exp(cqr_q10_lr) * est_mid)
                cqr_p90 = max(0, math.exp(cqr_q90_lr) * est_mid)
            else:
                cqr_p10 = max(0, math.expm1(cqr_q10_lr))
                cqr_p90 = max(0, math.expm1(cqr_q90_lr))

        return {
            "predicted_hammer":        round(hammer),
            "hammer_to_estimate_ratio":round(hammer / max(est_mid, 1), 3),
            # Raw LGBM quantile model intervals (uncorrected)
            "confidence_p10":          round(q10),
            "confidence_p90":          round(q90),
            # CQR-corrected adaptive intervals (preferred for per-lot accuracy)
            "cqr_p10":                 round(cqr_p10) if cqr_p10 is not None else round(q10),
            "cqr_p90":                 round(cqr_p90) if cqr_p90 is not None else round(q90),
            # Split-conformal intervals (guaranteed marginal coverage)
            "conformal_p10":           round(conformal_p10),
            "conformal_p90":           round(conformal_p90),
            # Sell-through probability
            "sell_through_prob":       round(sell_through_prob, 3),
            "estimate_low":            kw.get("estimate_low"),
            "estimate_high":           kw.get("estimate_high"),
            "estimate_mid":            round(est_mid),
            "artist_known":            known,
            "low_value_segment":       bool(is_low),
        }
