# Are Auction Estimates Beatable? A Leakage-Controlled, Multi-House Study

**Jessica Wang · University of Michigan · 2026**
*Proprietary — All Rights Reserved (see [LICENSE](LICENSE))*

---

## TL;DR

I built a leakage-controlled model to predict fine-art auction hammer prices and asked the only question that matters: **does it add anything beyond the auction house's own pre-sale estimate?** On a self-assembled, four-house, 129k-priced-lot (fine-art-only) dataset (2018–2026), the answer is **only marginally on point accuracy** — the specialist estimate is *near-efficient*. Under purged walk-forward CV (5 quarters, 18.6k OOS lots, Christie's included) the model beats the estimate on MedAPE (29.4% vs 30.2%) and R² (0.897 vs 0.887) — winning every fold; on the single most-recent quarter in isolation it ties (29.2% each). Where it adds genuine, *stable* value is **ranking** — sorting which lots beat vs. miss their estimate, AUC 0.62 [0.61–0.63] and steady in every fold — and **calibrated uncertainty**. **This independently corroborates the efficient-estimate finding of Mei & Moses (2002, 2025) — via a different methodology (cross-sectional metadata rather than repeat-sales) and a different dataset** — and is, I believe, a more useful confirmation *because* it failed to beat the baseline honestly rather than appearing to via leakage.

---

## 1. Problem framing

A naïve "predict the price" framing is misleading: the pre-sale estimate already explains ~87% of log-price variance, so any model that ingests the estimate will look accurate while contributing nothing. The honest target is therefore the **premium/discount over the estimate**:

> `target = log(hammer_price / estimate_mid)`

and the honest baseline is **the estimate itself** (`ratio = 1`). Every number below is the model *vs. that baseline*.

## 2. Data

*(Full schema, composition, and cleaning audit: see the [Data Card](DATA.md).)*

- **Source:** publicly available results from Christie's, Bonhams, Phillips, Sotheby's (2018–2026), self-assembled and normalized (currency, dedup, junk filtering, artist-name canonicalization). *Raw data and collection code are not released.*
- **Size:** **129,023 priced lots** (of 150,808 fine-art-only total; wine/spirits excluded; ~22k unsold lots feed a separate batch-scoring demo).
- **Features (42):** estimate signals, point-in-time artist history, artwork attributes, provenance flags, market-regime + S&P-500 macro signals, seasonality.

## 3. Method

- **Ensemble:** LightGBM + CatBoost, combined by non-negative least-squares (NNLS) stacking on out-of-fold predictions. (XGBoost and a neural net were evaluated and dropped — they added <0.06pp; this also removed the PyTorch dependency.)
- **Three-stage calibration (all fit on validation only):** (A) tiered isotonic; (B) medium/house group-bias correction; (C) an **evidence-gated, estimate-anchored shrink** — each price tier blends toward the estimate by a factor λ that is (i) only non-zero if it beats the estimate on that validation tier by a margin, and (ii) confidence-weighted by sample size. Because λ=0 is always available, the model is *provably no worse than the estimate on validation*. The evidence gate is a regularization strategy **designed to improve out-of-sample robustness**; parity-or-improvement was *observed* across the evaluated forward folds, but is not mathematically guaranteed there.
- **Uncertainty:** Conformalized Quantile Regression (CQR; Romano et al., 2019) for P10–P90 intervals.
- **Validation:** strictly temporal (train ≤2024, val 2025 H1, test 2025 H2); all artist-history features use `merge_asof(direction="backward", allow_exact_matches=False)` so a lot can never see its own, concurrent, or future sales. **This guarantee is unit-tested** (`tests/test_leakage.py`).

## 4. Results — primary: purged walk-forward CV (5 quarters, n = 18,643 OOS)

The headline evaluation is a **purged, rolling-origin walk-forward CV** (the standard rigorous protocol for time-series models), not a single split. Five folds each train on the past, refit the *entire* calibration chain (isotonic → group-bias → Stage-C → CQR) on their own validation quarter, and test on the next quarter with a 1-month embargo at the seam; predictions are pooled. This also restores **Christie's** (57.2% of training, capped 2025-03-27 and thus absent from any recent single split) to the test set with **5,631 clean OOS lots**.

| Metric | Estimate-only | Model | |
|---|---|---|---|
| MedAPE (all lots) | 30.2% | **29.4%** | model wins |
| Within-20% (all lots) | 36.9% | **37.0%** | ≈ tie |
| Raw-$ R² (log space) | 0.887 | **0.897** | model wins |
| Ranking AUC | 0.50 | **0.62** [0.61–0.63] | real signal |
| 80% interval coverage | n/a | **80.4%** | calibrated ✓ |

The MedAPE edge is small but **statistically supported**: an artist-clustered paired bootstrap of Δ = MedAPE(model) − MedAPE(estimate) over the pooled OOS lots gives **Δ = −0.85pp, 95% CI [−1.12, −0.24]** (excludes zero; the model is better in 100% of resamples). **The story is stability, not a single number.** MedAPE: model wins **all 5 quarters** (closest is 2024 Q4, 28.3 vs 29.7). Ranking AUC by quarter is 0.63 / 0.65 / 0.63 / 0.62 / 0.59 — steady throughout. Per-house (pooled): Christie's 32.8/33.9 (win, AUC 0.61), Bonhams 27.8/28.9 (win, 0.61), Phillips 27.9/29.2 (win, 0.61). *Walk-forward measures how the modeling approach generalizes across time — each fold is a separate fit.*

**Secondary — single deployed artifact, 2025 H2 holdout (n = 4,891):** the one most-recent quarter, held out of training and of the validation set that fits calibration. It is a harder, Christie's-free window and reads as a tie (MedAPE 29.2 / 29.2; R² 0.890 / 0.882; within-20 36.8 / 37.4; coverage 82.2%) — a conservative lower bound consistent with the walk-forward result.

Either way, against a *near-efficient* benchmark a large point-accuracy win would be a red flag: an unconstrained ensemble does measurably *worse* on MedAPE (31.6%); the Stage-C shrink (λ=0 in-grid ⇒ provably no worse than the estimate on validation) keeps the model at-or-above the baseline rather than drifting.

## 5. Where the real signal lives

The modest point-accuracy edge sits alongside a real, statistically unambiguous **ranking** skill that is stable across every fold. Pooled over the walk-forward, the model's unshrunk signal sorts whether a lot will exceed its **estimate midpoint**:

> **Ranking AUC = 0.62**, 95% CI excludes 0.5 under both an artist-clustered bootstrap **[0.61, 0.64]** and a sale-clustered (auction-level) bootstrap **[0.61, 0.64]** — robust to either correlation unit.
> **Decile lift: bottom decile 50% → top decile 83%** beat the midpoint, perfectly monotonic (Spearman ρ = 1.0).

The skill survives subgroup splits: pooled per-house AUC is ≈0.61 for all three houses (Christie's, Bonhams, Phillips); per-quarter it stays in 0.60–0.65; and it is stronger for artists with history than thin-history artists. The estimate itself carries no such ranking signal by construction (AUC 0.50). This ranking view is something the aggregate-R²/MAPE literature does not report. See `figures/walk_forward.png` and `figures/decile_lift.py`.

## 6. Contribution, honestly stated

1. **Independent, leakage-controlled, multi-house corroboration** of the efficient-estimate finding — different method (cross-sectional vs. repeat-sales) and data than Mei & Moses. Under purged walk-forward CV the model beats the estimate slightly on point accuracy in every fold and on every house (incl. Christie's) — a small but consistent edge over a near-efficient baseline.
2. **A ranking-skill characterization** (clustered-CI AUC, decile lift with monotonicity test, per-quarter and per-house breakdowns) that surfaces the residual signal aggregate metrics miss, and is stable across five forward quarters.
3. **Calibrated uncertainty** (empirically 80–82% coverage across evaluations, 80% nominal) — the deliverable the point estimate cannot provide.
4. **A clean negative result that reinforces the thesis:** eight richer provenance/title text features (documentation depth, blue-chip dealer & prestige cues, title language) were added and evaluated *under the same walk-forward CV*. They were **net neutral-to-slightly-negative** (pooled MedAPE +0.2pp worse, ranking AUC −0.005) and rejected — direct evidence that the specialist already prices the lot-record text, so signal of that kind cannot beat the estimate. (Inspection of the collected data showed `literature`/`exhibited` carry no free text in the source, so only provenance/title text was available.)

## 7. Limitations

- Point-accuracy edge over the estimate is small (sub-1pp MedAPE) — an efficient-market result, not a modeling failure.
- **Recent Christie's gap:** the Christie's data caps at 2025-03-27. Walk-forward restores Christie's to the *earlier* test folds (5,631 OOS lots), but the most recent folds and the deployed model's H2 holdout still lack recent Christie's. Collecting post-March-2025 Christie's results is the highest-value next step.
- **Ranking caveats:** decile thresholds are formed on the evaluation sample (a lift-chart convention); the bootstrap is clustered by artist and, as a sensitivity, by sale — but not two-way jointly; results are a single seed; walk-forward pools across folds that are each a separate fit (it estimates the approach's generalization, not one frozen model).
- **Metadata-only** — no artwork-image features. (The 2025 SOTA used images and still did not beat the estimate.)
- Single-author, data not publicly released (so not externally reproducible).

## References

- Mei, J. & Moses, M. (2002). *Art as an Investment and the Underperformance of Masterpieces.* **American Economic Review** 92(5), 1656–1668.
- Mei, J., Moses, M., & Wälty, J. (2025). *Deep Learning for Art Market Valuation.* (Estimates remain near-efficient under deep learning with images.)
- Romano, Y., Patterson, E., & Candès, E. (2019). *Conformalized Quantile Regression.* NeurIPS.
- Angelopoulos, A. & Bates, S. (2023). *A Gentle Introduction to Conformal Prediction.*
