# Model Card — Art Auction Price Predictor

*Following the [Google Model Card](https://modelcards.withgoogle.com/about) framework.*

---

## Model Details

| Field | Value |
|---|---|
| **Name** | Art Auction Price Predictor (Ensemble v2) |
| **Version** | 2.2 |
| **Date** | June 2026 |
| **Author** | Jessica Wang, University of Michigan |
| **License** | Proprietary — All Rights Reserved |
| **Contact** | See GitHub repository |

### Architecture

A stacked ensemble of **two deployed base learners** combined via non-negative least-squares (NNLS) meta-learning. (XGBoost and a neural net were trained during development; a validation ablation showed the neural net changed every metric by <0.06pp and XGBoost got zero NNLS weight, so both were dropped — which also removes the PyTorch dependency.)

| Component | Role | Weight (NNLS) |
|---|---|---|
| LightGBM | Base learner + quantile models (P10/P90) | 0.66 |
| CatBoost | Base learner (handles artist name natively as categorical) | 0.34 |
| NNLS meta-learner | Stacks base learners on out-of-fold predictions | — |
| Stage A — tiered isotonic calibration | Corrects price-tier bias in log space | — |
| Stage B — group-bias correction | Additive medium/house offsets | — |
| Stage C — estimate-anchored shrink | Blends toward the estimate by a per-tier `λ` (val-fit), confidence-weighted by sample size; provably no worse than the estimate on validation (λ=0 in-grid) | — |

**Target variable:** `log(hammer_price / estimate_mid)` — the log **premium/discount over the specialist estimate**, not the raw price. This makes the learning problem currency- and scale-agnostic and focuses the model on the only thing the estimate doesn't already tell you: whether a lot beats or misses its estimate.

**Prediction intervals:** LightGBM quantile models (α=0.10/0.90) conformalized via CQR (Romano et al., 2019) on validation. Empirical coverage on the held-out 2025 H2 test set is **82.2%** (target 80%).

### Training Infrastructure

- Python 3.11, scikit-learn 1.7, LightGBM 4.6, CatBoost 1.2.10 (exact pins in `requirements.txt`; no PyTorch / GPU required)
- CPU-only; runs on a laptop
- Training time: ~1.5 min on CPU with cached features (longer with full feature engineering from raw)

---

## Intended Use

### Primary Use Cases

- **Pre-auction valuation:** Given a specialist estimate and artwork metadata, predict where the hammer will fall relative to the estimate.
- **Portfolio analysis:** Batch-score a collection against historical auction comps.
- **Research:** Study what drives price premiums/discounts relative to specialist estimates.

### Intended Users

- Art market researchers and academics
- Collectors performing due diligence
- Art advisors and dealer principals

### Out-of-Scope Uses

- **Not financial advice.** Outputs are statistical estimates, not investment recommendations.
- **Not suitable for private sales.** The model is trained exclusively on public auction results.
- **Not suitable for non-Western auction markets.** Training data covers Christie's (NY/London/HK), Bonhams, Phillips, and Sotheby's only.
- **Not suitable for decorative arts, jewelry, or watches.** The dataset covers fine art only.

---

## Training Data

*Full dataset documentation — schema, composition, cleaning audit, provenance — is in the [Data Card](DATA.md).*

| Attribute | Value |
|---|---|
| Total cleaned lots | 150,808 (fine art only) |
| Lots with hammer price | 129,023 (86%) |
| Christie's | 86,252 (57.2%) |
| Bonhams | 44,425 (29.5%) |
| Phillips | 18,912 (12.5%) |
| Sotheby's | 1,219 (0.8%) |
| Date range | January 2018 – 2026 |
| Sell-through rate | ~85% (lots that received a hammer bid) |
| Validation set | 2025 H1, n=6,990 (fits calibration + Stage-C + CQR) |
| Test set | 2025 H2, n=4,891 (clean held-out; headline numbers) |

> **Christie's date cap:** the Christie's data coverage ends at **2025-03-27**, so it falls entirely in the 2025 H1 validation window. The clean 2025 H2 test set contains **no Christie's** — it is ~49% Bonhams / 51% Phillips. See *Known Data Biases* and *Performance Metrics*.

### Data Collection

The dataset was compiled from publicly available auction results. All prices are hammer prices in USD (non-USD lots converted using end-of-month exchange rates from the sale date). The data-collection code is not included in this repository.

**Cleaning audit (lots removed by reason).** From **170,352** raw collected lots, cleaning removes **19,558** (11.5%), leaving ~150.8k fine-art lots (`clean_data.py --report`, reproducible):

| Removed for | Lots |
|---|---|
| Wine / spirits / collectibles (not fine art) | 8,048 |
| Junk / non-artist "artist" names | 5,845 |
| Duplicates (cross-house + within-house) | 4,839 |
| Estimate < $200 (likely non-USD artifact) | 716 |
| Extreme sold/estimate ratio (>50× or <1%) | 109 |
| `estimate_high < estimate_low` | 1 |
| **Total removed** | **19,558** |
| **Fresh re-clean output** | **150,794** (= 170,352 − 19,558) |

> **Deployed-copy note:** the trained artifacts use an operationally maintained copy of the cleaned dataset — **150,808 total · 129,023 priced** (SHA-256 in `models/eval_manifest.json`); it differs from a fresh re-clean by 14 rows (<0.01%, deduplication tie-ordering). Metrics are computed on the deployed copy.

The wine/spirits exclusion is unit-tested (`tests/test_data_quality.py`) and spares real painters (e.g. Henri Fantin-Latour).

### Known Data Biases

- **House imbalance:** Sotheby's is severely underrepresented (0.8%). Predictions for Sotheby's lots should be treated with wider uncertainty.
- **Train/test house-composition shift (addressed by walk-forward):** Because the Christie's data caps at 2025-03-27, the deployed model's single 2025 H2 holdout is **Bonhams + Phillips only**. The primary walk-forward evaluation fixes this — its earlier folds test on **5,631 clean out-of-sample Christie's lots**, where the model now **wins** (32.8 vs 33.9). The residual gap is *recent* Christie's (post-March-2025), pending updated collection.
- **Survivorship bias:** Lots withdrawn before the sale are not included. The model therefore does not predict the probability of a lot being sold — only the conditional hammer price given it sells.

---

## Performance Metrics

### Does the model beat the estimate? Primary: walk-forward CV (5 quarters, n=18,643 OOS)

The auctioneer's low–high estimate is a near-efficient point predictor; the honest question is whether the model adds anything beyond it. The primary evaluation is a **purged walk-forward CV** — five rolling quarters, the full calibration chain refit per fold, a 1-month embargo at each seam, predictions pooled. This restores **Christie's** (capped 2025-03-27, absent from any recent single split) with **5,631 clean OOS lots**.

| Metric | Estimate-only baseline | **Ensemble (+ Stage-C shrink)** | Verdict |
|---|---|---|---|
| MedAPE (all lots) | 30.2% | **29.4%** | model wins |
| Within-20% (all lots) | 36.9% | **37.0%** | ≈ tie (model a hair ahead) |
| Raw-$ R² (log space) | 0.887 | **0.897** | model wins |
| **Ranking AUC** (beats vs. misses midpoint) | 0.50 | **0.62** [0.61–0.63] | **model adds signal** |
| 80% interval empirical coverage | n/a | **80.4%** | ✅ calibrated |

Stability: MedAPE wins **all 5 quarters** (closest is 2024 Q4, 28.3 vs 29.7); ranking AUC stays 0.59–0.65 every quarter. *Walk-forward measures how the modeling approach generalizes across time — each fold is a separate fit.* An *unconstrained* ensemble does **worse** on MedAPE (31.6%); the **Stage-C estimate-anchored shrink** (`λ=0` in-grid ⇒ provably no worse than the estimate on validation, with an evidence gate + sample-size shrinkage `w=n/(n+k)`) keeps the model at-or-above the baseline rather than drifting.

**Secondary — deployed artifact, 2025 H2 holdout (n=4,891):** the one most-recent quarter, held out of training and of the validation set that fits calibration. A harder, Christie's-free window → reads as a tie (MedAPE 29.2/29.2, within-20 36.8/37.4, R² 0.890/0.882, coverage 82.2%, ranking AUC 0.60). A conservative lower bound consistent with the walk-forward.

### Ranking skill (pooled walk-forward — the actual contribution)

The unshrunk ensemble signal sorts whether a lot's hammer will exceed its **estimate midpoint** (base rate ≈61%):

| Metric | Value |
|---|---|
| AUC | **0.62** — 95% CI excludes 0.5 under artist-clustered **[0.61, 0.64]** AND sale-clustered **[0.61, 0.64]** bootstraps |
| Decile lift | bottom 50% → top 83% beat midpoint, perfectly monotonic (Spearman ρ=1.0) |
| Per-quarter AUC | 0.63 · 0.65 · 0.63 · 0.62 · 0.59 (steady) |
| Per-house AUC (pooled) | Christie's 0.61 · Bonhams 0.61 · Phillips 0.61 |

### By Auction House — pooled walk-forward (MedAPE: model / estimate-only)

Walk-forward restores Christie's to the out-of-sample test set (5,631 lots). Each house is scored against *its own* estimates.

| House | OOS test lots | Model | Estimate-only | Notes |
|---|---|---|---|---|
| Christie's | 5,631 | **32.8%** | 33.9% | Model wins; now cleanly testable |
| Bonhams | 6,541 | **27.8%** | 28.9% | Model wins |
| Phillips | 6,471 | **27.9%** | 29.2% | Model wins |
| Sotheby's | ~0 | — | — | Negligible in data |

*Pooled over the walk-forward the model ties-or-beats the estimate-only baseline in every house.*

---

## Limitations

1. **Sotheby's thin coverage.** With only 0.8% of training data from Sotheby's, predictions for that house rely almost entirely on cross-house generalization. Confidence intervals are materially wider.

2. **Unknown artists.** For artists with fewer than 5 recorded lots, the model falls back to global median statistics. The "Artist History" panel in the app flags this explicitly.

3. **Lots below $10k.** The primary ensemble is less reliable below the $10,000 mark. A dedicated sub-model (`lgbm_low.txt`) is used for these lots, but accuracy is still lower than the mid-market.

4. **No live market data.** Market regime features (`market_index_3m`, `market_index_12m`) are computed from the training data and held fixed at inference time. A sudden market shift (e.g., a financial crisis) would not be captured until the model is retrained.

5. **Non-English titles and provenance.** Provenance parsing uses regex patterns tuned for English-language text. French and Chinese-language provenance notes are partially parsed.

6. **Temporal drift.** Art market dynamics shift over multi-year cycles. The model should be retrained annually.

---

## Ethical Considerations

- **Not financial advice.** Hammer-price predictions carry substantial uncertainty (median absolute error ~29%, on par with the specialist estimate itself). Decisions involving significant capital should involve qualified human expertise.
- **Market sensitivity.** Publishing systematic over/under-valuation signals for specific artists could, in theory, influence market prices. Use responsibly.
- **Artist privacy.** The model uses only publicly available auction records. No personally identifiable information about living artists beyond their publicly attributed sale records is used.
- **Fairness.** The model is trained on Christie's/Bonhams/Phillips/Sotheby's results which skew toward Western European and American art. Artists from underrepresented traditions (African, South Asian, Latin American) may have fewer training samples and lower prediction accuracy.

---

## How to Cite

```bibtex
@software{art_auction_predictor_2026,
  author       = {Wang, Jessica},
  title        = {Art Auction Price Predictor},
  year         = {2026},
  institution  = {University of Michigan},
  url          = {https://github.com/w-jessica-wang/art_price_predictor},
  note         = {Two-model NNLS ensemble (LightGBM + CatBoost) with tiered
                  isotonic calibration and CQR conformal intervals, trained on
                  ~129k priced lots (2018--2026), fine art only. Training data not publicly released.}
}
```

---

*Last updated: June 2026. Model version: 2.2 (fine-art-only dataset, walk-forward primary evaluation, Stage-C estimate-anchored shrink).*
