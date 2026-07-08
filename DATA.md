# Data Card — Fine-Art Auction Dataset

*Following the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al., 2018) framework. Companion to the [Model Card](model_card.md) and [Technical Report](TECHNICAL_REPORT.md).*

> **The dataset itself is not released.** This card documents what it contains and how it was built so the modeling results can be assessed. The raw data and the collection code are excluded from this repository by design.

---

## 1. Overview

| | |
|---|---|
| **What** | Lot-level fine-art auction results: artwork metadata, pre-sale estimates, and realized hammer prices |
| **Sources** | Publicly available auction results from four houses: Christie's, Bonhams, Phillips, Sotheby's |
| **Period** | 2018-02-07 → 2026-05-31 |
| **Size** | **150,808** cleaned fine-art lots · **129,023** with a hammer price (86%) |
| **Unit** | One row = one auction lot |
| **Purpose** | Research: can a model add anything beyond the specialist's own pre-sale estimate? |
| **Released?** | No — prices/metadata are the houses' published results; only derived artifacts (metrics, hashes, figures) are public |

## 2. Collection (high level)

Records were compiled from the houses' **publicly published auction results** — the same lot pages any visitor can read — and normalized into a single schema. All prices are converted to USD hammer prices (non-USD lots converted at end-of-month rates for the sale date). Withdrawn lots never appear (they are not published); lots that failed to sell (bought-in) appear without a hammer price.

The collection code is deliberately not part of this repository. Reproducibility is provided at the *artifact* level instead: the cleaned dataset's SHA-256 is recorded in [`models/eval_manifest.json`](models/eval_manifest.json), and every reported metric traces to hashed artifacts (see §6).

## 3. Schema

17 raw fields per lot. "Populated" = non-empty share of the 150.8k cleaned rows.

| Field | Type | Populated | Description |
|---|---|---|---|
| `house` | str | 100% | Auction house (christies / bonhams / phillips / sothebys) |
| `auction_id` | str | 100% | Sale identifier (2,347 unique sales) — used for sale-level clustering |
| `lot_number` | str | 100% | Lot number within the sale |
| `artist` | str | 100% | Canonicalized artist name (43,761 unique; birth/death years and honorifics stripped) |
| `title` | str | 96% | Artwork title |
| `year_made` | int | 39% | Year of creation, where catalogued |
| `medium` | str | 67% | Normalized medium ("oil on canvas", "screenprint", …); remaining 33% = "unknown" |
| `width_cm`, `height_cm` | float | 74% | Physical dimensions, normalized to cm |
| `provenance` | str | 68% | Free-text ownership/exhibition history as catalogued |
| `signed` | bool | 100% | Catalogue indicates the work is signed (71% True) |
| `exhibited` | bool | 100% | Catalogue lists an exhibition history (19% True) |
| `literature` | bool | 100% | Catalogue cites literature (32% True) |
| `estimate_low`, `estimate_high` | float (USD) | 100% | Specialist pre-sale estimate band ($200 – $550M) |
| `sold_price` | float (USD) | 86% | Hammer price; missing ⇒ bought-in / no recorded hammer |
| `sale_date` | date | 100% | Sale date |

## 4. Composition

| House | Lots | Share |
|---|---|---|
| Christie's | 86,252 | 57.2% |
| Bonhams | 44,425 | 29.5% |
| Phillips | 18,912 | 12.5% |
| Sotheby's | 1,219 | 0.8% |

- **By year (approx.):** 2018: 15.1k · 2019: 21.0k · 2020: 13.4k · 2021: 17.9k · 2022: 20.6k · 2023: 20.1k · 2024: 25.5k · 2025: 14.6k · 2026 (through May): 2.7k
- **Hammer prices:** $12 – $323.6M (median ≈ $17k — the dataset is mid-market-heavy, not evening-sale-only)
- **Sell-through:** ~86% of lots have a recorded hammer
- **Known coverage gap:** Christie's records end **2025-03-27** (source-side publication lag at collection time) — this drives the evaluation design (see §7)

## 5. Cleaning & quality control

From **170,352** raw collected rows, cleaning removes **19,558** (11.5%). Reproducible via `python clean_data.py --report`:

| Removed for | Lots |
|---|---|
| Wine / spirits / collectibles (not fine art) | 8,048 |
| Junk / non-artist "artist" strings | 5,845 |
| Duplicates (within- and cross-house) | 4,839 |
| Estimate < $200 (likely non-USD currency artifact) | 716 |
| Extreme sold/estimate ratio (>50× or <1% — data errors) | 109 |
| `estimate_high < estimate_low` | 1 |
| **Total removed** | **19,558** |
| **Fresh re-clean output** | **150,794** (= 170,352 − 19,558) |

> **Deployed-copy note:** the trained artifacts use an operationally maintained copy of the cleaned dataset — **150,808 total · 129,023 priced** (SHA-256 recorded in `models/eval_manifest.json`). It differs from a fresh end-to-end re-clean by **14 rows (<0.01%)** due to deduplication tie-ordering between cleaning runs. All reported metrics are computed on the deployed copy.

Notable steps:
- **Fine-art-only guard.** The houses also sell wine and spirits, which leak in with the producer/vintage as the "artist" (e.g. *"Château Latour 1863"*). A two-tier filter (unambiguous producer/spirit terms always; ambiguous terms like *latour* only with a vintage year, so **Henri Fantin-Latour survives**) removes them — enforced both in the cleaner and at load time, and unit-tested (`tests/test_data_quality.py`).
- **Artist canonicalization.** Trailing life dates, honorifics, and case variants are stripped so an artist's history aggregates correctly.
- **Deduplication** on a fuzzy (canonicalized artist, first-30-chars-of-title, sale year-month) key — catches within-house relistings and cross-house double-postings.
- **Currency.** All values USD; sub-$200 estimates are treated as unconverted-currency artifacts and dropped.

## 6. Provenance & integrity

- The cleaned CSV's **SHA-256** and the feature cache's hash are recorded in `models/eval_manifest.json`, along with full hashes of every model artifact, package versions, and platform.
- `tests/test_docs_sync.py` asserts that the headline numbers quoted across the README, model card, and this data card match the trained artifacts, and that known-stale numbers never reappear in any of the four docs — documentation cannot silently drift from the data. It runs locally via `pytest` and on every push via GitHub Actions (`.github/workflows/ci.yml`).

## 7. Splits & evaluation design

- **Deployed-model split (temporal, never shuffled):** train ≤ 2024 · validation 2025 H1 (fits calibration/shrink/conformal) · test 2025 H2 (n = 4,891; contains no Christie's due to the coverage gap).
- **Primary evaluation:** purged walk-forward CV — 5 rolling quarterly folds with a 1-month embargo, full calibration chain refit per fold; restores **5,631 Christie's lots** to clean out-of-sample testing. See the Technical Report.

## 8. Limitations & known biases

1. **Survivorship / publication bias.** Withdrawn lots are absent; the data conditions on "offered and published."
2. **House imbalance.** Sotheby's is 0.8% of rows — house-specific conclusions there are unsupported.
3. **Christie's date cap (2025-03-27).** Recent Christie's is missing; handled by the walk-forward design, but any single recent window under-represents them.
4. **Catalogue-quality missingness is not random.** `year_made` (65% populated for ≥$100k lots vs 23% below $10k) and `provenance` (84% vs 52%) are far better populated for higher-value lots; models see missingness patterns correlated with value. (`medium` missingness is roughly value-neutral.)
5. **Bought-in lots have no hammer**, so the price model is conditional on selling; sell-through is modeled separately as a feature.
6. **Mid-market skew.** The long tail of sub-$10k lots dominates counts; headline metrics are reported per price tier for this reason.

## 9. Intended use & ethics

- **Intended:** research into estimate efficiency and price modeling; the portfolio app in this repository.
- **Not intended:** resale/redistribution of the data, investment decision-making, or scraping guidance.
- **Privacy:** rows describe artworks and public sale outcomes, not private individuals; consignor/buyer identities are not collected. Living artists appear only via their publicly attributed sale records.
- **Licensing posture:** the underlying results are the auction houses' published information; this project does not republish the data, only aggregate statistics and model artifacts derived from it.

---

*Last updated: June 2026. Companion documents: [README](README.md) · [Model Card](model_card.md) · [Technical Report](TECHNICAL_REPORT.md).*
