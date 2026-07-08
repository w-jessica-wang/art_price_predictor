# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
generate_presale.py
Runs batch predictions on lots that did not record a hammer (sold==0) in the
features cache and saves results to data/no_hammer_batch_scores.csv for the
Batch Scoring demo tab.

CAVEAT: "sold==0" means the lot has no recorded hammer — that conflates genuinely
upcoming/unsold lots with bought-in (failed-to-sell) lots. The dashboard labels
these in a "Batch Scoring (demo)" tab; it is a demonstration of batch model
scoring, NOT a curated calendar of future auctions.
"""
import os, sys
import pandas as pd

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from ml.predict import ArtPricePredictor as Predictor
from clean_data import MIN_USD_ESTIMATE

def fmt_dollar(v):
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}k"
    return f"${v:,.0f}"

def main():
    print("Loading features cache...")
    df = pd.read_parquet(os.path.join(PROJECT_DIR, "models/features_cache.parquet"))

    # no_hammer = lots with no recorded hammer (sold==0). See module docstring caveat.
    no_hammer = df[df["sold"] == 0].copy()
    # Drop non-USD currency artifacts using the same floor as the cleaner.
    no_hammer = no_hammer[no_hammer["estimate_low"].fillna(0) >= MIN_USD_ESTIMATE]
    print(f"Found {len(no_hammer):,} no-hammer lots")

    if no_hammer.empty:
        print("No no-hammer lots found.")
        return

    print("Loading predictor...")
    predictor = Predictor(models_dir=os.path.join(PROJECT_DIR, "models"))

    results = []
    errors = 0
    for i, (idx, row) in enumerate(no_hammer.iterrows()):
        if i % 100 == 0:
            print(f"  {i}/{len(no_hammer)}...", end="\r")
        try:
            res = predictor.predict(
                artist=str(row.get("artist", "")),
                estimate_low=float(row.get("estimate_low") or 10_000),
                estimate_high=float(row.get("estimate_high") or 15_000),
                medium=str(row.get("medium", "unknown")),
                width_cm=float(row["width_cm"]) if pd.notna(row.get("width_cm")) and row.get("width_cm", 0) > 0 else None,
                height_cm=float(row["height_cm"]) if pd.notna(row.get("height_cm")) and row.get("height_cm", 0) > 0 else None,
                year_made=int(row["year_made"]) if pd.notna(row.get("year_made")) and row.get("year_made", 0) > 1800 else None,
                signed=bool(row.get("signed", False)),
                exhibited=bool(row.get("exhibited", False)),
                literature=bool(row.get("literature", False)),
                provenance=str(row.get("provenance", "")),
                house=str(row.get("house", "christies")),
                sale_year=int(row.get("sale_year", 2025)),
                sale_month=int(row.get("sale_month", 6)),
            )
            est_mid = (float(row.get("estimate_low") or 10_000) + float(row.get("estimate_high") or 15_000)) / 2
            edge_pct = (res["predicted_hammer"] / est_mid - 1) * 100 if est_mid > 0 else 0

            results.append({
                "artist":            row.get("artist", ""),
                "medium":            row.get("medium", "unknown"),
                "house":             row.get("house", ""),
                "sale_year":         int(row.get("sale_year", 2025)),
                "sale_month":        int(row.get("sale_month", 6)),
                "sale_date":         f"{int(row.get('sale_year',2025))}-{int(row.get('sale_month',6)):02d}-01",
                "estimate_low":      float(row.get("estimate_low") or 0),
                "estimate_high":     float(row.get("estimate_high") or 0),
                "estimate_mid":      est_mid,
                "predicted_hammer":  res["predicted_hammer"],
                "p10":               res["confidence_p10"],
                "p90":               res["confidence_p90"],
                "edge_pct":          edge_pct,
                "hammer_est_ratio":  res["hammer_to_estimate_ratio"],
                "artist_known":      res["artist_known"],
                # Formatted display
                "Est. Low":          fmt_dollar(float(row.get("estimate_low") or 0)),
                "Est. High":         fmt_dollar(float(row.get("estimate_high") or 0)),
                "Predicted Hammer":  fmt_dollar(res["predicted_hammer"]),
                "P10":               fmt_dollar(res["confidence_p10"]),
                "P90":               fmt_dollar(res["confidence_p90"]),
                "Edge %":            f"{edge_pct:+.0f}%",
            })
        except Exception as e:
            errors += 1
            if errors <= 5:   # surface the first few instead of silently swallowing
                print(f"\n  ⚠ prediction failed for '{row.get('artist','?')}': {type(e).__name__}: {e}")

    print(f"\nDone. {len(results)} predictions, {errors} errors.")

    out_df = pd.DataFrame(results)
    out_path = os.path.join(PROJECT_DIR, "data/no_hammer_batch_scores.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(f"\nTop model-flagged lots (predicted >40% above midpoint — illustrative, not advice):")
    top = out_df[out_df["edge_pct"] > 40].sort_values("edge_pct", ascending=False).head(10)
    print(top[["artist","house","sale_year","sale_month","Predicted Hammer","Edge %"]].to_string())

if __name__ == "__main__":
    main()
