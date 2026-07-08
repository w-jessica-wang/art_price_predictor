# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Leakage test — proves the point-in-time artist features use ONLY sales that
occurred *strictly before* each lot's sale_date (no same-date, no future leak).

This is the core correctness guarantee of the pipeline: artist-history features
(n_past_sales, median hammer, median ratio) are built via
    merge_asof(direction="backward", allow_exact_matches=False)
so a lot can never see its own sale, a concurrent same-date sale, or any future
sale of the same artist. The test constructs a synthetic timeline with KNOWN
answers and asserts the features match exactly.

Run:  pytest tests/ -q      (or)      python tests/test_leakage.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import load_data, compute_artist_features, temporal_split  # noqa: E402

CSV_COLS = ["house", "auction_id", "lot_number", "artist", "title", "year_made",
            "medium", "width_cm", "height_cm", "provenance", "signed", "exhibited",
            "literature", "estimate_low", "estimate_high", "sold_price", "sale_date"]


def _row(artist, title, date, sold_price):
    return {"house": "christies", "auction_id": "A1", "lot_number": 1, "artist": artist,
            "title": title, "year_made": 2000, "medium": "oil on canvas", "width_cm": 50,
            "height_cm": 40, "provenance": "Private collection", "signed": "true",
            "exhibited": "false", "literature": "false", "estimate_low": 10000,
            "estimate_high": 15000, "sold_price": sold_price, "sale_date": date}


def _featurize():
    """Build a synthetic CSV, run the real load_data + feature pipeline, index by title."""
    rows = [
        # Artist A — three sales at strictly increasing dates (known prior-counts)
        _row("Artist A", "A_lot1", "2020-01-15", 12000),
        _row("Artist A", "A_lot2", "2021-06-10", 20000),
        _row("Artist A", "A_lot3", "2022-03-01", 30000),
        # Artist B — two sales on the SAME date (same-date no-leak case)
        _row("Artist B", "B_lot1", "2021-05-05", 18000),
        _row("Artist B", "B_lot2", "2021-05-05", 22000),
        # Filler so global statistics are well-defined
        _row("Artist C", "C_lot1", "2019-02-02", 9000),
        _row("Artist C", "C_lot2", "2020-08-08", 11000),
        _row("Artist D", "D_lot1", "2019-09-09", 50000),
        # Lots spanning the 2025 split boundary (train ≤2024 / val 2025 H1 / test 2025 H2)
        _row("Artist E", "E_val",  "2025-02-01", 25000),
        _row("Artist E", "E_test", "2025-08-01", 35000),
    ]
    df = pd.DataFrame(rows, columns=CSV_COLS)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        feat = compute_artist_features(load_data(path))
    finally:
        os.unlink(path)
    return feat.set_index("title")


def test_n_past_sales_is_strictly_prior():
    """A lot's prior-sale count reflects only sales BEFORE its own date."""
    f = _featurize()
    assert int(f.loc["A_lot1", "artist_n_past_sales"]) == 0   # first ever sale
    assert int(f.loc["A_lot2", "artist_n_past_sales"]) == 1   # sees only A_lot1
    assert int(f.loc["A_lot3", "artist_n_past_sales"]) == 2   # sees A_lot1, A_lot2


def test_same_date_does_not_leak():
    """Two lots on the SAME date must not see each other (strictly-before join)."""
    f = _featurize()
    assert int(f.loc["B_lot1", "artist_n_past_sales"]) == 0
    assert int(f.loc["B_lot2", "artist_n_past_sales"]) == 0


def test_feature_value_uses_only_prior_sales():
    """A_lot2's artist median hammer must equal A_lot1's hammer (its sole prior sale),
    not its own and not the later A_lot3."""
    f = _featurize()
    expected = np.log1p(12000)            # A_lot1's log hammer
    assert abs(float(f.loc["A_lot2", "artist_median_log_hammer"]) - expected) < 1e-6


def test_market_index_does_not_see_current_month():
    """A month's market median is computed from the WHOLE month, so it must not be
    visible to lots sold within that same month. We create a sharp price regime
    shift in April and assert April's lots still reflect the pre-April level — i.e.
    their market_index was built from strictly-earlier months, not their own."""
    rows = []
    # Three baseline months at a low price level (well-defined monthly medians)
    for mo in ("2020-01", "2020-02", "2020-03"):
        for k in range(8):
            rows.append(_row(f"Base{k}", f"b_{mo}_{k}", f"{mo}-15", 10_000))
    # April: a 100x regime shift — if the current month leaked, April lots would
    # see this huge level in their own market_index.
    for k in range(8):
        rows.append(_row(f"Apr{k}", f"apr_{k}", "2020-04-15", 1_000_000))
    df = pd.DataFrame(rows, columns=CSV_COLS)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fh:
        df.to_csv(fh.name, index=False)
        path = fh.name
    try:
        feat = compute_artist_features(load_data(path)).set_index("title")
    finally:
        os.unlink(path)

    apr = float(feat.loc["apr_0", "market_index_3m"])
    low, high = np.log1p(10_000), np.log1p(1_000_000)
    # April must be far closer to the pre-April level than to its own month's level
    assert abs(apr - low) < abs(apr - high), (
        f"market_index leaked the current month: got {apr:.3f}, "
        f"pre-April≈{low:.3f}, April-own≈{high:.3f}"
    )


def test_temporal_split_has_no_date_overlap():
    """train < val < test in time — the split itself introduces no leakage."""
    f = _featurize()
    tr, va, te = temporal_split(f.reset_index())
    if len(tr) and len(va):
        assert tr["sale_date"].max() < va["sale_date"].min()
    if len(va) and len(te):
        assert va["sale_date"].max() < te["sale_date"].min()


if __name__ == "__main__":
    test_n_past_sales_is_strictly_prior()
    test_same_date_does_not_leak()
    test_feature_value_uses_only_prior_sales()
    test_market_index_does_not_see_current_month()
    test_temporal_split_has_no_date_overlap()
    print("✅ all leakage tests passed — artist + market features are strictly point-in-time")
