# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Inference-contract tests — the predictor returns a sane, well-formed result.
Skipped automatically when trained model artifacts aren't present (e.g. CI on a
fresh clone), so the suite stays green without committing the model.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(_MODELS, "lgbm.txt")),
    reason="trained model artifacts not present — run `python -m ml.train` first",
)


def _predict():
    from ml.predict import ArtPricePredictor
    p = ArtPricePredictor()
    return p.predict(artist="Andy Warhol", medium="screenprint", house="bonhams",
                     estimate_low=80_000, estimate_high=120_000,
                     width_cm=90, height_cm=90, year_made=1972,
                     sale_date="2025-09-01", provenance="Private collection")


def test_predict_returns_expected_keys():
    r = _predict()
    for k in ("predicted_hammer", "conformal_p10", "conformal_p90", "estimate_mid"):
        assert k in r, f"missing key: {k}"


def test_predicted_hammer_positive_and_finite():
    r = _predict()
    assert r["predicted_hammer"] > 0
    assert math.isfinite(r["predicted_hammer"])


def test_conformal_interval_is_ordered_and_positive():
    r = _predict()
    assert 0 < r["conformal_p10"] < r["conformal_p90"]


def test_inference_uses_persisted_momentum_and_market():
    """Train/inference parity — the deployed predictor must consume the persisted
    latest per-artist momentum and the latest market index, NOT hardcoded 0.0 /
    a flat constant. Otherwise the served model sees different features than the
    one whose metrics we report."""
    import json
    import pandas as pd

    stats = pd.read_parquet(os.path.join(_MODELS, "artist_stats.parquet"))
    for c in ("artist_momentum_6m", "artist_momentum_12m", "artist_momentum_accel"):
        assert c in stats.columns, f"artist_stats missing persisted column: {c}"

    with open(os.path.join(_MODELS, "meta.json")) as f:
        meta = json.load(f)
    assert meta.get("market_latest"), "meta.json missing market_latest"

    from ml.predict import ArtPricePredictor
    p = ArtPricePredictor(models_dir=_MODELS)
    cols = p.feature_cols

    # market_index_3m default must equal the persisted latest, not g_med_hammer
    X, *_ = p._fvec(artist="__no_such_artist__", estimate_low=50_000,
                    estimate_high=70_000, medium="oil on canvas",
                    house="christies", sale_year=2025, sale_month=9)
    mi = float(X[0][cols.index("market_index_3m")])
    assert abs(mi - float(meta["market_latest"]["market_index_3m"])) < 1e-3, (
        f"market_index_3m not wired to persisted latest: {mi}")

    # An artist with non-zero persisted momentum must surface it at inference
    nz = stats[stats["artist_momentum_6m"].fillna(0) != 0]
    if len(nz):
        artist = str(nz.iloc[0]["artist"])
        Xa, *_ = p._fvec(artist=artist, estimate_low=50_000, estimate_high=70_000,
                         medium="oil on canvas", house="christies",
                         sale_year=2025, sale_month=9)
        assert float(Xa[0][cols.index("artist_momentum_6m")]) != 0.0, (
            "artist_momentum_6m still hardcoded to 0.0 at inference")

    # aspect_ratio parity: training fills missing dims as 0.0 (prep fillna);
    # inference must emit the same marker, not 1.0
    Xn, *_ = p._fvec(artist="__no_such_artist__", estimate_low=50_000,
                     estimate_high=70_000, medium="oil on canvas",
                     house="christies", sale_year=2025, sale_month=9)
    assert float(Xn[0][cols.index("aspect_ratio")]) == 0.0, (
        "aspect_ratio for missing dims must be 0.0 to match training fillna(0)")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(_MODELS, "lgbm.txt")):
        print("⏭  skipped — no trained model present")
    else:
        test_predict_returns_expected_keys()
        test_predicted_hammer_positive_and_finite()
        test_conformal_interval_is_ordered_and_positive()
        test_inference_uses_persisted_momentum_and_market()
        print("✅ inference contract + train/inference parity hold")
