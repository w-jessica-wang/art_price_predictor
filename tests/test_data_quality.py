# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Data-quality tests — this is a FINE-ART dataset, so wine/spirits/collectible lots
(which the auction houses also sell, with the producer/vintage parsed as the
"artist") must be excluded. Proves the exclusion both at the cleaner level
(is_junk / is_wine_or_spirit) and at the runtime guard in load_data().
"""
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_data import is_wine_or_spirit, is_junk  # noqa: E402
from ml.features import load_data  # noqa: E402


def test_wine_and_spirits_are_flagged():
    for a in ["Château Latour 1863", "Les Forts de Latour 1966", "Mixed Bordeaux",
              "Domaine d'Eugénie, Montrachet 2014", "Karuizawa 12 Year Old Whisky",
              "Hennessy 'Richard Hennessy' Cognac", "Krug Rosé Mixed Releases"]:
        assert is_wine_or_spirit(a), f"should flag wine/spirit: {a}"


def test_real_painters_are_spared():
    # The discriminator (vintage year / 'Mixed') must NOT catch real artists,
    # especially Henri Fantin-Latour (shares 'latour' with Château Latour).
    for a in ["Henri Fantin-Latour", "Victoria Dubourg Fantin-Latour", "Pablo Picasso",
              "Andy Warhol", "Gerhard Richter", "LATOUR, Alain de"]:
        assert not is_wine_or_spirit(a), f"should NOT flag real artist: {a}"


def test_is_junk_rejects_wine():
    row = {"artist": "Château Latour 1982", "estimate_low": 5000, "estimate_high": 8000}
    junk, reason = is_junk(row)
    assert junk and "wine" in reason.lower()


def test_load_data_drops_wine_rows():
    rows = [
        {"house": "christies", "auction_id": "A", "lot_number": 1, "artist": "Andy Warhol",
         "title": "Flowers", "year_made": 1970, "medium": "screenprint", "width_cm": 50,
         "height_cm": 50, "provenance": "", "signed": "", "exhibited": "", "literature": "",
         "estimate_low": 50000, "estimate_high": 70000, "sold_price": 60000, "sale_date": "2024-05-01"},
        {"house": "christies", "auction_id": "A", "lot_number": 2, "artist": "Château Latour 1982",
         "title": "", "year_made": "", "medium": "wine", "width_cm": "", "height_cm": "",
         "provenance": "", "signed": "", "exhibited": "", "literature": "",
         "estimate_low": 4000, "estimate_high": 6000, "sold_price": 5000, "sale_date": "2024-05-01"},
    ]
    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        out = load_data(path)
        artists = out["artist"].tolist()
        assert "Andy Warhol" in artists
        assert not any("Latour" in a for a in artists), "wine lot leaked through load_data"
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_wine_and_spirits_are_flagged()
    test_real_painters_are_spared()
    test_is_junk_rejects_wine()
    test_load_data_drops_wine_rows()
    print("✅ data-quality: wine/spirits excluded, real painters spared")
