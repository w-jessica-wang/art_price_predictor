# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Feature engineering for art auction price prediction.
======================================================
Performance:
  - Artist features use vectorized pandas groupby + expanding() + shift(1)
    → O(n log n) instead of O(n²) row-by-row loops
  - All operations stay in numpy/pandas (no Python-level iteration over rows)
  - Results cached to parquet for instant re-runs

Key design:
  - Target: log(hammer / estimate_mid) — currency/scale agnostic premium
  - Artist features are point-in-time (shift(1) ensures no future leakage)
  - Market index features use monthly resampled medians, merged strictly
    before each lot's date (allow_exact_matches=False)
  - Artist momentum features use merge_asof with date windows — vectorized,
    no row-by-row iteration
"""

import re
import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────

MEDIUM_CATEGORIES = {
    "oil on canvas":     0,
    "oil on board":      1,
    "oil on panel":      2,
    "oil on paper":      3,
    "acrylic on canvas": 4,
    "acrylic on board":  5,
    "acrylic on panel":  6,
    "acrylic on paper":  7,
    "watercolor":        8,
    "gouache":           9,
    "ink on paper":      10,
    "graphite on paper": 11,
    "pencil on paper":   12,
    "pastel":            13,
    "charcoal on paper": 14,
    "bronze sculpture":  15,
    "marble sculpture":  16,
    "photograph":        17,
    "screenprint":       18,
    "lithograph":        19,
    "etching":           20,
    "woodcut":           21,
    "mixed media":       22,
    "collage":           23,
    "print":             24,
    "installation/video":25,
    "unknown":           26,
    # ── Consolidation of high-frequency near-miss mediums (were falling into
    #    "unknown"=26, lumping materially different works together as noise) ──
    "ink on canvas":                          10,  # → ink on paper (same medium, different support)
    "acrylic on linen":                        4,  # → acrylic on canvas (linen = canvas variant)
    "acrylic and spray paint on canvas":       4,
    "oil, acrylic and spray paint on canvas":  4,
    "spray paint on canvas":                   4,
    "household gloss on canvas":               4,  # contemporary (e.g. Hirst spot paintings)
    "pigment on linen":                        4,
    "oil on masonite":                         1,  # → oil on board (masonite = engineered hardboard)
    "oil on wood":                             2,  # → oil on panel
    "acrylic on wood":                         6,  # → acrylic on panel
    "butterflies and household gloss on canvas": 22,  # → mixed media
    "oil and acrylic on linen":                4,
    "acrylic and spray paint on canvas, in artist's frame": 4,
    "enamel on canvas":                        4,  # synthetic paint, closer to acrylic mkt than oil
    "oil on velvet":                           0,  # → oil on canvas (oil paint, alt. support)
    "marker on paper":                         10, # → ink on paper
}

HOUSE_MAP = {"christies": 0, "phillips": 1, "sothebys": 2, "bonhams": 3}
SMOOTH_K  = 10   # Bayesian shrinkage weight toward global mean

# ── Artist nationality / region lookup ────────────────────────────────────────
# Regions: 0=unknown  1=western_europe  2=north_america  3=east_asia
#          4=latin_america  5=eastern_europe  6=middle_east_africa
#          7=south_asia  8=oceania_other
ARTIST_NATIONALITY: dict[str, int] = {
    "Pablo Picasso": 1, "Andy Warhol": 2, "David Hockney": 1,
    "Marc Chagall": 5, "Damien Hirst": 1, "Man Ray": 2,
    "Joan Miró": 1, "Line Vautrin": 1, "Yayoi Kusama": 3,
    "Henri Matisse": 1, "Utagawa Hiroshige": 3, "Tiffany Studios": 2,
    "Rembrandt Harmensz. Van Rijn": 1, "Roy Lichtenstein": 2,
    "Zhang Daqian": 3, "François-Xavier Lalanne": 1, "Yoshitomo Nara": 3,
    "Alexander Calder": 2, "Jean Dubuffet": 1, "Ed Ruscha": 2,
    "George Condo": 2, "Gerhard Richter": 1, "Takashi Murakami": 3,
    "Pu Ru": 3, "Kaws": 2, "Katsushika Hokusai": 3,
    "Pierre-Auguste Renoir": 1, "Alex Katz": 2, "Irving Penn": 2,
    "Claude Lalanne": 1, "Fernando Botero": 4, "Jasper Johns": 2,
    "Max Ernst": 1, "Keith Haring": 2, "Zao Wou-Ki": 3,
    "Alberto Giacometti": 1, "Helmut Newton": 1, "Frank Stella": 2,
    "Jean Royère": 1, "Tom Wesselmann": 2, "Salvador Dalí": 1,
    "Robert Rauschenberg": 2, "Henry Moore": 1, "Francis Newton Souza": 7,
    "Charlotte Perriand": 1, "Richard Prince": 2, "Bernard Buffet": 1,
    "Jean Prouvé": 1, "Edgar Degas": 1, "Lin Fengmian": 3,
    "Sam Francis": 2, "Cindy Sherman": 2, "Hiroshi Sugimoto": 3,
    "Qi Baishi": 3, "Egon Schiele": 1, "Joan Mitchell": 2,
    "Jean-Michel Basquiat": 2, "Jenny Holzer": 2, "Louise Bourgeois": 1,
    "Georg Baselitz": 1, "Lucian Freud": 1, "Bruce Nauman": 2,
    "Felix Vallotton": 1, "Gustave Courbet": 1, "Banksy": 1,
    "Christopher Wool": 2, "Cy Twombly": 2, "Brice Marden": 2,
    "Wade Guyton": 2, "Henri de Toulouse-Lautrec": 1,
    "Kerry James Marshall": 2, "Wolfgang Tillmans": 1, "Cecily Brown": 1,
    "Julian Opie": 1, "Fernand Léger": 1, "Henri Fantin-Latour": 1,
    "René Magritte": 1, "Elizabeth Peyton": 2, "Günther Förg": 1,
    "George Stubbs": 1, "Henri-Georges Adam": 1, "Paul Signac": 1,
    "Pierre Bonnard": 1, "James Turrell": 2, "Bridget Riley": 1,
    "Frida Kahlo": 4, "Jean Fautrier": 1, "Martial Raysse": 1,
    "George Bellows": 2, "Edward Weston": 2, "Berthe Morisot": 1,
    "Pierre Klossowski": 1, "Henri Cartier-Bresson": 1, "Glenn Ligon": 2,
    "Lee Ufan": 3, "Luc Tuymans": 1, "Victor Vasarely": 1,
    "Dieter Roth": 1, "Martin Kippenberger": 1,
    # East Asian
    "Ai Weiwei": 3, "Liu Ye": 3, "Sanyu": 3, "Pan Yuliang": 3,
    "Wu Guanzhong": 3, "Xu Beihong": 3, "Chen Yifei": 3,
    "Yue Minjun": 3, "Zeng Fanzhi": 3, "Fan Kuan": 3,
    "On Kawara": 3, "Yoko Ono": 3,
    # Latin American
    "Wifredo Lam": 4, "Diego Rivera": 4, "Rufino Tamayo": 4,
    "Tarsila do Amaral": 4, "Leonora Carrington": 1,
    # Eastern European
    "Chaim Soutine": 5, "Wassily Kandinsky": 5, "Kazimir Malevich": 5,
    "El Lissitzky": 5, "Natalia Goncharova": 5,
    # Western European (additional)
    "Emil Nolde": 1, "Ernst Ludwig Kirchner": 1, "Amedeo Modigliani": 1,
    "Oskar Kokoschka": 1, "Gustav Klimt": 1, "Paul Klee": 1,
    # North American (additional)
    "Kehinde Wiley": 2, "Kara Walker": 2, "Mark Rothko": 2,
    "Franz Kline": 2, "Willem de Kooning": 2, "Jackson Pollock": 2,
    "Lee Krasner": 2, "Helen Frankenthaler": 2, "Ellsworth Kelly": 2,
    "Donald Judd": 2, "Dan Flavin": 2, "Richard Serra": 2,
    "Carl Andre": 2, "Sol LeWitt": 2, "Lawrence Weiner": 2,
    # South Asian
    "Francis Newton Souza": 7,
    # ── Expansion batch (high-frequency unknowns) ─────────────────────────────
    # North American
    "Ansel Adams": 2, "Robert Mapplethorpe": 2, "Jeff Koons": 2,
    "William Eggleston": 2, "Wayne Thiebaud": 2, "Robert Longo": 2,
    "Robert Motherwell": 2, "Josef Albers": 2, "James Rosenquist": 2,
    "Jim Dine": 2, "Jonas Wood": 2, "Raymond Pettibon": 2,
    "Rashid Johnson": 2, "Richard Diebenkorn": 2, "Kenny Scharf": 2,
    "Robert Indiana": 2, "Diane Arbus": 2, "John Baldessari": 2,
    "Shara Hughes": 2, "Louise Nevelson": 2, "George Grosz": 2,
    # Western European
    "Alighiero Boetti": 1, "Raoul Dufy": 1, "Albrecht Dürer": 1,
    "Laurence Stephen Lowry": 1, "Diego Giacometti": 1, "Auguste Rodin": 1,
    "Georges Jouve": 1, "Christo": 1, "Jean Dunand": 1, "Tracey Emin": 1,
    "Henry Moore, O.M.": 1, "Lynn Chadwick": 1, "Georges Braque": 1,
    "Eugène Boudin": 1, "Maurice Utrillo": 1, "Pierre Chareau": 1,
    "Louis Valtat": 1, "Camille Pissarro": 1, "Gio Ponti": 1,
    "Lucio Fontana": 1, "David Shrigley": 1, "Antony Gormley": 1,
    "Albert Marquet": 1, "Nicolas Party": 1, "Émile Gallé": 1,
    "Francis Picabia": 1, "Howard Hodgkin": 1, "Clovis Trouille": 1,
    "Hans Hartung": 1, "Henri De Toulouse-Lautrec": 1,  # capital-D variant
    # East Asian
    "Kawase Hasui": 3, "Chu Teh-Chun": 3, "Ding Yanyong": 3,
    # South Asian (additional)
    "Maqbool Fida Husain": 7, "Jamini Roy": 7,
    # Eastern European
    "Victor Brauner": 5,
    # Southeast Asian / other (region 8 = oceania/other)
    "Le Pho": 8, "Vu Cao Dam": 8,
}


# ── data loading ───────────────────────────────────────────────────────────

def load_data(path: str = "data/art_auction_clean.csv", train_end=None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Runtime guard: this is a FINE-ART model — drop any wine/spirits/collectible
    # lots that leaked through cleaning (producer/vintage parsed as the artist).
    # Shares one regex with clean_data so the rule can't drift. Vectorized.
    try:
        from clean_data import _WINE_STRONG, _WINE_WEAK, _VINTAGE_YEAR
        _a = df["artist"].fillna("").astype(str)
        _med = df["medium"].fillna("").astype(str) if "medium" in df.columns else ""
        _wine = (_a.str.contains(_WINE_STRONG)
                 | (_a.str.contains(_WINE_WEAK) & (_a.str.contains(_VINTAGE_YEAR) | _a.str.startswith("Mixed")))
                 | (_med.str.contains(r"(?i)\bwine\b|\bspirits?\b") if "medium" in df.columns else False))
        if _wine.any():
            df = df[~_wine].reset_index(drop=True)
    except Exception:
        pass

    df["sale_date"]    = pd.to_datetime(df["sale_date"], errors="coerce")
    df["sale_year"]    = df["sale_date"].dt.year.astype("Int64")
    df["sale_month"]   = df["sale_date"].dt.month.astype("Int64")
    df["sale_quarter"] = df["sale_date"].dt.quarter.astype("Int64")

    for col in ["estimate_low", "estimate_high", "sold_price",
                "width_cm", "height_cm", "year_made"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["signed", "exhibited", "literature"]:
        df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)

    df["estimate_mid"]     = (df["estimate_low"] + df["estimate_high"]) / 2
    df["log_estimate_mid"] = np.log1p(df["estimate_mid"])
    df["estimate_range_pct"] = (
        (df["estimate_high"] - df["estimate_low"]) / df["estimate_low"].clip(lower=1)
    )

    df["area_cm2"]    = df["width_cm"] * df["height_cm"]
    df["log_area"]    = np.log1p(df["area_cm2"])
    df["log_width"]   = np.log1p(df["width_cm"])
    df["log_height"]  = np.log1p(df["height_cm"])
    df["has_dims"]    = (df["width_cm"].notna() & df["height_cm"].notna()).astype(int)
    df["aspect_ratio"]= (df["width_cm"] / df["height_cm"].clip(lower=0.01)).clip(0.1, 10)

    # ── Medium imputation (41 % of rows have missing medium) ─────────────────
    # For each artist, fill a missing medium with their most common medium. Medium
    # is a fixed property of the artwork, but to keep the pipeline strictly
    # leakage-free the mode is computed only from lots strictly before train_end
    # (when given) — so future rows never inform an earlier lot's imputed medium.
    _mode_src = df[df["medium"].notna()]
    if train_end is not None:
        _mode_src = _mode_src[_mode_src["sale_date"] < pd.Timestamp(train_end)]
    if len(_mode_src) > 0:
        _artist_mode = (
            _mode_src.groupby("artist")["medium"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        )
        df["medium"] = df["medium"].fillna(df["artist"].map(_artist_mode))

    med = df["medium"].str.lower().str.strip()
    df["medium_cat"] = med.map(MEDIUM_CATEGORIES).fillna(26).astype(int)

    df["house_cat"] = df["house"].str.lower().map(HOUSE_MAP).fillna(0).astype(int)

    df["provenance"]         = df["provenance"].fillna("").astype(str)
    df["provenance_len"]     = df["provenance"].str.len().clip(upper=500)
    df["has_provenance"]     = (df["provenance_len"] > 30).astype(int)
    df["prov_has_museum"]    = df["provenance"].str.contains(
        r"museum|gallery|foundation|moma|tate|guggenheim|whitney|mca",
        case=False, regex=True, na=False).astype(int)
    df["prov_auction_count"] = df["provenance"].str.count(
        r"(?:christie|sotheby|phillips|bonham|auction)", re.IGNORECASE)
    # "Private collection" mention (15.1% coverage): correlates with LOWER realized
    # ratios (median log_ratio 0.182 vs 0.231, a -4.9pp signal) — likely reflects
    # specialist caution when a work lacks a public/institutional exhibition history.
    df["prov_private_collection"] = df["provenance"].str.contains(
        r"private collection|private collector", case=False, regex=True, na=False
    ).astype(int)

    df["artwork_age"] = (df["sale_year"] - df["year_made"]).clip(0, 500)

    df["sold"]       = df["sold_price"].notna().astype(int)
    df["log_ratio"]  = np.log(df["sold_price"] / df["estimate_mid"].clip(lower=1))
    df["log_hammer"] = np.log1p(df["sold_price"])

    # ── New features ──────────────────────────────────────────────────────
    # title_is_untitled: artworks titled "Untitled" sell differently
    df["title_is_untitled"] = (df["title"].str.lower()
                               .str.contains("untitled", na=False)
                               .astype(int))

    # estimate_in_millions: flag for high-value lots (different dynamics)
    df["estimate_in_millions"] = (df["estimate_mid"] >= 1_000_000).astype(int)

    # log_lot_number: lot position in the sale (early/late lots sell differently)
    df["lot_number"]    = pd.to_numeric(df.get("lot_number", pd.Series(dtype=float)),
                                        errors="coerce").fillna(0)
    df["log_lot_number"] = np.log1p(df["lot_number"])

    # ── Cyclical month encoding ───────────────────────────────────────────
    # Raw integer month (1–12) treats Jan and Dec as maximally distant.
    # Sin/cos encoding wraps correctly so they are adjacent.
    _month = df["sale_month"].fillna(6).astype(float)
    df["sale_month_sin"] = np.sin(2 * np.pi * _month / 12)
    df["sale_month_cos"] = np.cos(2 * np.pi * _month / 12)

    # ── Estimate spread in log space ──────────────────────────────────────
    # Complements estimate_range_pct (relative %) with an absolute signal.
    df["log_estimate_spread"] = np.log1p(
        (df["estimate_high"] - df["estimate_low"]).clip(lower=0)
    )

    # ── Estimate round-number flag ────────────────────────────────────────
    # Specialists give round estimates ($1M, $500k) when less certain;
    # precise estimates ($987,500) signal stronger private-sale conviction.
    df["estimate_is_round"] = (
        (df["estimate_low"].fillna(0) % 10_000 == 0) &
        (df["estimate_low"] > 0)
    ).astype(int)

    # ── Title documentation depth ─────────────────────────────────────────
    # Longer, more descriptive titles → better-documented / significant works.
    df["log_title_length"] = np.log1p(
        df["title"].str.len().fillna(0).clip(upper=300)
    )

    # ── Estimate tightness ────────────────────────────────────────────────
    # Tight estimate ($950k–$1M) = specialist confidence;
    # wide estimate ($200k–$800k) = uncertainty.
    df["estimate_tightness"] = (1.0 - (
        (df["estimate_high"] - df["estimate_low"]) /
        df["estimate_mid"].clip(lower=1)
    )).clip(-1, 1)

    # ── Evening sale flag ─────────────────────────────────────────────────
    # Evening sales (high median estimate, few lots) price 20-40% above comparable
    # day-sale works. Computed from per-auction_id aggregate statistics — the full
    # sale catalog is publicly announced before any lot sells, so this is NOT leaky.
    # Thresholds: ≤80 lots AND median estimate ≥$500k → evening/prestige sale.
    # At inference time, pass is_evening_sale=1 explicitly for evening/prestige sales.
    _auc_stats = (
        df.groupby("auction_id")
        .agg(_auc_n=("auction_id", "count"), _auc_med=("estimate_mid", "median"))
    )
    df = df.join(_auc_stats, on="auction_id")
    df["is_evening_sale"] = (
        (df["_auc_n"] <= 80) & (df["_auc_med"] >= 500_000)
    ).astype(int)
    # Online-only sales: 300+ lots, low median estimate → different bidder pool
    # and different hammer dynamics from physical day/evening sales.
    df["is_online_sale"] = (df["_auc_n"] >= 300).astype(int)
    # Normalized lot position within the sale (0 = first lot, 1 = last lot).
    # Orthogonal to log_lot_number: lot 30/60 (evening sale) vs 30/500 (online sale)
    # have the same log_lot_number but very different relative placement.
    df["lot_position_pct"] = (
        df["lot_number"] / df["_auc_n"].clip(lower=1)
    ).clip(0, 1).fillna(0.5)
    df.drop(columns=["_auc_n", "_auc_med"], inplace=True)

    # ── Artist nationality / region ───────────────────────────────────────────
    # Lookup in hand-curated dict; unknown artists default to 0.
    # Regions: 0=unknown  1=W.Europe  2=N.America  3=E.Asia
    #          4=Latin Am  5=E.Europe  6=ME/Africa  7=S.Asia  8=other
    df["artist_nationality_region"] = (
        df["artist"].map(ARTIST_NATIONALITY).fillna(0).astype(int)
    )

    # ── Artist × medium key ───────────────────────────────────────────────
    # Stored for CatBoost native categorical encoding — Picasso oils ≠ ceramics.
    # Not a numeric feature (excluded from FEATURE_COLS).
    df["artist_medium"] = (df["artist"].astype(str) + " | " +
                           df["medium"].fillna("unknown").astype(str))

    return df


# ── artist features — vectorized O(n log n) ────────────────────────────────

def compute_artist_features(df: pd.DataFrame, train_end=None) -> pd.DataFrame:
    """
    Compute point-in-time artist features using merge_asof.

    `train_end` (optional): if given, the global NaN-fallback constants are
    computed only from sold lots STRICTLY BEFORE this date, so validation/test
    outcomes never influence the fallbacks used for earlier observations.
    None → use all data (correct for the deploy artifact, which has no held-out
    test set).

    Strategy:
      1. Build a "cumulative stats as-of" table from sold lots only —
         for each (artist, date), stats include all sold lots by that artist
         UP TO AND INCLUDING that date.
      2. merge_asof with allow_exact_matches=False joins each lot in df
         to the most recent stats snapshot STRICTLY BEFORE the lot's date.
         → Guarantees zero temporal leakage even for same-auction lots.

    Also computes:
      - Market index features (market_index_3m, market_index_12m, market_momentum)
        using monthly resampled medians across all artists, merged strictly
        before each lot's date — macro signal for overall market health.
      - Artist momentum features (artist_momentum_6m, artist_momentum_12m):
        the point-in-time CHANGE in the artist's expanding median log_ratio vs
        ~6m/12m earlier (a trend proxy, not a windowed median — see
        _compute_artist_momentum). Replaces the stale artist_trend_12m_ratio.

    O(n log n): dominated by sort + merge_asof.
    """
    import time
    t0 = time.time()

    df = df.sort_values("sale_date").reset_index(drop=True)

    # ── Global stats for NaN fallback (TRAIN-ONLY when train_end given) ───
    # Constants used when an artist has no prior history. Restricting to lots
    # strictly before train_end stops validation/test data from leaking into the
    # fallbacks used for earlier observations.
    sold_all = df[df["sold"] == 1]
    if train_end is not None:
        _te = pd.Timestamp(train_end)
        _train_df = df[df["sale_date"] < _te]
        _glob_src = _train_df[_train_df["sold"] == 1]
        _n_total  = max(len(_train_df), 1)
        if len(_glob_src) == 0:                   # degenerate (e.g. tiny test data)
            _glob_src, _n_total = sold_all, max(len(df), 1)
    else:
        _glob_src, _n_total = sold_all, max(len(df), 1)
    global_med_ratio  = float(_glob_src["log_ratio"].median())
    global_med_hammer = float(_glob_src["log_hammer"].median())
    global_std_hammer = float(_glob_src["log_hammer"].std())
    global_sell_thru  = float(len(_glob_src) / _n_total)

    # ── Build cumulative stats per (artist, date) ─────────────────────────
    # Sort sold lots by (artist, sale_date) — within artist chronologically
    sold = (df[df["sold"] == 1][["artist", "sale_date", "log_ratio", "log_hammer"]]
            .sort_values(["artist", "sale_date"])
            .reset_index(drop=True))

    grp = sold.groupby("artist", sort=False)

    # Inclusive expanding stats (include current row)
    sold["_n_incl"]        = grp.cumcount() + 1
    sold["_med_ratio_incl"]= grp["log_ratio"].expanding().median().reset_index(level=0, drop=True)
    sold["_med_ham_incl"]  = grp["log_hammer"].expanding().median().reset_index(level=0, drop=True)
    sold["_std_ham_incl"]  = grp["log_hammer"].expanding().std().reset_index(level=0, drop=True)
    sold["_last_ratio"]    = grp["log_ratio"].shift(1)   # previous sale ratio

    # Keep one snapshot per (artist, date) — take the LAST row per group
    # (= most up-to-date stats for that date)
    snap = (sold.sort_values(["artist", "sale_date"])
                .groupby(["artist", "sale_date"], sort=False)
                .last()
                .reset_index())

    # ── Total lots (sold + unsold) per (artist, date) snapshot ───────────
    all_lots = (df[["artist", "sale_date"]]
                .sort_values(["artist", "sale_date"])
                .reset_index(drop=True))
    all_lots["_total_incl"] = all_lots.groupby("artist", sort=False).cumcount() + 1
    all_snap = (all_lots.groupby(["artist", "sale_date"], sort=False)
                         .last()
                         .reset_index())

    snap = snap.merge(all_snap, on=["artist", "sale_date"], how="left")

    # ── merge_asof: join each lot to the snapshot strictly BEFORE its date ─
    # Must sort both left and right by the join key (sale_date)
    df_sorted  = df.sort_values("sale_date").reset_index(drop=True)
    snap_sorted = snap.sort_values("sale_date").reset_index(drop=True)

    merged = pd.merge_asof(
        df_sorted,
        snap_sorted[["artist", "sale_date", "_n_incl", "_med_ratio_incl",
                     "_med_ham_incl", "_std_ham_incl", "_last_ratio", "_total_incl"]],
        on="sale_date",
        by="artist",
        direction="backward",
        allow_exact_matches=False,   # STRICTLY before — no same-date leakage
        suffixes=("", "_snap"),
    )

    # ── Derive final feature columns ──────────────────────────────────────
    n     = merged["_n_incl"].fillna(0)
    total = merged["_total_incl"].fillna(0)

    merged["artist_n_past_sales"]      = n
    merged["artist_has_history"]       = (n >= 5).astype(int)
    merged["artist_sell_through_rate"] = (n / total.clip(lower=1)).fillna(global_sell_thru)
    merged["artist_median_log_hammer"] = merged["_med_ham_incl"].fillna(global_med_hammer)
    merged["artist_median_ratio"]      = merged["_med_ratio_incl"].fillna(global_med_ratio)
    merged["artist_log_price_std"]     = merged["_std_ham_incl"].fillna(global_std_hammer)
    merged["artist_last_ratio"]        = merged["_last_ratio"].fillna(global_med_ratio)
    # artist_trend_12m_ratio kept for backward compat — now replaced by momentum features
    merged["artist_trend_12m_ratio"]   = merged["_last_ratio"].fillna(global_med_ratio)

    w = n / (n + SMOOTH_K)
    merged["artist_smoothed_ratio"] = (
        w * merged["_med_ratio_incl"].fillna(global_med_ratio) +
        (1 - w) * global_med_ratio
    )

    # ── artist_prestige_score: career-stage proxy ─────────────────────────
    # log(n_past_sales + 1) × artist_median_log_hammer.
    # captures: Picasso with 1000 sales at $500k >> unknown artist with 3 sales at $5k
    # NOT normalised: tree models are scale-invariant, so dividing by the dataset-
    # wide max (a) leaked future data into the constant and (b) created a train/
    # inference scale mismatch (inference used the raw value). Using raw fixes both.
    merged["artist_prestige_score"] = (
        np.log1p(n) * merged["_med_ham_incl"].fillna(global_med_hammer)
    )

    df = merged.drop(columns=["_n_incl", "_med_ratio_incl", "_med_ham_incl",
                               "_std_ham_incl", "_last_ratio", "_total_incl"],
                     errors="ignore")

    # Restore original sort order
    df = df.sort_values("sale_date").reset_index(drop=True)

    # ── Market index features ─────────────────────────────────────────────
    # Build a monthly median log_hammer across ALL artists (sold lots only)
    # Then merge each lot to the market index strictly before its date.
    # This gives a macro signal for overall auction market health.
    df = _compute_market_index(df, global_med_hammer)

    # ── Artist momentum features ──────────────────────────────────────────
    # Cumulative-median CHANGE over ~6m/12m (a leakage-safe trend proxy, NOT a
    # trailing-window median — see _compute_artist_momentum docstring).
    df = _compute_artist_momentum(df, global_med_ratio)

    # ── Momentum acceleration ─────────────────────────────────────────────
    # Positive = improving short-term vs long-term; negative = decelerating.
    # Set to 0.0 at inference time (neutral fallback).
    df["artist_momentum_accel"] = (
        df["artist_momentum_6m"] - df["artist_momentum_12m"]
    )

    # ── Lot vs artist median ──────────────────────────────────────────────
    # Is this lot expensive or cheap for this artist?
    # Primary signal under log_ratio target: positive = above-median offering.
    df["lot_vs_artist_median"] = (
        df["log_estimate_mid"] -
        df["artist_median_log_hammer"].fillna(df["log_estimate_mid"])
    )

    # ── Days since artist's last sold lot ─────────────────────────────────
    df = _compute_days_since_last_sale(df)

    # ── Rolling 12-month sell-through rate per artist ─────────────────────
    df = _compute_rolling_sellthru(df, global_sell_thru)

    # ── Artist × house median ratio ────────────────────────────────────────
    # Point-in-time median log_ratio per (artist, house) pair.
    # Captures house-specific track records LGBM can't see via interactions.
    df = _compute_artist_house_ratio(df, global_med_ratio)

    # ── External macro features (S&P 500) ──────────────────────────────────
    # Collector wealth signal orthogonal to internal art market index.
    # Falls back to neutral constants if macro_sp500.parquet not found.
    df = _compute_sp500_features(df)

    # Sanity check
    filled = (df["artist_n_past_sales"] > 0).sum()
    elapsed = time.time() - t0
    print(f"  Artist features done in {elapsed:.1f}s  "
          f"(artists={df['artist'].nunique():,}  "
          f"rows_with_history={filled:,}/{len(df):,})")
    return df


def _compute_market_index(df: pd.DataFrame, global_med_hammer: float) -> pd.DataFrame:
    """
    Compute rolling market index features from monthly median log_hammer
    across all sold lots.

    market_index_3m  — rolling 3-month median log_hammer (all artists)
    market_index_12m — rolling 12-month median log_hammer (all artists)
    market_momentum  — market_index_3m / market_index_12m  (acceleration)

    Strictly point-in-time: merge_asof with allow_exact_matches=False so each
    lot only sees the market state as of the PREVIOUS month boundary.
    """
    sold = df[df["sold"] == 1][["sale_date", "log_hammer"]].copy()
    sold = sold.dropna(subset=["sale_date", "log_hammer"])

    # Resample to monthly periods — use period start as the timestamp
    sold["month"] = sold["sale_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (sold.groupby("month")["log_hammer"]
               .median()
               .reset_index()
               .rename(columns={"month": "sale_date", "log_hammer": "_mkt_med"}))
    monthly = monthly.sort_values("sale_date").reset_index(drop=True)

    # Rolling 3-month and 12-month windows over the monthly series
    monthly["market_index_3m"]  = (monthly["_mkt_med"]
                                   .rolling(window=3,  min_periods=1).median())
    monthly["market_index_12m"] = (monthly["_mkt_med"]
                                   .rolling(window=12, min_periods=1).median())
    monthly = monthly.drop(columns=["_mkt_med"])

    # LEAKAGE FIX: a month's median is computed from the WHOLE month, so it must
    # not be visible until that month has ended. Shift each month's stats forward
    # to the start of the NEXT month — then a lot in month M only ever sees stats
    # built from months strictly before M. (Previously, month-start timestamps let
    # a mid-month lot merge_asof onto its own month's full-month median.)
    monthly["sale_date"] = monthly["sale_date"] + pd.DateOffset(months=1)

    # Merge strictly before each lot's date — use the monthly grid
    df_sorted = df.sort_values("sale_date").reset_index(drop=True)
    df_sorted = pd.merge_asof(
        df_sorted,
        monthly[["sale_date", "market_index_3m", "market_index_12m"]],
        on="sale_date",
        direction="backward",
        allow_exact_matches=False,   # strictly before
    )

    # Fill NaN (lots before any market data) with global fallback
    df_sorted["market_index_3m"]  = df_sorted["market_index_3m"].fillna(global_med_hammer)
    df_sorted["market_index_12m"] = df_sorted["market_index_12m"].fillna(global_med_hammer)

    # Momentum: ratio of short-term to long-term index
    denom = df_sorted["market_index_12m"].clip(lower=1e-6)
    df_sorted["market_momentum"] = (df_sorted["market_index_3m"] / denom).clip(0.5, 2.0)

    return df_sorted.sort_values("sale_date").reset_index(drop=True)


def _compute_sp500_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge S&P 500 monthly macro features into each lot.

    Features
    --------
    sp500_ret_3m  — S&P 500 trailing 3-month return  (collector wealth signal)
    sp500_ret_12m — S&P 500 trailing 12-month return (longer-term market health)
    sp500_level   — Normalised S&P 500 close [0–1] over 2018–2026

    Source: data/macro_sp500.parquet (monthly end-of-month closes via yfinance).
    Merged with allow_exact_matches=True so a lot on a month-end date gets that
    month's level (publicly known data, no future leakage).
    Neutral fallbacks (0.02 / 0.10 / 0.50) used if parquet not found.
    """
    import os
    macro_path = os.path.join(os.path.dirname(__file__), "..", "data", "macro_sp500.parquet")
    if not os.path.exists(macro_path):
        df["sp500_ret_3m"]  = 0.02
        df["sp500_ret_12m"] = 0.10
        df["sp500_level"]   = 0.50
        return df

    sp = pd.read_parquet(macro_path).reset_index()
    sp = sp.rename(columns={"Date": "sale_date"})
    sp["sale_date"] = pd.to_datetime(sp["sale_date"])
    sp = sp.sort_values("sale_date").reset_index(drop=True)

    df_sorted = df.sort_values("sale_date").reset_index(drop=True)
    df_sorted = pd.merge_asof(
        df_sorted,
        sp[["sale_date", "sp500_ret_3m", "sp500_ret_12m", "sp500_level"]],
        on="sale_date",
        direction="backward",   # most recent complete month-end at or before sale date
        allow_exact_matches=True,
    )
    df_sorted["sp500_ret_3m"]  = df_sorted["sp500_ret_3m"].fillna(0.02)
    df_sorted["sp500_ret_12m"] = df_sorted["sp500_ret_12m"].fillna(0.10)
    df_sorted["sp500_level"]   = df_sorted["sp500_level"].fillna(0.50)

    return df_sorted.sort_values("sale_date").reset_index(drop=True)


def _compute_artist_momentum(df: pd.DataFrame, global_med_ratio: float) -> pd.DataFrame:
    """
    Compute artist momentum features — vectorized via merge_asof.

    artist_momentum_6m  — CHANGE in this artist's expanding (cumulative, point-in-time)
                          median log_ratio between just-before the lot's date and its
                          state ~6 months earlier.
    artist_momentum_12m — same, vs. ~12 months earlier.

    ⚠️ NOTE ON DEFINITION: these are *cumulative-median changes*, NOT trailing-window
    medians. A trailing-window median (median of sales within [date-6m, date)) is a
    different quantity and cannot be recovered by differencing two cumulative medians —
    e.g. lifetime values [0,0,0,10] give cumulative-median-now=0 and 6m-ago=0 → momentum 0,
    while the true recent-window median could be 10. We use the cumulative-median change
    deliberately as a cheap, leakage-safe trend/momentum *proxy* (it is one of the
    lowest-importance features). The name "momentum" denotes this directional change,
    not a windowed statistic.

    Strategy: build a per-artist expanding median snapshot at each sale date, then two
    backward merge_asof calls (as-of the lot's date, and as-of date − window) and take
    the difference — all strictly point-in-time.
    """
    sold = (df[df["sold"] == 1][["artist", "sale_date", "log_ratio"]]
            .dropna(subset=["sale_date", "log_ratio"])
            .sort_values(["artist", "sale_date"])
            .reset_index(drop=True))

    # For each artist, compute a snapshot of expanding median at each sale date
    # We'll use two merge_asof calls with date offsets to get window medians
    #
    # Build cumulative snapshot: for each (artist, sale_date), store all
    # log_ratio values up to and including that date
    grp = sold.groupby("artist", sort=False)
    sold["_cummed"] = grp["log_ratio"].expanding().median().reset_index(level=0, drop=True)
    sold["_cumcnt"] = grp.cumcount() + 1

    # One snapshot per (artist, date) — last row = most recent cumulative state
    snap = (sold.sort_values(["artist", "sale_date"])
                .groupby(["artist", "sale_date"], sort=False)
                .last()[["_cummed", "_cumcnt"]]
                .reset_index())

    # ── 6-month change ────────────────────────────────────────────────────
    # For each lot, find what this artist's cumulative (expanding) median was ~6
    # months ago by merging on (artist, date - 6m); momentum = current cumulative
    # median − that. This is a cumulative-median CHANGE, not a trailing-window median.
    df_sorted = df.sort_values("sale_date").reset_index(drop=True)

    # Current state (strictly before lot date)
    snap_sorted = snap.sort_values("sale_date").reset_index(drop=True)
    merged_now = pd.merge_asof(
        df_sorted,
        snap_sorted.rename(columns={"_cummed": "_med_now", "_cumcnt": "_cnt_now"}),
        on="sale_date",
        by="artist",
        direction="backward",
        allow_exact_matches=False,
    )

    # State 6 months ago (strictly before lot_date - 6m)
    df_lag6 = df_sorted.copy()
    df_lag6["_lag_date"] = df_lag6["sale_date"] - pd.DateOffset(months=6)
    df_lag6 = df_lag6.sort_values("_lag_date").reset_index(drop=True)

    snap_for_lag = snap_sorted.rename(columns={
        "sale_date": "_lag_date",
        "_cummed": "_med_6m_ago",
        "_cumcnt": "_cnt_6m_ago",
    })
    merged_lag6 = pd.merge_asof(
        df_lag6,
        snap_for_lag,
        on="_lag_date",
        by="artist",
        direction="backward",
        allow_exact_matches=False,
    )[["_lag_date", "artist", "_med_6m_ago", "_cnt_6m_ago"]].copy()
    # Restore original order by re-attaching via index
    merged_lag6.index = df_lag6.index

    # State 12 months ago
    df_lag12 = df_sorted.copy()
    df_lag12["_lag_date"] = df_lag12["sale_date"] - pd.DateOffset(months=12)
    df_lag12 = df_lag12.sort_values("_lag_date").reset_index(drop=True)

    snap_for_lag12 = snap_sorted.rename(columns={
        "sale_date": "_lag_date",
        "_cummed": "_med_12m_ago",
        "_cumcnt": "_cnt_12m_ago",
    })
    merged_lag12 = pd.merge_asof(
        df_lag12,
        snap_for_lag12,
        on="_lag_date",
        by="artist",
        direction="backward",
        allow_exact_matches=False,
    )[["_lag_date", "artist", "_med_12m_ago", "_cnt_12m_ago"]].copy()
    merged_lag12.index = df_lag12.index

    # Re-sort back to the original df_sorted order
    # merged_now is already in df_sorted order
    # For lag dfs we need to invert the sort — use original positional indices
    # Use the fact that df_sorted still has its RangeIndex (0..N-1)
    # We sorted df_lag by _lag_date, so we need to map back.
    # Easiest: add an _orig_idx column before sorting, then reindex.
    df_lag6_orig = df_sorted.copy()
    df_lag6_orig["_orig_idx"] = df_lag6_orig.index
    df_lag6_orig["_lag_date"] = df_lag6_orig["sale_date"] - pd.DateOffset(months=6)
    df_lag6_orig = df_lag6_orig.sort_values("_lag_date").reset_index(drop=True)

    snap_for_lag_v2 = snap_sorted.rename(columns={
        "sale_date": "_lag_date",
        "_cummed": "_med_6m_ago",
        "_cumcnt": "_cnt_6m_ago",
    })
    lag6_result = pd.merge_asof(
        df_lag6_orig[["_orig_idx", "_lag_date", "artist"]],
        snap_for_lag_v2[["_lag_date", "artist", "_med_6m_ago", "_cnt_6m_ago"]],
        on="_lag_date",
        by="artist",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("_orig_idx").sort_index()

    df_lag12_orig = df_sorted.copy()
    df_lag12_orig["_orig_idx"] = df_lag12_orig.index
    df_lag12_orig["_lag_date"] = df_lag12_orig["sale_date"] - pd.DateOffset(months=12)
    df_lag12_orig = df_lag12_orig.sort_values("_lag_date").reset_index(drop=True)

    snap_for_lag12_v2 = snap_sorted.rename(columns={
        "sale_date": "_lag_date",
        "_cummed": "_med_12m_ago",
        "_cumcnt": "_cnt_12m_ago",
    })
    lag12_result = pd.merge_asof(
        df_lag12_orig[["_orig_idx", "_lag_date", "artist"]],
        snap_for_lag12_v2[["_lag_date", "artist", "_med_12m_ago", "_cnt_12m_ago"]],
        on="_lag_date",
        by="artist",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("_orig_idx").sort_index()

    # Attach momentum columns
    # artist_momentum_6m: current expanding median − state 6m ago
    # (if counts haven't changed = no sales in window → use overall median)
    result = merged_now.copy()
    result["_med_6m_ago"]  = lag6_result["_med_6m_ago"].values
    result["_cnt_6m_ago"]  = lag6_result["_cnt_6m_ago"].fillna(0).values
    result["_med_12m_ago"] = lag12_result["_med_12m_ago"].values
    result["_cnt_12m_ago"] = lag12_result["_cnt_12m_ago"].fillna(0).values

    # If no new sales in the window → momentum = current median (neutral)
    has_6m_activity  = result["_cnt_now"].fillna(0) > result["_cnt_6m_ago"].fillna(0)
    has_12m_activity = result["_cnt_now"].fillna(0) > result["_cnt_12m_ago"].fillna(0)

    # Use current expanding median as the "recent" signal;
    # fallback to global if artist has no history
    med_now  = result["_med_now"].fillna(global_med_ratio)
    med_6ago = result["_med_6m_ago"].fillna(global_med_ratio)
    med_12ago= result["_med_12m_ago"].fillna(global_med_ratio)

    result["artist_momentum_6m"]  = np.where(has_6m_activity,  med_now - med_6ago,  0.0)
    result["artist_momentum_12m"] = np.where(has_12m_activity, med_now - med_12ago, 0.0)

    df = result.drop(columns=[
        "_med_now", "_cnt_now", "_med_6m_ago", "_cnt_6m_ago",
        "_med_12m_ago", "_cnt_12m_ago",
    ], errors="ignore")

    return df.sort_values("sale_date").reset_index(drop=True)


def _compute_days_since_last_sale(df: pd.DataFrame) -> pd.DataFrame:
    """Add days_since_last_sale: days since artist's most recent sold lot, strictly before this date."""
    sold_dates = (df[df["sold"] == 1][["artist", "sale_date"]]
                  .drop_duplicates(["artist", "sale_date"])
                  .rename(columns={"sale_date": "_prev_date"})
                  .sort_values("_prev_date")
                  .reset_index(drop=True))

    df_sorted = df.sort_values("sale_date").reset_index(drop=True)

    merged = pd.merge_asof(
        df_sorted[["artist", "sale_date"]],
        sold_dates,
        left_on="sale_date",
        right_on="_prev_date",
        by="artist",
        direction="backward",
        allow_exact_matches=False,
    )

    gap = (df_sorted["sale_date"] - pd.to_datetime(merged["_prev_date"].values))
    df_sorted["days_since_last_sale"] = (
        pd.to_timedelta(gap).dt.days
        .fillna(3 * 365).clip(0, 10 * 365)
    )
    return df_sorted.sort_values("sale_date").reset_index(drop=True)


def _compute_rolling_sellthru(df: pd.DataFrame, global_sell_thru: float) -> pd.DataFrame:
    """
    Add artist_sell_thru_12m: fraction of this artist's lots that sold
    in the 12 months strictly before each lot's date.
    Vectorized via merge_asof — no row-by-row iteration.
    """
    all_lots = (df[["artist", "sale_date", "sold"]]
                .dropna(subset=["sale_date"])
                .sort_values(["artist", "sale_date"])
                .reset_index(drop=True))

    grp = all_lots.groupby("artist", sort=False)
    all_lots["_cum_total"] = grp.cumcount() + 1
    all_lots["_cum_sold"]  = grp["sold"].cumsum()

    # One snapshot per (artist, date) — last row = most up-to-date for that date
    snap = (all_lots.sort_values(["artist", "sale_date"])
                    .groupby(["artist", "sale_date"], sort=False)
                    .last()[["_cum_total", "_cum_sold"]]
                    .reset_index()
                    .sort_values("sale_date")
                    .reset_index(drop=True))

    df_sorted = df.sort_values("sale_date").reset_index(drop=True)

    # Cumulative state strictly before lot date
    now = pd.merge_asof(
        df_sorted[["artist", "sale_date"]],
        snap.rename(columns={"_cum_total": "_tot_now", "_cum_sold": "_sold_now"}),
        on="sale_date", by="artist", direction="backward", allow_exact_matches=False,
    )

    # Cumulative state 12 months ago
    lag_df = df_sorted[["artist", "sale_date"]].copy()
    lag_df["_orig_idx"] = lag_df.index
    lag_df["_lag_date"] = lag_df["sale_date"] - pd.DateOffset(months=12)
    lag_df = lag_df.sort_values("_lag_date").reset_index(drop=True)

    snap_lag = snap.rename(columns={
        "sale_date": "_lag_date",
        "_cum_total": "_tot_ago",
        "_cum_sold":  "_sold_ago",
    })
    lag_res = pd.merge_asof(
        lag_df[["_orig_idx", "_lag_date", "artist"]],
        snap_lag[["_lag_date", "artist", "_tot_ago", "_sold_ago"]],
        on="_lag_date", by="artist", direction="backward", allow_exact_matches=False,
    ).set_index("_orig_idx").sort_index()

    delta_total = (now["_tot_now"].fillna(0).values -
                   lag_res["_tot_ago"].fillna(0).values).clip(0)
    delta_sold  = (now["_sold_now"].fillna(0).values -
                   lag_res["_sold_ago"].fillna(0).values).clip(0)

    has_data = delta_total > 0
    df_sorted["artist_sell_thru_12m"] = np.where(
        has_data,
        np.clip(delta_sold / np.where(has_data, delta_total, 1), 0, 1),
        global_sell_thru,
    )
    return df_sorted.sort_values("sale_date").reset_index(drop=True)


def _compute_artist_house_ratio(df: pd.DataFrame, global_med_ratio: float) -> pd.DataFrame:
    """
    Add artist_house_ratio: each artist's historical median log_ratio at this
    specific auction house, computed point-in-time (strictly before lot date).

    Why useful: artist×house interaction that LGBM can't see directly — the same
    artist may systematically beat estimates more at Christie's than Phillips
    (different buyer clienteles, relationship depth, catalogue placement).

    Vectorized: uses a compound (artist||house_cat) key for merge_asof.
    """
    sold = df[df["sold"] == 1][["artist", "house_cat", "sale_date", "log_ratio"]].copy()
    sold = sold.dropna(subset=["sale_date", "log_ratio"])
    sold["_key"] = sold["artist"].astype(str) + "||" + sold["house_cat"].astype(str)
    sold = sold.sort_values(["_key", "sale_date"]).reset_index(drop=True)

    grp = sold.groupby("_key", sort=False)
    sold["_cum_med"] = grp["log_ratio"].expanding().median().reset_index(level=0, drop=True)

    snap = (sold.sort_values(["_key", "sale_date"])
                .groupby(["_key", "sale_date"], sort=False)
                .last()[["_cum_med"]]
                .reset_index()
                .sort_values("sale_date")
                .reset_index(drop=True))

    df_sorted = df.sort_values("sale_date").reset_index(drop=True)
    df_sorted["_key"] = df_sorted["artist"].astype(str) + "||" + df_sorted["house_cat"].astype(str)

    merged = pd.merge_asof(
        df_sorted,
        snap[["sale_date", "_key", "_cum_med"]],
        on="sale_date",
        by="_key",
        direction="backward",
        allow_exact_matches=False,
    )

    merged["artist_house_ratio"] = merged["_cum_med"].fillna(global_med_ratio)
    df_sorted = merged.drop(columns=["_key", "_cum_med"], errors="ignore")
    return df_sorted.sort_values("sale_date").reset_index(drop=True)


# ── feature columns for model ──────────────────────────────────────────────

FEATURE_COLS = [
    # Estimate (strongest signal)
    "log_estimate_mid",
    "estimate_range_pct",

    # Artist history
    "artist_n_past_sales",
    "artist_median_log_hammer",
    "artist_median_ratio",
    "artist_sell_through_rate",
    "artist_log_price_std",
    # DROPPED: artist_trend_12m_ratio — identical to artist_last_ratio (code confirmed)
    # DROPPED: artist_has_history — binary threshold of artist_n_past_sales (redundant)
    "artist_last_ratio",       # single-sale recency — hedged back in; ablation showed val regressed without it
    "artist_smoothed_ratio",
    "artist_house_ratio",      # artist's median log_ratio at THIS house — house-specific track record

    # Artwork
    "medium_cat",
    "log_width",
    "log_height",
    # DROPPED: has_dims — 0.01% gain; model learns to discount log_w/h=0 naturally
    "aspect_ratio",
    # DROPPED: log_area (99.96% corr with log_w+log_h — mathematically redundant)
    #          is_work_on_paper (redundant with medium_cat, avg 0.2%)
    #          exhibited (confounded with estimate level, avg 0.2%)

    # Provenance
    "provenance_len",
    "prov_auction_count",
    "prov_private_collection",   # "private collection" mention → -4.9pp signal on log_ratio
    # NOTE: 8 richer provenance/title text features (documentation depth, blue-chip
    # dealer/prestige cues, title language) were tested under walk-forward CV and
    # REJECTED — net neutral-to-slightly-negative (pooled MedAPE 28.4→28.6, AUC
    # 0.621→0.616). The specialist already prices lot-record text; adding it as raw
    # features only added noise. Kept out for parsimony.

    # Artist nationality / region (new feature)
    "artist_nationality_region",

    # Temporal
    "sale_year",
    # DROPPED: sale_month — redundant with sale_month_sin/cos (which encode wraparound correctly)
    # DROPPED: sale_quarter — derived from sale_month, no additional signal
    "artwork_age",

    # House / sale type
    "house_cat",
    "is_evening_sale",   # prestige/evening sale (≤80 lots, median est ≥$500k)
    "is_online_sale",    # online-only sale (≥300 lots) — different bidder pool dynamics

    # Market index (macro signal — NEW)
    "market_index_3m",
    "market_index_12m",
    # DROPPED: market_momentum — ratio of 3m/12m index; tree models can derive this
    #          implicitly from the two raw features above (calculated redundancy)

    # External macro — S&P 500 (collector wealth / risk-on signal)
    "sp500_ret_3m",            # trailing 3-month S&P 500 return
    "sp500_ret_12m",           # trailing 12-month S&P 500 return
    "sp500_level",             # normalised S&P 500 price level [0-1] (2015–2026)

    # Artist momentum (better than artist_trend_12m_ratio — NEW)
    "artist_momentum_6m",
    "artist_momentum_12m",

    # Additional engineered signals
    "artist_prestige_score",
    # DROPPED: title_is_untitled — 0.11% gain; fully captured by log_title_length=0 for untitled works
    # DROPPED: estimate_in_millions — log_estimate_mid already carries this info continuously;
    #          binary $1M threshold adds minimal signal for tree models
    "log_lot_number",
    "lot_position_pct",        # lot_number / total_lots_in_sale — relative placement vs absolute position

    # Cyclical seasonality (better than raw month integer)
    "sale_month_sin",
    "sale_month_cos",

    # Estimate signal improvements
    "log_estimate_spread",
    # DROPPED: estimate_is_round — 0.12% gain; correlated with log_estimate_spread (round estimates
    #          tend to have wider spreads), adding negligible signal at cost of noise

    # Documentation depth  (title_is_untitled dropped: 0.11% gain, captured by log_title_length=0)
    "log_title_length",

    # Artist momentum acceleration (short-term vs long-term trajectory)
    "artist_momentum_accel",

    # Relative positioning
    "lot_vs_artist_median",       # is this lot above/below artist's career median?
    "estimate_tightness",         # how tight is the estimate band?

    # Artist temporal signals
    "days_since_last_sale",       # recency of artist's market activity
    "artist_sell_thru_12m",       # rolling 12-month sell-through rate
]


def temporal_split(df: pd.DataFrame):
    """Train: 2018–2024 | Val: 2025 H1 (Jan–Jun) | Test: 2025 H2 (Jul–Dec)

    2026 rows have no hammer prices (upcoming/unsold), so 2025 is split
    half-yearly: H1 for calibration/conformal, H2 for honest evaluation.
    This adds all of 2024 (+16k rows) to training vs. the prior split.
    """
    sale_month = df["sale_date"].dt.month.fillna(0).astype(int)
    sale_year  = df["sale_year"].fillna(0).astype(int)

    train = df[sale_year <= 2024]
    val   = df[(sale_year == 2025) & (sale_month <= 6)]
    test  = df[(sale_year == 2025) & (sale_month >= 7)]
    print(f"  Split → train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test
