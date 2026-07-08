# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Data cleaning pipeline for art auction ML dataset.
===================================================
Combines Christie's, Bonhams, Phillips, and Sotheby's CSVs into a single
clean training file: data/art_auction_clean.csv

Steps:
  1. Load all four CSVs
  2. Normalize artist names (strip birth/death dates, fix casing)
  3. Filter junk rows (decorative objects, missing estimates, extreme outliers)
  4. Deduplicate across houses
  5. Normalize medium to standard categories
  6. Cast and validate numerics
  7. Save combined clean CSV + print stats report

Usage:
  python clean_data.py
  python clean_data.py --output data/clean.csv --report
"""

import argparse
import csv
import re
import os
from collections import defaultdict
from typing import Optional

# ── paths ──────────────────────────────────────────────────────────────────

CHRISTIES = "data/christies_raw.csv"
PHILLIPS  = "data/phillips_real.csv"
SOTHEBYS  = "data/sothebys_raw.csv"
BONHAMS   = "data/bonhams_raw.csv"
OUTPUT    = "data/art_auction_clean.csv"

# Minimum plausible USD low-estimate. Lots below this are almost certainly a
# foreign currency stored as a raw number (e.g. HK$500 = 500). Single source of
# truth so the cleaner and downstream scripts (generate_presale) stay consistent.
MIN_USD_ESTIMATE = 200

# ── Wine / spirits / collectibles exclusion (this is a FINE-ART dataset) ─────
# The houses also sell wine & spirits; those lots leak in with the producer/vintage
# as the "artist" (e.g. "Château Latour 1863", "Karuizawa 12 Year Old Whisky").
# Two tiers to avoid nuking real painters: STRONG terms are unambiguous (producers,
# spirits, bottle sizes); WEAK terms (latour, margaux, champagne…) also appear in
# real names (Henri Fantin-Latour!) so they only count WITH a vintage year or a
# "Mixed" prefix. Single source of truth, reused by load_data().
_WINE_STRONG = re.compile(
    r"(?i)\b(?:domaine|montrachet|vougeot|romanée|romanee|chambertin|meursault|pommard|"
    r"gevrey|vosne|echézeaux|echezeaux|musigny|corton|barolo|brunello|sauternes|pauillac|"
    r"yquem|petrus|pétrus|krug|whisky|whiskey|cognac|armagnac|macallan|karuizawa|bourbon|"
    r"tequila|mezcal)\b|grand cru|premier cru|\b(?:magnum|jeroboam|methuselah|nebuchadnezzar)\b|"
    r"dom p[ée]rignon|haut[- ]brion|ch[âa]teau")
_WINE_WEAK = re.compile(r"(?i)\b(?:latour|margaux|lafite|mouton|champagne|bordeaux|burgund|chablis|comtes)\b")
_VINTAGE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def is_wine_or_spirit(artist: str, medium: str = "") -> bool:
    """True if a lot is wine/spirits/collectible rather than fine art."""
    a = (artist or "")
    if _WINE_STRONG.search(a):
        return True
    if _WINE_WEAK.search(a) and (_VINTAGE_YEAR.search(a) or a.startswith("Mixed")):
        return True
    if re.search(r"(?i)\bwine\b|\bspirits?\b", medium or ""):
        return True
    return False

# ── artist name normalization ───────────────────────────────────────────────

# Matches trailing parenthetical containing a 4-digit year:
#   "(Paris 1901-1985 London)", "(b. 1965)", "(EDINBURGH 1796-1864 LONDON)"
_TRAILING_PARENS = re.compile(r'\s*\([^)]*\b\d{4}\b[^)]*\)\s*$')
# Trailing honorifics: ", R.A." ", P.R.A." ", A.R.A."
_HONORIFICS = re.compile(r',?\s+[A-Z]\.(?:[A-Z]\.)+\s*$')

def normalize_artist(raw: str) -> str:
    name = raw.strip()
    # Remove trailing year-containing parentheticals (Christie's verbose format)
    name = _TRAILING_PARENS.sub("", name).strip()
    # Remove trailing honorifics
    name = _HONORIFICS.sub("", name).strip().rstrip(",").strip()
    # Christie's stores names in ALL CAPS — convert to Title Case
    if name.isupper():
        name = name.title()
        # Restore lowercase particles: "De La", "Van Der", "Le", "La", "Da"
        for particle in ["De ", "Di ", "Du ", "Van ", "Von ", "Le ", "La ", "Da "]:
            name = name.replace(particle, particle[0].upper() + particle[1:].lower())
    return name[:100]


# ── medium normalization ────────────────────────────────────────────────────

_MEDIUM_MAP = [
    # More specific patterns first
    (r"oil.{0,15}canvas",                  "oil on canvas"),
    (r"oil.{0,15}board",                   "oil on board"),
    (r"oil.{0,15}panel",                   "oil on panel"),
    (r"oil.{0,15}paper",                   "oil on paper"),
    (r"oil.{0,15}linen",                   "oil on canvas"),
    (r"acrylic.{0,15}canvas",              "acrylic on canvas"),
    (r"acrylic.{0,15}board",               "acrylic on board"),
    (r"acrylic.{0,15}panel",               "acrylic on panel"),
    (r"acrylic.{0,15}paper",               "acrylic on paper"),
    (r"watercolou?r",                       "watercolor"),
    (r"gouache",                            "gouache"),
    (r"ink.{0,15}paper",                   "ink on paper"),
    (r"ink.{0,15}canvas",                  "ink on canvas"),
    (r"graphite",                           "graphite on paper"),
    (r"pencil",                             "pencil on paper"),
    (r"pastel",                             "pastel"),
    (r"charcoal",                           "charcoal on paper"),
    (r"bronze",                             "bronze sculpture"),
    (r"marble",                             "marble sculpture"),
    (r"chromogenic|c-print|silver.?gelatin", "photograph"),
    (r"archival pigment|inkjet",            "photograph"),
    (r"photograph",                         "photograph"),
    (r"screenprint|silkscreen",             "screenprint"),
    (r"lithograph",                         "lithograph"),
    (r"etching",                            "etching"),
    (r"woodcut|woodblock",                  "woodcut"),
    (r"mixed media",                        "mixed media"),
    (r"collage",                            "collage"),
    (r"video|neon|installation",            "installation/video"),
    (r"print",                              "print"),
]

def normalize_medium(raw: str) -> str:
    if not raw or raw.strip().lower() in ("", "unknown"):
        return "unknown"
    t = raw.lower()
    for pattern, label in _MEDIUM_MAP:
        if re.search(pattern, t):
            return label
    return raw.strip()[:80]


# ── junk row filter ─────────────────────────────────────────────────────────

# Artist field is actually an object description (slipped through data pipeline filters)
_JUNK_ARTIST = re.compile(
    r"^(a |an |the |ensemble |cendrier|vase |plat |bol |paire |lot of |"
    r"property |collection of |various |unsigned|after |circle of |"
    r"attributed |follower |manner of |workshop of |studio of )",
    re.IGNORECASE,
)

def is_junk(row: dict) -> tuple[bool, str]:
    artist = (row.get("artist") or "").strip()
    if not artist:
        return True, "no artist"
    if _JUNK_ARTIST.match(artist):
        return True, "junk artist"
    if is_wine_or_spirit(artist, row.get("medium") or ""):
        return True, "wine/spirits/collectible (not fine art)"

    try:
        est_lo = float(row.get("estimate_low") or 0)
        est_hi = float(row.get("estimate_high") or 0)
    except (ValueError, TypeError):
        return True, "invalid estimate"

    if est_lo <= 0 or est_hi <= 0:
        return True, "missing estimate"
    if est_hi < est_lo:
        return True, "estimate_hi < estimate_lo"
    # Likely non-USD currency stored as raw number (e.g. HK$500 = 500 raw)
    if est_lo < MIN_USD_ESTIMATE:
        return True, f"estimate < ${MIN_USD_ESTIMATE} (likely non-USD)"

    # Extreme sold/estimate ratios — catches guitars, instrument sales, memorabilia
    # books, and data errors that slipped through the artist-name filter.
    # Kept deliberately coarse (50x) to preserve genuine 10x–20x surprises
    # (e.g. Bob Ross 19.7x, Toulouse-Lautrec 19.9x) that are valid training signal.
    try:
        sp = float(row.get("sold_price") or 0)
        if sp > 0:
            if sp > est_lo * 50:
                return True, "sold >50x estimate (likely wrong category or data error)"
            if sp < est_hi * 0.01:
                return True, "sold <1% of estimate (data error)"
    except (ValueError, TypeError):
        pass

    return False, ""


# ── deduplication ───────────────────────────────────────────────────────────

def dedup_key(row: dict) -> str:
    """Fuzzy key: artist + first 30 chars of title + YYYY-MM of sale."""
    artist = normalize_artist(row.get("artist", "")).lower()
    title  = re.sub(r"[^a-z0-9]", "", (row.get("title") or "").lower())[:30]
    date   = (row.get("sale_date") or "")[:7]
    return f"{artist}|{title}|{date}"


# ── main pipeline ───────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows):>7,} rows  ← {path}")
    return rows


def clean(output: str = OUTPUT, report: bool = False):
    print("\n── Loading raw CSVs ──────────────────────────────────────────")
    all_rows: list[dict] = []
    for path in [CHRISTIES, PHILLIPS, SOTHEBYS, BONHAMS]:
        all_rows.extend(load_csv(path))
    print(f"  Total raw: {len(all_rows):,}")

    print("\n── Cleaning ──────────────────────────────────────────────────")
    clean_rows: list[dict] = []
    junk_tally: dict[str, int] = defaultdict(int)
    seen_keys:  set[str] = set()
    dupes = 0

    for row in all_rows:
        # 1. Junk filter
        bad, reason = is_junk(row)
        if bad:
            junk_tally[reason] += 1
            continue

        # 2. Normalize artist
        row["artist"] = normalize_artist(row["artist"])

        # 3. Normalize medium
        row["medium"] = normalize_medium(row.get("medium") or "")

        # 4. Dedup (cross-house fuzzy match)
        key = dedup_key(row)
        if key in seen_keys:
            dupes += 1
            continue
        seen_keys.add(key)

        # 5. Cast numerics
        for field in ("estimate_low", "estimate_high", "sold_price",
                      "width_cm", "height_cm"):
            v = row.get(field, "")
            row[field] = "" if not v else _safe_float(v)

        for field in ("year_made", "lot_number"):
            v = row.get(field, "")
            row[field] = "" if not v else _safe_int(v)

        # 6. Normalize booleans
        for field in ("signed", "exhibited", "literature"):
            v = str(row.get(field, "")).strip().lower()
            row[field] = True if v in ("true", "1", "yes") else False

        # 7. Normalize sale_date to YYYY-MM-DD
        sd = str(row.get("sale_date") or "")
        if re.match(r"^\d{4}-\d{2}$", sd):
            sd = sd + "-01"   # Phillips: YYYY-MM → YYYY-MM-01
        row["sale_date"] = sd[:10]

        clean_rows.append(row)

    print(f"  Clean rows: {len(clean_rows):,}")
    print(f"  Dupes removed: {dupes:,}")
    print(f"  Junk removed:")
    for reason, n in sorted(junk_tally.items(), key=lambda x: -x[1]):
        print(f"    {n:>7,}  {reason}")

    # ── stats ──────────────────────────────────────────────────────────────
    print("\n── Dataset stats ─────────────────────────────────────────────")
    by_house:  dict[str, int] = defaultdict(int)
    by_year:   dict[str, int] = defaultdict(int)
    sold_n = unsold_n = 0

    for r in clean_rows:
        by_house[r.get("house", "?")] += 1
        yr = (r.get("sale_date") or "")[:4]
        if yr:
            by_year[yr] += 1
        if r.get("sold_price"):
            sold_n += 1
        else:
            unsold_n += 1

    total = len(clean_rows)
    for house, n in sorted(by_house.items()):
        print(f"  {house:<12} {n:>7,}")
    print(f"  {'TOTAL':<12} {total:>7,}")
    print(f"\n  Sold (hammer price):  {sold_n:,} ({100*sold_n//max(total,1)}%)")
    print(f"  Unsold / pre-sale:    {unsold_n:,} ({100*unsold_n//max(total,1)}%)")

    print(f"\n  By year:")
    for yr, n in sorted(by_year.items()):
        print(f"    {yr}: {n:,}")

    miss_medium = sum(1 for r in clean_rows if r.get("medium") == "unknown")
    miss_dims   = sum(1 for r in clean_rows if not r.get("width_cm"))
    miss_yr     = sum(1 for r in clean_rows if not r.get("year_made"))
    print(f"\n  Missing medium:     {miss_medium:,} ({100*miss_medium//max(total,1)}%)")
    print(f"  Missing dimensions: {miss_dims:,} ({100*miss_dims//max(total,1)}%)")
    print(f"  Missing year_made:  {miss_yr:,} ({100*miss_yr//max(total,1)}%)")

    # ── write ──────────────────────────────────────────────────────────────
    fields = [
        "house", "auction_id", "lot_number", "artist", "title",
        "year_made", "medium", "width_cm", "height_cm", "provenance",
        "signed", "exhibited", "literature",
        "estimate_low", "estimate_high", "sold_price", "sale_date",
    ]
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(clean_rows)
    print(f"\n  Saved → {output}")

    if report:
        rpt = output.replace(".csv", "_report.txt")
        with open(rpt, "w") as f:
            f.write(f"Art Auction Clean Dataset\n{'='*40}\n")
            f.write(f"Total rows:          {total:,}\n")
            f.write(f"With hammer price:   {sold_n:,} ({100*sold_n//max(total,1)}%)\n\n")
            f.write("By house:\n")
            for h, n in sorted(by_house.items()):
                f.write(f"  {h}: {n:,}\n")
            f.write("\nBy year:\n")
            for yr, n in sorted(by_year.items()):
                f.write(f"  {yr}: {n:,}\n")
            f.write(f"\nMissing medium:     {miss_medium:,} ({100*miss_medium//max(total,1)}%)\n")
            f.write(f"Missing dimensions: {miss_dims:,} ({100*miss_dims//max(total,1)}%)\n")
            f.write(f"Missing year_made:  {miss_yr:,} ({100*miss_yr//max(total,1)}%)\n")
        print(f"  Report → {rpt}")


# ── helpers ────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def _safe_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=OUTPUT, help="Output CSV path")
    ap.add_argument("--report", action="store_true", help="Also write a text report")
    args = ap.parse_args()
    clean(args.output, args.report)
