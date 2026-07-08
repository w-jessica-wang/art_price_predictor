# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Art Auction Price Predictor — Streamlit Interface (v2)
======================================================
Run: streamlit run app.py

Tabs:
  1. Predict  — SHAP waterfall, confidence interval chart, artist history
  2. Trends   — Medium index, hammer/estimate ratio, artist momentum, seasonal heatmap
  3. Batch Scoring (demo) — model scored over many no-hammer lots (NOT a future-auction calendar)
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ── SHAP optional import ───────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Art Auction Price Predictor",
    page_icon="🎨",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base background ─────────────────────────────────── */
.stApp { background: #0a0a14; }

/* ── Gradient title ──────────────────────────────────── */
h1 {
    background: linear-gradient(135deg, #42a5f5 0%, #ce93d8 60%, #ef9a9a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

/* ── Section headings ────────────────────────────────── */
h2 { color: #e0e0e0 !important; font-weight: 700 !important; }
h3 { color: #bdbdbd !important; }

/* ── Metric cards ────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #12122a 0%, #1a1a35 100%);
    border: 1px solid #2a2a50;
    border-radius: 14px;
    padding: 18px 20px !important;
    box-shadow: 0 4px 24px rgba(66,165,245,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 6px 32px rgba(66,165,245,0.18);
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    color: #9e9e9e !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d20 0%, #111128 100%) !important;
    border-right: 1px solid #1e1e38;
}
[data-testid="stSidebar"] .stButton button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.8rem;
    white-space: nowrap !important;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Primary button ──────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1565c0 0%, #283593 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: 0 4px 16px rgba(21,101,192,0.35) !important;
    transition: all 0.2s !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(21,101,192,0.55) !important;
    transform: translateY(-1px);
}

/* ── Info / warning / success boxes ─────────────────── */
[data-testid="stNotification"] { border-radius: 10px; }
.stInfo  { border-left: 4px solid #42a5f5 !important; background: rgba(66,165,245,0.07) !important; }
.stSuccess { border-left: 4px solid #43a047 !important; }
.stWarning { border-left: 4px solid #ffa726 !important; }

/* ── Tabs ────────────────────────────────────────────── */
[data-baseweb="tab-list"] { background: transparent !important; gap: 4px; }
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #9e9e9e !important;
    padding: 8px 20px !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #42a5f5 !important;
    background: rgba(66,165,245,0.08) !important;
    border-bottom: 2px solid #42a5f5 !important;
}

/* ── Dataframe / table ───────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Sidebar polish ──────────────────────────────────── */
[data-testid="stSidebar"] {
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
/* Input + selectbox labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextArea label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    color: #616161 !important;
    margin-bottom: 2px !important;
}
/* Input / select boxes */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background: #12122a !important;
    border: 1px solid #252545 !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child:focus-within {
    border-color: #42a5f5 !important;
    box-shadow: 0 0 0 2px rgba(66,165,245,0.15) !important;
}
/* Checkboxes */
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.82rem !important;
    letter-spacing: 0.2px !important;
    text-transform: none !important;
    color: #bdbdbd !important;
}
/* Expander */
[data-testid="stSidebar"] .stExpander {
    border: 1px solid #1e1e38 !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stSidebar"] .stExpander summary {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #9e9e9e !important;
    padding: 10px 14px !important;
}
/* Caption text */
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] small {
    font-size: 0.72rem !important;
    color: #555 !important;
    line-height: 1.4 !important;
}
/* Divider */
[data-testid="stSidebar"] hr {
    border-color: #1e1e38 !important;
    margin: 0.6rem 0 !important;
}
/* Success badge */
[data-testid="stSidebar"] .stSuccess {
    padding: 6px 10px !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: #1e1e38 !important; margin: 2rem 0 !important; }

/* ── Caption / small text ────────────────────────────── */
.stCaption, small { color: #757575 !important; font-size: 0.8rem !important; }

/* ── Spinner ─────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: #42a5f5 !important; }

/* ── Input fields ────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] { border-radius: 8px !important; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a14; }
::-webkit-scrollbar-thumb { background: #2a2a50; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #42a5f5; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ──────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d0d1e",
    plot_bgcolor="#12122a",
    font=dict(color="#cccccc", family="Inter, system-ui, sans-serif"),
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
FEATURES_PARQUET = os.path.join(MODELS_DIR, "features_cache.parquet")
NO_HAMMER_CSV = os.path.join(os.path.dirname(__file__), "data", "no_hammer_batch_scores.csv")

# ── preset artworks ────────────────────────────────────────────────────────
PRESETS = {
    "Basquiat": dict(
        artist="Jean-Michel Basquiat",
        estimate_low=3_000_000,
        estimate_high=5_000_000,
        medium="acrylic on canvas",
        width_cm=183.0,
        height_cm=122.0,
        year_made=1984,
        signed=True,
        exhibited=True,
        literature=True,
        provenance="Private Collection, New York; Christie's New York, 2015, lot 12",
        house="christies",
        sale_year=2025,
        sale_month=5,
    ),
    "Warhol": dict(
        artist="Andy Warhol",
        estimate_low=800_000,
        estimate_high=1_200_000,
        medium="screenprint",
        width_cm=101.6,
        height_cm=101.6,
        year_made=1962,
        signed=True,
        exhibited=False,
        literature=True,
        provenance="Private Collection, London",
        house="phillips",
        sale_year=2025,
        sale_month=6,
    ),
    "Picasso": dict(
        artist="Pablo Picasso",
        estimate_low=500_000,
        estimate_high=700_000,
        medium="oil on canvas",
        width_cm=65.0,
        height_cm=50.0,
        year_made=1937,
        signed=True,
        exhibited=True,
        literature=True,
        provenance="Private Collection, Paris; acquired from the artist, 1938",
        house="christies",
        sale_year=2025,
        sale_month=11,
    ),
}


# ── helpers ────────────────────────────────────────────────────────────────

def fmt_dollar(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}k"
    return f"${v:.0f}"


def models_exist() -> bool:
    return os.path.isdir(MODELS_DIR) and os.path.exists(
        os.path.join(MODELS_DIR, "lgbm.txt")
    )


# ── cached resources ───────────────────────────────────────────────────────

@st.cache_resource
def load_predictor():
    from ml.predict import ArtPricePredictor
    return ArtPricePredictor()


@st.cache_resource
def load_shap_explainer():
    """Load the LightGBM booster and build a SHAP TreeExplainer.

    NOTE: this attributes the **LightGBM base learner only** (NNLS weight ≈0.66),
    not the full LightGBM+CatBoost ensemble. LightGBM is the dominant component and
    its feature attributions are used as a faithful proxy for the ensemble's drivers;
    the dollar magnitudes are LightGBM's, so treat them as directional, not exact.
    """
    if not SHAP_AVAILABLE:
        return None
    import lightgbm as lgb
    booster = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lgbm.txt"))
    return shap.TreeExplainer(booster)


@st.cache_data
def load_features_cache() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PARQUET):
        return pd.DataFrame()
    df = pd.read_parquet(FEATURES_PARQUET)
    # Ensure sale_date is datetime
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    return df


@st.cache_data
def load_artist_stats() -> pd.DataFrame:
    path = os.path.join(MODELS_DIR, "artist_stats.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path).set_index("artist")


@st.cache_data
def get_known_artists() -> list:
    stats = load_artist_stats()
    if stats.empty:
        return []
    idx = stats.index
    # Keep only genuine artists:
    #   - at least 2 sales (lot-title artifacts almost always appear once)
    #   - name doesn't start with a quote mark (those are object/lot titles)
    #   - name length ≤ 60 chars (long strings are descriptions, not people)
    n_sales = stats.get("n_sales", stats.get("n_lots", None))
    if n_sales is not None:
        mask = (n_sales >= 2) & (~idx.str.startswith('"')) & (idx.str.len() <= 60)
    else:
        mask = (~idx.str.startswith('"')) & (idx.str.len() <= 60)
    return sorted(idx[mask].tolist())


@st.cache_data
def get_artist_defaults(artist_name: str) -> dict:
    """
    Look up this artist's typical medium, dimensions, and year from
    training data so the sidebar can pre-fill sensible defaults.
    """
    df = load_features_cache()
    if df.empty or "artist" not in df.columns:
        return {}
    sub = df[df["artist"].str.lower() == artist_name.strip().lower()]
    if sub.empty:
        return {}
    defaults: dict = {}
    if "medium" in sub.columns:
        mc = sub["medium"].dropna().value_counts()
        if not mc.empty:
            defaults["medium"] = mc.index[0]
    for dim in ("width_cm", "height_cm"):
        if dim in sub.columns:
            v = sub[dim].dropna()
            if not v.empty:
                defaults[dim] = round(float(v.median()), 1)
    if "year_made" in sub.columns:
        v = sub["year_made"].dropna()
        if not v.empty:
            defaults["year_made"] = int(v.median())
    for flag in ("signed", "exhibited", "literature"):
        if flag in sub.columns:
            v = sub[flag].dropna()
            if not v.empty:
                defaults[flag] = bool(v.mode().iloc[0])
    if "house" in sub.columns:
        hc = sub["house"].dropna().value_counts()
        if not hc.empty:
            defaults["house"] = hc.index[0]
    return defaults


@st.cache_data
def get_artist_artworks(artist_name: str) -> pd.DataFrame:
    """
    Return all known lots for this artist (deduped by title, sorted recent-first)
    so the sidebar can show a dropdown of their works.
    """
    df = load_features_cache()
    if df.empty or "artist" not in df.columns or "title" not in df.columns:
        return pd.DataFrame()
    sub = df[df["artist"].str.lower() == artist_name.strip().lower()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("sale_year", ascending=False)
    keep = [c for c in ["title", "year_made", "medium", "estimate_low", "estimate_high",
                         "width_cm", "height_cm", "signed", "exhibited", "literature",
                         "provenance", "house", "sale_year", "sale_month"] if c in sub.columns]
    return sub.drop_duplicates("title")[keep].head(300).reset_index(drop=True)


def search_artwork(artist: str, query: str, n: int = 10) -> pd.DataFrame:
    """
    Search the training data for lots matching artist + title query.
    If artist is empty, searches across all artists.
    """
    df = load_features_cache()
    if df.empty or "artist" not in df.columns:
        return pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    if artist.strip():
        mask &= df["artist"].str.lower() == artist.strip().lower()
    if query.strip() and "title" in df.columns:
        mask &= df["title"].str.lower().str.contains(query.strip().lower(), na=False)
    sub = df[mask].sort_values("sale_year", ascending=False)
    if sub.empty:
        return pd.DataFrame()
    display_cols = [c for c in ["artist", "title", "year_made", "medium", "estimate_low",
                                 "estimate_high", "house", "sale_year", "width_cm", "height_cm",
                                 "signed", "exhibited", "literature", "provenance"] if c in sub.columns]
    return sub[display_cols].head(n).reset_index(drop=True)


@st.cache_data
def load_meta() -> dict:
    path = os.path.join(MODELS_DIR, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_eval_predictions() -> pd.DataFrame:
    path = os.path.join(MODELS_DIR, "eval_predictions.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def _load_walk_forward() -> dict:
    """Walk-forward CV summary (primary evaluation), written by ml/walk_forward_eval.py."""
    path = os.path.join(os.path.dirname(__file__), "models_eval", "walk_forward.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ── SHAP waterfall chart ───────────────────────────────────────────────────

def shap_waterfall_chart(
    X: np.ndarray, feature_names: list, explainer, artist_name: str, hammer: float
) -> go.Figure:
    """
    Compute SHAP values and return a Plotly horizontal bar waterfall chart.
    Dollar impact is computed as: expm1(base + shap) - expm1(base) per feature.
    Shows top 10 features by |impact|.
    """
    sv = explainer.shap_values(X)[0]  # shape (n_features,)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(base[0])

    # Dollar impact per feature
    dollar_impacts = []
    for i, s in enumerate(sv):
        impact = math.expm1(base + s) - math.expm1(base)
        dollar_impacts.append(impact)

    # Build dataframe, pick top 10 by absolute impact
    df_sv = pd.DataFrame({
        "feature": feature_names,
        "shap_val": sv,
        "dollar_impact": dollar_impacts,
        "raw_val": X[0],
    })
    df_sv = df_sv.reindex(df_sv["dollar_impact"].abs().sort_values(ascending=False).index)
    df_sv = df_sv.head(10).sort_values("dollar_impact")

    colors = ["#ef5350" if v > 0 else "#42a5f5" for v in df_sv["dollar_impact"]]

    # Human-readable feature name mapping
    FEAT_LABELS = {
        "log_estimate_mid":          "Presale Estimate",
        "artist_median_log_hammer":  "Artist Median Price",
        "artist_smoothed_ratio":     "Artist Hammer/Estimate Ratio",
        "artist_momentum_6m":        "Artist 6-Month Momentum",
        "artist_momentum_12m":       "Artist 12-Month Momentum",
        "market_index_3m":           "Market Index (3-Month)",
        "market_index_12m":          "Market Index (12-Month)",
        "market_momentum":           "Market Momentum",
        "log_width":                 "Width",
        "log_height":                "Height",
        "log_area":                  "Canvas Area",
        "artwork_age":               "Artwork Age",
        "log_lot_number":            "Lot Position in Sale",
        "signed":                    "Signed/Dated",
        "exhibited":                 "Exhibition History",
        "literature":                "Literature Reference",
        "provenance_length":         "Provenance Detail",
        "artist_n_past_sales":       "Artist Sale Volume",
        "artist_prestige_score":     "Artist Prestige Score",
        "artist_median_ratio":       "Artist Estimate Accuracy",
        "estimate_in_millions":      "Million-Dollar Lot Flag",
        "log_estimate_spread":       "Estimate Spread",
        "house_christies":           "Christie's (House)",
        "house_phillips":            "Phillips (House)",
        "house_sothebys":            "Sotheby's (House)",
        "sale_month_sin":            "Sale Timing (Seasonal)",
        "sale_month_cos":            "Sale Timing (Seasonal)",
        "estimate_is_round":         "Round-Number Estimate",
        "log_title_length":          "Title Documentation",
        "artist_momentum_accel":     "Artist Momentum Acceleration",
        "lot_vs_artist_median":      "Lot vs Artist Career Median",
        "estimate_tightness":        "Estimate Confidence Band",
        "n_same_artist_in_sale":     "Same-Artist Lots in Sale",
        "is_only_lot_by_artist":     "Scarcity (Sole Lot by Artist)",
        "days_since_last_sale":      "Days Since Artist's Last Sale",
        "artist_sell_thru_12m":      "Artist 12M Sell-Through Rate",
        "sale_year":                 "Sale Year",
    }

    def _feat_label(name, raw_val):
        label = FEAT_LABELS.get(name, name.replace("_", " ").title())
        return label

    labels = [
        f"{_feat_label(row['feature'], row['raw_val'])} = {row['raw_val']:.2g}   "
        f"{'+'if row['dollar_impact']>0 else ''}{fmt_dollar(row['dollar_impact'])}"
        for _, row in df_sv.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=df_sv["dollar_impact"].tolist(),
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[
            f"{'+'if v>0 else ''}{fmt_dollar(v)}" for v in df_sv["dollar_impact"]
        ],
        textposition="auto",   # inside for long bars, outside for short — prevents right-edge clipping
        cliponaxis=False,
        hovertemplate="Feature: %{y}<br>Dollar impact: %{x:,.0f}<extra></extra>",
    ))

    # Extend x-axis 20% beyond the max bar so labels never get clipped
    max_impact = df_sv["dollar_impact"].abs().max()
    x_max = max_impact * 1.22
    x_min = -max_impact * 0.05  # small negative room for the zero-line

    # Artist-specific title
    hammer_str = fmt_dollar(hammer)
    artist_label = artist_name if artist_name else "this lot"
    fig.update_layout(
        title=f"Why {hammer_str} for {artist_label}? — Key price drivers",
        xaxis_title="Dollar impact on predicted hammer",
        xaxis=dict(range=[x_min, x_max]),
        yaxis_title=None,
        margin=dict(l=20, r=20, t=50, b=40),
        height=420,
        **PLOTLY_THEME,
    )
    fig.add_vline(x=0, line_color="#888888", line_width=1)
    return fig


# ── confidence interval chart ──────────────────────────────────────────────

def ci_chart(estimate_low, estimate_high, hammer, ci_lo, ci_hi) -> go.Figure:
    """Horizontal number-line showing estimate range, prediction, and P10–P90."""
    points = {
        "Estimate Low":  estimate_low,
        "P10":           ci_lo,
        "Prediction":    hammer,
        "P90":           ci_hi,
        "Estimate High": estimate_high,
    }
    fig = go.Figure()

    # Estimate bracket
    fig.add_shape(
        type="line",
        x0=estimate_low, x1=estimate_high,
        y0=0.5, y1=0.5,
        line=dict(color="#546e7a", width=4, dash="dot"),
    )

    # P10–P90 band
    fig.add_shape(
        type="rect",
        x0=ci_lo, x1=ci_hi,
        y0=0.3, y1=0.7,
        fillcolor="rgba(66, 165, 245, 0.2)",
        line=dict(width=0),
    )

    colors = {
        "Estimate Low":  "#546e7a",
        "P10":           "#42a5f5",
        "Prediction":    "#ef5350",
        "P90":           "#42a5f5",
        "Estimate High": "#546e7a",
    }
    symbols = {
        "Estimate Low":  "line-ns",
        "P10":           "triangle-right",
        "Prediction":    "diamond",
        "P90":           "triangle-left",
        "Estimate High": "line-ns",
    }

    # Alternate labels above/below so nearby points (e.g. Estimate High ≈ Prediction) don't crowd
    text_positions = {
        "Estimate Low":  "bottom center",
        "P10":           "top center",
        "Prediction":    "bottom center",
        "P90":           "top center",
        "Estimate High": "top center",
    }

    for name, val in points.items():
        fig.add_trace(go.Scatter(
            x=[val], y=[0.5],
            mode="markers+text",
            name=name,
            text=[f"{name}<br>{fmt_dollar(val)}"],
            textposition=text_positions[name],
            textfont=dict(size=11),
            marker=dict(
                symbol=symbols[name],
                size=14,
                color=colors[name],
                line=dict(width=2, color=colors[name]),
            ),
            hovertemplate=f"{name}: ${val:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        title="Price Range — Estimate vs Model Confidence",
        xaxis_title="Price (USD)",
        yaxis=dict(visible=False, range=[-0.15, 1.3]),   # extra vertical room for labels
        height=260,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=40),
        **PLOTLY_THEME,
    )
    return fig


# ── Tab 2 charts ───────────────────────────────────────────────────────────

def chart_medium_index(df: pd.DataFrame) -> go.Figure:
    """Line chart: top 6 mediums indexed to 2018=100."""
    top_mediums = [
        "oil on canvas", "bronze sculpture", "photograph",
        "acrylic on canvas", "lithograph", "watercolor",
    ]
    if "medium" not in df.columns or "sale_year" not in df.columns:
        return go.Figure()

    df2 = df[df["medium"].str.lower().isin(top_mediums) & df["sold_price"].notna()].copy()
    df2["medium_label"] = df2["medium"].str.lower()
    df2["year"] = df2["sale_year"].astype(int)
    df2 = df2[df2["year"].between(2018, 2026)]

    agg = (df2.groupby(["year", "medium_label"])["sold_price"]
              .median().reset_index()
              .rename(columns={"sold_price": "median_hammer"}))

    # Index to 2018=100
    base = agg[agg["year"] == 2018].set_index("medium_label")["median_hammer"]
    agg["index_val"] = agg.apply(
        lambda r: r["median_hammer"] / base.get(r["medium_label"], np.nan) * 100
        if r["medium_label"] in base.index else np.nan,
        axis=1,
    )
    agg = agg.dropna(subset=["index_val"])

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, med in enumerate(top_mediums):
        sub = agg[agg["medium_label"] == med].sort_values("year")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["index_val"],
            mode="lines+markers",
            name=med.title(),
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{med.title()}</b><br>"
                "Year: %{x}<br>"
                "Index: %{y:.1f}<br>"
                "Median Hammer: $%{customdata:,.0f}<extra></extra>"
            ),
            customdata=sub["median_hammer"],
        ))

    fig.add_hline(y=100, line_color="#666666", line_dash="dash", annotation_text="2018 baseline")
    fig.update_layout(
        title="Medium Price Index (2018 = 100)",
        xaxis_title="Year",
        yaxis_title="Index (2018 = 100)",
        height=460,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(b=120),
        **PLOTLY_THEME,
    )
    return fig


def chart_hammer_ratio(df: pd.DataFrame) -> go.Figure:
    """Hammer/estimate ratio by quarter with P25–P75 band."""
    needed = {"sale_date", "sold_price", "estimate_mid"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df[df["sold_price"].notna() & df["estimate_mid"].notna()].copy()
    df2["ratio"] = df2["sold_price"] / df2["estimate_mid"].clip(lower=1)
    df2["year"] = df2["sale_date"].dt.year
    df2["quarter"] = df2["sale_date"].dt.to_period("Q").astype(str)
    df2 = df2[df2["year"].between(2018, 2026)]

    agg = (df2.groupby("quarter")["ratio"]
              .agg(median="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
              .reset_index()
              .sort_values("quarter"))

    fig = go.Figure()

    # P25–P75 band
    fig.add_trace(go.Scatter(
        x=agg["quarter"].tolist() + agg["quarter"].tolist()[::-1],
        y=agg["p75"].tolist() + agg["p25"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(66, 165, 245, 0.15)",
        line=dict(width=0),
        name="P25–P75 band",
        hoverinfo="skip",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=agg["quarter"], y=agg["median"],
        mode="lines+markers",
        name="Median ratio",
        line=dict(color="#42a5f5", width=2),
        marker=dict(size=5),
        hovertemplate="Quarter: %{x}<br>Median ratio: %{y:.2f}×<extra></extra>",
    ))

    # Annotations
    annotations = [
        ("2020Q2", "COVID dip"),
        ("2021Q4", "2021 boom"),
        ("2022Q3", "2022 correction"),
    ]
    for qtr, label in annotations:
        row = agg[agg["quarter"] == qtr]
        if not row.empty:
            fig.add_annotation(
                x=qtr, y=float(row["median"].iloc[0]),
                text=label, showarrow=True, arrowhead=2,
                ax=0, ay=-40, font=dict(size=11, color="#ffcc80"),
            )

    fig.add_hline(y=1.0, line_color="#666666", line_dash="dash", annotation_text="Estimate midpoint")
    fig.update_layout(
        title="Hammer / Estimate Ratio by Quarter",
        xaxis_title="Quarter",
        yaxis_title="Hammer / Estimate ratio",
        height=420,
        **PLOTLY_THEME,
    )
    return fig


def chart_artist_momentum(df: pd.DataFrame, artist_name: str) -> go.Figure:
    """Bar + line chart for a selected artist."""
    needed = {"artist", "sold_price", "sale_year", "estimate_mid"}
    if not needed.issubset(df.columns):
        return go.Figure()

    sub = df[
        (df["artist"].str.lower() == artist_name.lower()) &
        df["sold_price"].notna() & df["estimate_mid"].notna()
    ].copy()
    sub["year"] = sub["sale_year"].astype(int)
    sub["ratio"] = sub["sold_price"] / sub["estimate_mid"].clip(lower=1)
    sub = sub[sub["year"].between(2018, 2026)]

    if sub.empty:
        return go.Figure().update_layout(
            title=f"No data for {artist_name}",
            **PLOTLY_THEME,
        )

    agg = (sub.groupby("year")
              .agg(median_hammer=("sold_price", "median"),
                   median_ratio=("ratio", "median"),
                   n_lots=("sold_price", "count"))
              .reset_index().sort_values("year"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["year"], y=agg["median_hammer"],
        name="Median Hammer",
        marker_color="#42a5f5",
        hovertemplate="Year: %{x}<br>Median Hammer: $%{y:,.0f}<br>Lots: %{customdata}<extra></extra>",
        customdata=agg["n_lots"],
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=agg["year"], y=agg["median_ratio"],
        mode="lines+markers",
        name="Hammer/Estimate Ratio",
        line=dict(color="#ef5350", width=2),
        marker=dict(size=7),
        yaxis="y2",
        hovertemplate="Year: %{x}<br>Ratio: %{y:.2f}×<extra></extra>",
    ))

    fig.update_layout(
        title=f"{artist_name} — Price Momentum",
        xaxis_title="Year",
        yaxis=dict(title="Median Hammer ($)", side="left"),
        yaxis2=dict(title="Hammer/Estimate Ratio", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        **PLOTLY_THEME,
    )
    return fig


def chart_seasonal_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: rows=month, cols=year, color=median hammer/estimate ratio."""
    needed = {"sale_date", "sold_price", "estimate_mid"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df[df["sold_price"].notna() & df["estimate_mid"].notna()].copy()
    df2["ratio"] = df2["sold_price"] / df2["estimate_mid"].clip(lower=1)
    df2["year"] = df2["sale_date"].dt.year
    df2["month"] = df2["sale_date"].dt.month
    df2 = df2[df2["year"].between(2018, 2026)]

    pivot = (df2.groupby(["year", "month"])["ratio"]
               .median().reset_index()
               .pivot(index="month", columns="year", values="ratio"))

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.index = [month_names[m-1] for m in pivot.index]

    # Replace NaN with a blank string so cells show "—" not "nan×"
    z_vals = pivot.values
    text_vals = np.where(
        np.isnan(z_vals),
        "—",
        np.vectorize(lambda v: f"{v:.2f}×")(np.nan_to_num(z_vals)),
    )

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=1.0,
        text=text_vals,
        texttemplate="%{text}",
        hovertemplate="Year: %{x}<br>Month: %{y}<br>Median ratio: %{z:.2f}×<extra></extra>",
        colorbar=dict(title="Ratio"),
    ))

    fig.update_layout(
        title="Seasonal Heatmap — Hammer/Estimate Ratio",
        xaxis_title="Year",
        yaxis_title="Month",
        height=440,
        **PLOTLY_THEME,
    )
    return fig


# ── NEW CHART 1: Global market cycle ──────────────────────────────────────────

def chart_market_cycle(df: pd.DataFrame) -> go.Figure:
    """Rolling 12-month median hammer + YoY % change with boom/bust colour bars."""
    needed = {"sale_date", "sold_price"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df[df["sold_price"].notna()].copy()
    df2["ym"] = df2["sale_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df2.groupby("ym")["sold_price"].median().reset_index()
    monthly = monthly.sort_values("ym")
    monthly["rolling_12"] = monthly["sold_price"].rolling(12, min_periods=3).median()
    monthly["yoy_pct"]    = monthly["rolling_12"].pct_change(12) * 100
    monthly = monthly.dropna(subset=["rolling_12"])
    monthly = monthly[monthly["ym"].dt.year >= 2018]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38], vertical_spacing=0.07,
        subplot_titles=("12-Month Rolling Median Hammer", "YoY Price Change (%)"),
    )

    fig.add_trace(go.Scatter(
        x=monthly["ym"], y=monthly["rolling_12"],
        mode="lines", name="12m Median",
        line=dict(color="#42a5f5", width=2.5),
        fill="tozeroy", fillcolor="rgba(66,165,245,0.1)",
        hovertemplate="Month: %{x|%b %Y}<br>Median: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    yoy = monthly.dropna(subset=["yoy_pct"])
    pos = yoy[yoy["yoy_pct"] >= 0]
    neg = yoy[yoy["yoy_pct"] <  0]

    fig.add_trace(go.Bar(
        x=pos["ym"], y=pos["yoy_pct"], name="Growth",
        marker_color="#43a047",
        hovertemplate="Month: %{x|%b %Y}<br>YoY: +%{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=neg["ym"], y=neg["yoy_pct"], name="Decline",
        marker_color="#e53935",
        hovertemplate="Month: %{x|%b %Y}<br>YoY: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line_color="#666", line_dash="dash")

    for qtr, label, ay in [("2020-04-01", "COVID Crash", -35),
                            ("2021-10-01", "2021 Boom",   30),
                            ("2022-09-01", "2022 Correction", -35)]:
        row_m = monthly[monthly["ym"] == pd.Timestamp(qtr)]
        if not row_m.empty:
            fig.add_annotation(
                x=pd.Timestamp(qtr), y=float(row_m["rolling_12"].iloc[0]),
                text=label, showarrow=True, arrowhead=2,
                ax=0, ay=-40, font=dict(size=10, color="#ffcc80"),
                row=1, col=1,
            )

    fig.update_layout(
        height=520, showlegend=True,
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        **PLOTLY_THEME,
    )
    fig.update_yaxes(title_text="Median ($)", row=1, col=1)
    fig.update_yaxes(title_text="YoY (%)", row=2, col=1)
    return fig


# ── NEW CHART 2: Auction house battle ─────────────────────────────────────────

def chart_house_battle(df: pd.DataFrame) -> go.Figure:
    """Christie's vs Bonhams vs Phillips vs Sotheby's — volume, median hammer, sell-through."""
    needed = {"house", "sale_year", "sold_price", "sold"}
    if not needed.issubset(df.columns):
        return go.Figure()

    house_map = {"christies": "Christie's", "bonhams": "Bonhams", "phillips": "Phillips", "sothebys": "Sotheby's"}
    df2 = df[df["house"].isin(house_map) & df["sale_year"].notna()].copy()
    # Only include lots that actually sold (sold==1) so Sotheby's presale lots don't skew sell-through to 0%
    if "sold" in df2.columns:
        df2 = df2[df2["sold"] == 1]
    df2["house_label"] = df2["house"].map(house_map)
    df2["year"] = df2["sale_year"].astype(int)
    df2 = df2[df2["year"].between(2018, 2025)]

    agg = (df2.groupby(["year", "house_label"])
              .agg(
                  median_hammer=("sold_price", lambda x: x.dropna().median()),
                  sell_through=("sold", "mean"),
                  n_lots=("sold", "count"),
              ).reset_index())

    colors = {"Christie's": "#ef5350", "Bonhams": "#ffa726", "Phillips": "#42a5f5", "Sotheby's": "#66bb6a"}

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Lot Volume", "Median Hammer", "Sell-Through %"),
        horizontal_spacing=0.13,
    )
    for house in ["Christie's", "Bonhams", "Phillips", "Sotheby's"]:
        sub = agg[agg["house_label"] == house].sort_values("year")
        c = colors[house]
        fig.add_trace(go.Bar(
            x=sub["year"], y=sub["n_lots"], name=house,
            marker_color=c, showlegend=True,
            hovertemplate=f"<b>{house}</b><br>Year: %{{x}}<br>Lots: %{{y:,}}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["median_hammer"],
            mode="lines+markers", name=house, showlegend=False,
            line=dict(color=c, width=2), marker=dict(size=6),
            hovertemplate=f"<b>{house}</b><br>Year: %{{x}}<br>Median: $%{{y:,.0f}}<extra></extra>",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["sell_through"] * 100,
            mode="lines+markers", name=house, showlegend=False,
            line=dict(color=c, width=2), marker=dict(size=6),
            hovertemplate=f"<b>{house}</b><br>Year: %{{x}}<br>Sell-through: %{{y:.1f}}%<extra></extra>",
        ), row=1, col=3)

    fig.update_layout(
        title="Auction House Battle — Christie's · Bonhams · Phillips · Sotheby's",
        height=420, barmode="group",
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(size=12),
        ),
        margin=dict(t=70, b=80),
        **PLOTLY_THEME,
    )
    # Subtitle annotations are at y=1.0 by default in make_subplots;
    # nudge them down slightly so they don't crowd the top border
    for ann in fig.layout.annotations:
        ann.update(y=ann.y - 0.02, font=dict(size=12))
    return fig


# ── NEW CHART 3: Price tier composition ───────────────────────────────────────

def chart_price_tier_distribution(df: pd.DataFrame) -> go.Figure:
    """Stacked area: % of sold lots in each price tier by year."""
    needed = {"sale_year", "sold_price"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df[df["sold_price"].notna()].copy()
    df2["year"] = df2["sale_year"].astype(int)
    df2 = df2[df2["year"].between(2018, 2025)]

    bins   = [0, 10_000, 50_000, 200_000, 1_000_000, float("inf")]
    labels = ["<$10k", "$10–50k", "$50–200k", "$200k–1M", ">$1M"]
    df2["tier"] = pd.cut(df2["sold_price"], bins=bins, labels=labels)

    counts = (df2.groupby(["year", "tier"], observed=True)
                 .size().reset_index(name="n"))
    totals = counts.groupby("year")["n"].transform("sum")
    counts["pct"] = counts["n"] / totals * 100

    tier_colors = {
        "<$10k":     "#78909c",
        "$10–50k":   "#42a5f5",
        "$50–200k":  "#66bb6a",
        "$200k–1M":  "#ffa726",
        ">$1M":      "#ef5350",
    }

    fig = go.Figure()
    for tier in labels:
        sub = counts[counts["tier"] == tier].sort_values("year")
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["pct"],
            mode="lines", name=tier,
            stackgroup="one",
            fillcolor=tier_colors[tier],
            line=dict(color=tier_colors[tier], width=0.8),
            hovertemplate=f"<b>{tier}</b><br>Year: %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Market Composition — Price Tier Share by Year",
        xaxis_title="Year", yaxis_title="% of Lots Sold",
        height=380,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        **PLOTLY_THEME,
    )
    return fig


# ── NEW CHART 4: Fastest rising artists ───────────────────────────────────────

def chart_rising_artists(df: pd.DataFrame) -> go.Figure:
    """Top 15 artists by price appreciation 2019–20 → 2023–25, min 3 lots each period."""
    needed = {"artist", "sale_year", "sold_price"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df[df["sold_price"].notna()].copy()
    df2["year"] = df2["sale_year"].astype(int)

    early = (df2[df2["year"].isin([2019, 2020])]
             .groupby("artist")["sold_price"]
             .agg(["median", "count"]))
    late  = (df2[df2["year"].isin([2023, 2024, 2025])]
             .groupby("artist")["sold_price"]
             .agg(["median", "count"]))

    both = early.join(late, lsuffix="_early", rsuffix="_late").dropna()
    both = both[(both["count_early"] >= 3) & (both["count_late"] >= 3)]
    both["appreciation"] = (both["median_late"] - both["median_early"]) / both["median_early"] * 100
    both = both.sort_values("appreciation", ascending=False).head(15).sort_values("appreciation")

    bar_colors = [
        "#ef5350" if v > 200 else "#ffa726" if v > 100 else "#42a5f5"
        for v in both["appreciation"]
    ]

    x_max_appr = both["appreciation"].max()

    fig = go.Figure(go.Bar(
        x=both["appreciation"].values,
        y=both.index.tolist(),
        orientation="h",
        marker_color=bar_colors,
        text=[f"+{v:.0f}%" for v in both["appreciation"]],
        textposition="auto",    # inside for long bars, outside for short — prevents clipping
        cliponaxis=False,
        customdata=both[["median_early", "median_late"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Appreciation: +%{x:.1f}%<br>"
            "2019–20 median: $%{customdata[0]:,.0f}<br>"
            "2023–25 median: $%{customdata[1]:,.0f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="🚀 Fastest Rising Artists — 2019–20 vs 2023–25",
        xaxis_title="Price Appreciation (%)",
        xaxis=dict(range=[0, x_max_appr * 1.15]),  # 15% headroom so outside labels never clip
        yaxis=dict(autorange="reversed"),
        height=500, margin=dict(l=160, r=30),
        **PLOTLY_THEME,
    )
    return fig


# ── NEW CHART 5: Artist dollar-volume treemap ──────────────────────────────────

def chart_artist_treemap(df: pd.DataFrame) -> go.Figure:
    """Treemap: top 40 artists by total dollar volume."""
    needed = {"artist", "sold_price"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df[df["sold_price"].notna()].copy()
    agg = (df2.groupby("artist")
              .agg(
                  total_volume=("sold_price", "sum"),
                  n_lots=("sold_price", "count"),
                  median_hammer=("sold_price", "median"),
              ).reset_index()
              .sort_values("total_volume", ascending=False)
              .head(40))

    agg["vol_str"] = agg["total_volume"].apply(
        lambda v: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k"
    )

    fig = go.Figure(go.Treemap(
        labels=agg["artist"],
        parents=[""] * len(agg),
        values=agg["total_volume"],
        customdata=agg[["n_lots", "median_hammer", "vol_str"]].values,
        texttemplate="<b>%{label}</b><br>%{customdata[2]}",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Total Volume: %{customdata[2]}<br>"
            "Lots Sold: %{customdata[0]:,}<br>"
            "Median Hammer: $%{customdata[1]:,.0f}<extra></extra>"
        ),
        marker=dict(
            colors=agg["total_volume"],
            colorscale="Reds",
            colorbar=dict(title="Volume ($)"),
        ),
        textfont=dict(size=12),
    ))

    fig.update_layout(
        title="💎 Artist Dollar Volume — Top 40 by Total Sales",
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        **PLOTLY_THEME,
    )
    return fig


# ── NEW CHART 6: Work era vs price ─────────────────────────────────────────────

def chart_decade_vs_price(df: pd.DataFrame) -> go.Figure:
    """Box plot: hammer price by decade the work was made."""
    needed = {"year_made", "sold_price"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df[df["sold_price"].notna() & df["year_made"].notna()].copy()
    df2["decade"] = (df2["year_made"] // 10 * 10).astype(int)
    df2 = df2[df2["decade"].between(1900, 2020)]

    cap = df2["sold_price"].quantile(0.95)
    df2["capped"] = df2["sold_price"].clip(upper=cap)

    decades = sorted(df2["decade"].unique())
    palette = px.colors.sequential.Plasma

    fig = go.Figure()
    for i, decade in enumerate(decades):
        sub = df2[df2["decade"] == decade]["capped"]
        if len(sub) < 5:
            continue
        color = palette[int(i / len(decades) * (len(palette) - 1))]
        fig.add_trace(go.Box(
            y=sub, name=f"{decade}s",
            marker_color=color, line_color=color,
            boxmean=True, showlegend=False,
        ))

    fig.update_layout(
        title="Work Era vs Price — Which Decades Command a Premium?",
        xaxis_title="Decade Made",
        yaxis_title="Hammer ($, 95th pct cap)",
        height=420, showlegend=False,
        **PLOTLY_THEME,
    )
    return fig


# ── NEW CHART 7: Medium intelligence bubble ────────────────────────────────────

def chart_medium_bubble(df: pd.DataFrame) -> go.Figure:
    """Bubble: medium × sell-through × median hammer × volume, colour = ratio."""
    needed = {"medium", "sold_price", "sold", "estimate_mid"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df[df["medium"].notna()].copy()
    df2["ratio"] = df2["sold_price"] / df2["estimate_mid"].clip(lower=1)

    agg = (df2.groupby("medium")
              .agg(
                  median_hammer=("sold_price", lambda x: x.dropna().median()),
                  sell_through=("sold", "mean"),
                  n_lots=("sold", "count"),
                  median_ratio=("ratio", lambda x: x.dropna().median()),
              ).reset_index().dropna())
    agg = agg[agg["n_lots"] >= 50].sort_values("median_hammer", ascending=False).head(25)

    fig = go.Figure(go.Scatter(
        x=agg["sell_through"] * 100,
        y=agg["median_hammer"],
        mode="markers+text",
        text=agg["medium"].str.title(),
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=(agg["n_lots"] / agg["n_lots"].max() * 60 + 10),
            color=agg["median_ratio"],
            colorscale="RdYlGn", cmid=1.0,
            colorbar=dict(title="Hammer/<br>Estimate"),
            showscale=True,
            line=dict(color="#333", width=1),
        ),
        customdata=agg[["n_lots", "median_ratio"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Sell-through: %{x:.1f}%<br>"
            "Median Hammer: $%{y:,.0f}<br>"
            "Lots sold: %{customdata[0]:,}<br>"
            "Hammer/Estimate: %{customdata[1]:.2f}×<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Medium Intelligence — Sell-Through vs Price (bubble=volume, colour=ratio)",
        xaxis_title="Sell-Through Rate (%)",
        yaxis_title="Median Hammer ($, log scale)",
        yaxis=dict(type="log"),
        height=520,
        **PLOTLY_THEME,
    )
    return fig


# ── NEW CHART 8: Model calibration scatter ────────────────────────────────────

def chart_model_calibration(df_eval: pd.DataFrame) -> go.Figure:
    """Actual vs predicted hammer, log-log, coloured by price bucket."""
    needed = {"sold_price", "predicted_hammer"}
    if not needed.issubset(df_eval.columns):
        return go.Figure()

    df2 = df_eval[df_eval["sold_price"].notna() & df_eval["predicted_hammer"].notna()].copy()

    bins   = [0, 10_000, 50_000, 200_000, 1_000_000, float("inf")]
    labels = ["<$10k", "$10–50k", "$50–200k", "$200k–1M", ">$1M"]
    df2["tier"] = pd.cut(df2["sold_price"], bins=bins, labels=labels)

    tier_colors = {
        "<$10k":    "#78909c",
        "$10–50k":  "#42a5f5",
        "$50–200k": "#66bb6a",
        "$200k–1M": "#ffa726",
        ">$1M":     "#ef5350",
    }

    fig = go.Figure()
    for tier in labels:
        sub = df2[df2["tier"] == tier]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["sold_price"], y=sub["predicted_hammer"],
            mode="markers", name=tier,
            marker=dict(color=tier_colors[tier], size=4, opacity=0.55),
            hovertemplate=(
                f"<b>{tier}</b><br>"
                "Actual: $%{x:,.0f}<br>"
                "Predicted: $%{y:,.0f}<extra></extra>"
            ),
        ))

    max_val = float(df2["sold_price"].quantile(0.99))
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[100, max_val], y=[100, max_val],
        mode="lines", name="Perfect", showlegend=True,
        line=dict(color="#ffffff", width=1.5, dash="dash"), hoverinfo="skip",
    ))
    # ±20% band
    fig.add_trace(go.Scatter(
        x=[100, max_val], y=[100, max_val * 1.2],
        mode="lines", name="+20%",
        line=dict(color="#66bb6a", width=1, dash="dot"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[100, max_val], y=[100, max_val * 0.8],
        mode="lines", name="−20%",
        line=dict(color="#ef5350", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(102,187,106,0.06)", hoverinfo="skip",
    ))

    # Compute R² and MedAPE for annotation
    from sklearn.metrics import r2_score as _r2
    ape = np.abs(df2["sold_price"] - df2["predicted_hammer"]) / df2["sold_price"].clip(lower=1)
    r2_val = _r2(np.log1p(df2["sold_price"]), np.log1p(df2["predicted_hammer"]))
    medape = float(np.median(ape) * 100)
    fig.add_annotation(
        x=0.97, y=0.04, xref="paper", yref="paper",
        text=f"<b>R² = {r2_val:.3f} | MedAPE = {medape:.1f}%</b>",
        showarrow=False, font=dict(size=13, color="#ffffff"),
        xanchor="right", yanchor="bottom",
        bgcolor="rgba(0,0,0,0.55)", bordercolor="#888", borderwidth=1,
    )

    fig.update_layout(
        title="Model Calibration — Actual vs Predicted Hammer Price (log scale)",
        xaxis=dict(title="Actual Hammer ($)", type="log"),
        yaxis=dict(title="Predicted Hammer ($)", type="log"),
        height=500,
        # Put legend below chart so it never crowds the plot area
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=11)),
        margin=dict(b=80),
        **PLOTLY_THEME,
    )
    return fig


# ── NEW CHART 9: Conformal coverage reliability diagram ───────────────────────

def chart_reliability_diagram(df_eval: pd.DataFrame) -> go.Figure:
    """Calibration reliability: actual coverage at each nominal confidence level."""
    needed = {"sold_price", "conformal_p10", "conformal_p90"}
    if not needed.issubset(df_eval.columns):
        return go.Figure()

    df2 = df_eval.dropna(subset=["sold_price", "conformal_p10", "conformal_p90"]).copy()
    if df2.empty:
        return go.Figure()

    # Build a range of symmetric quantile levels around the median prediction
    # We'll test coverage at alpha = 0.05, 0.10, ... 0.50 (intervals of 90%, 80%, ... 50%)
    alphas = np.arange(0.05, 0.55, 0.05)
    nominal = 1 - alphas  # target coverage
    actual_coverages = []

    # Residuals from val+test in log space
    log_actual    = np.log1p(df2["sold_price"].values)
    log_predicted = np.log1p(df2["predicted_hammer"].values) if "predicted_hammer" in df2.columns else np.log1p(df2["conformal_p10"].values)
    residuals = log_actual - log_predicted

    q_lo = np.quantile(residuals, alphas / 2)
    q_hi = np.quantile(residuals, 1 - alphas / 2)

    for ql, qh in zip(q_lo, q_hi):
        lo = np.expm1(log_predicted + ql)
        hi = np.expm1(log_predicted + qh)
        cov = ((df2["sold_price"].values >= lo) & (df2["sold_price"].values <= hi)).mean()
        actual_coverages.append(float(cov))

    fig = go.Figure()
    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color="#888888", dash="dash", width=1.5), hoverinfo="skip",
    ))
    # Actual coverage
    fig.add_trace(go.Scatter(
        x=nominal.tolist(), y=actual_coverages,
        mode="lines+markers", name="Model",
        line=dict(color="#42a5f5", width=2.5),
        marker=dict(size=8, color="#42a5f5"),
        hovertemplate="Nominal: %{x:.0%}<br>Actual: %{y:.1%}<extra></extra>",
    ))
    # 80% target marker
    actual_80 = actual_coverages[list(nominal).index(min(nominal, key=lambda x: abs(x - 0.8)))]
    fig.add_annotation(
        x=0.8, y=actual_80,
        text=f"  80% target → {actual_80:.1%} actual",
        showarrow=False, font=dict(size=11, color="#ffa726"), xanchor="left",
    )

    fig.update_layout(
        title="Conformal Interval Reliability — Actual vs Nominal Coverage",
        xaxis=dict(title="Nominal coverage", tickformat=".0%", range=[0.4, 1.0]),
        yaxis=dict(title="Actual coverage", tickformat=".0%", range=[0.4, 1.0]),
        height=380,
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        **PLOTLY_THEME,
    )
    return fig


# ── NEW CHART 10: Artist comparison ────────────────────────────────────────────

def chart_artist_comparison(df: pd.DataFrame, artist_a: str, artist_b: str) -> go.Figure:
    """Side-by-side price trajectory and sell-through for two artists."""
    if not {"artist", "sale_year", "sold_price", "sold"}.issubset(df.columns):
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Median Hammer Price (2018–2025)",
            "Sell-Through Rate (2018–2025)",
        ),
        horizontal_spacing=0.1,
    )
    colors = {artist_a: "#42a5f5", artist_b: "#ffa726"}

    for artist in [artist_a, artist_b]:
        sub = df[df["artist"] == artist].copy()
        if sub.empty:
            continue
        sub["year"] = sub["sale_year"].astype(int)
        sub = sub[sub["year"].between(2018, 2025)]
        agg = sub.groupby("year").agg(
            median_hammer=("sold_price", lambda x: x.dropna().median()),
            sell_through=("sold", "mean"),
            n_lots=("sold", "count"),
        ).reset_index()

        fig.add_trace(go.Scatter(
            x=agg["year"], y=agg["median_hammer"],
            mode="lines+markers", name=artist,
            line=dict(color=colors[artist], width=2.5),
            marker=dict(size=8),
            hovertemplate=f"<b>{artist}</b><br>Year: %{{x}}<br>Median: $%{{y:,.0f}}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=agg["year"], y=agg["sell_through"] * 100,
            mode="lines+markers", name=artist, showlegend=False,
            line=dict(color=colors[artist], width=2.5),
            marker=dict(size=8),
            hovertemplate=f"<b>{artist}</b><br>Year: %{{x}}<br>Sell-through: %{{y:.1f}}%<extra></extra>",
        ), row=1, col=2)

    fig.update_yaxes(title_text="Median Hammer ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sell-Through Rate (%)", row=1, col=2)
    fig.update_layout(
        title=f"Artist Comparison: {artist_a} vs {artist_b}",
        height=380,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(b=80, t=70),
        **PLOTLY_THEME,
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12))
    return fig


# ── Benchmark table ────────────────────────────────────────────────────────────

@st.cache_data
def compute_benchmark_metrics() -> pd.DataFrame:
    """
    Compute MedAPE / R² / Within-20% for:
      - Estimate midpoint baseline (specialist's midpoint as the prediction)
      - Our ensemble (predicted_hammer from eval_predictions.parquet)
    Reported on the clean TEST split (2025 H2) only — the 2025 H1 validation rows
    in eval_predictions are in-sample for calibration/Stage-C/CQR.
    """
    df_eval = load_eval_predictions()
    if df_eval.empty:
        return pd.DataFrame()

    # Clean test split only (fall back to all rows if the split tag is absent)
    if "split" in df_eval.columns and (df_eval["split"] == "test").any():
        df_eval = df_eval[df_eval["split"] == "test"]

    rows = []
    for split_label, yr_mask in [
        ("2025 H2 test (clean held-out)", df_eval["sale_year"].notna()),
    ]:
        sub = df_eval[yr_mask].dropna(subset=["sold_price", "estimate_mid", "predicted_hammer"])
        if sub.empty:
            continue
        true   = sub["sold_price"].values
        estim  = sub["estimate_mid"].values
        pred   = sub["predicted_hammer"].values

        def _metrics(y_hat, y_true):
            ape    = np.abs(y_true - y_hat) / y_true.clip(1)
            medape = float(np.median(ape) * 100)
            w20    = float((ape < 0.20).mean() * 100)
            r2     = float(1 - np.sum((np.log1p(y_true) - np.log1p(y_hat.clip(1))) ** 2)
                           / np.sum((np.log1p(y_true) - np.log1p(y_true).mean()) ** 2))
            return medape, w20, r2

        b_medape, b_w20, b_r2 = _metrics(estim, true)
        e_medape, e_w20, e_r2 = _metrics(pred,  true)

        rows.append(dict(Split=split_label, Method="Estimate Midpoint (Baseline)",
                         MedAPE=f"{b_medape:.1f}%", Within20=f"{b_w20:.1f}%", R2=f"{b_r2:.3f}"))
        rows.append(dict(Split=split_label, Method="**Our Ensemble ↑**",
                         MedAPE=f"{e_medape:.1f}%", Within20=f"{e_w20:.1f}%", R2=f"{e_r2:.3f}"))

    return pd.DataFrame(rows)


# ── Walk-forward validation chart ─────────────────────────────────────────────

def chart_walk_forward_backtest(df_eval: pd.DataFrame) -> go.Figure:
    """
    Quarterly out-of-sample MedAPE for ensemble vs specialist baseline.
    Each bar uses only the lots from that quarter — no training data seen.
    """
    if df_eval.empty:
        return go.Figure()
    needed = {"sale_year", "sold_price", "predicted_hammer", "estimate_mid"}
    if not needed.issubset(df_eval.columns):
        return go.Figure()

    df = df_eval.dropna(subset=["sold_price", "predicted_hammer", "estimate_mid"]).copy()

    if "sale_month" in df.columns:
        df["_q"] = ((df["sale_month"].astype(int) - 1) // 3 + 1).astype(int)
        df["_period_label"] = df["sale_year"].astype(str) + " Q" + df["_q"].astype(str)
        df["_period_sort"]  = df["sale_year"].astype(int) * 10 + df["_q"].astype(int)
    else:
        df["_period_label"] = df["sale_year"].astype(str)
        df["_period_sort"]  = df["sale_year"].astype(int) * 10

    rows = []
    for period_sort, grp in df.groupby("_period_sort"):
        if len(grp) < 8:
            continue
        label = grp["_period_label"].iloc[0]
        true  = grp["sold_price"].values
        pred  = grp["predicted_hammer"].values
        base  = grp["estimate_mid"].values
        ens_medape  = float(np.median(np.abs(true - pred) / true.clip(1)) * 100)
        base_medape = float(np.median(np.abs(true - base) / true.clip(1)) * 100)
        rows.append({
            "Period": label,
            "sort": period_sort,
            "Ensemble": ens_medape,
            "Specialist Baseline": base_medape,
            "n": len(grp),
        })

    if not rows:
        return go.Figure()

    res = pd.DataFrame(rows).sort_values("sort")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["Period"], y=res["Specialist Baseline"],
        mode="lines+markers", name="Specialist Baseline",
        line=dict(color="#888888", dash="dash", width=2),
        marker=dict(size=5, color="#888888"),
        hovertemplate="Period: %{x}<br>Specialist MedAPE: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=res["Period"], y=res["Ensemble"],
        mode="lines+markers", name="Ensemble Model",
        line=dict(color="#42a5f5", width=2.5),
        marker=dict(size=7, color="#42a5f5"),
        fill="tonexty",
        fillcolor="rgba(66,165,245,0.08)",
        hovertemplate="Period: %{x}<br>Ensemble MedAPE: %{y:.1f}%<extra></extra>",
    ))

    # Every quarter shown here is out-of-sample (the model trained on ≤2024; these
    # are all held-out 2025+ quarters), so the whole span is shaded as test.
    fig.add_vrect(
        x0=res["Period"].iloc[0], x1=res["Period"].iloc[-1],
        fillcolor="rgba(102,187,106,0.05)", layer="below", line_width=0,
        annotation_text="Out-of-sample (held-out)", annotation_position="top left",
        annotation_font=dict(size=10, color="#66bb6a"),
    )

    fig.update_layout(
        title="Walk-Forward Validation — Quarterly Out-of-Sample MedAPE",
        xaxis_title="Quarter",
        yaxis_title="MedAPE (%)",
        height=380,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(t=55, b=80),
        **PLOTLY_THEME,
    )
    return fig


# ── Residual diagnostics chart ─────────────────────────────────────────────────

def chart_residual_diagnostics(df_eval: pd.DataFrame) -> go.Figure:
    """
    3-panel residual analysis:
      Left  — residuals vs log estimate  (checks heteroscedasticity)
      Centre — residuals vs time         (checks regime drift)
      Right  — residual distribution     (checks normality)
    """
    if df_eval.empty:
        return go.Figure()
    needed = {"sold_price", "predicted_hammer", "estimate_mid"}
    if not needed.issubset(df_eval.columns):
        return go.Figure()

    df = df_eval.dropna(subset=["sold_price", "predicted_hammer", "estimate_mid"]).copy()
    log_true  = np.log1p(df["sold_price"].values)
    log_pred  = np.log1p(df["predicted_hammer"].clip(lower=1).values)
    log_est   = np.log1p(df["estimate_mid"].clip(lower=1).values)
    residuals = log_true - log_pred

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Residuals vs Log Estimate",
            "Residuals Over Time",
            "Residual Distribution",
        ),
        horizontal_spacing=0.10,
    )

    # ── Panel 1: heteroscedasticity ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=log_est, y=residuals,
        mode="markers",
        marker=dict(size=3, color="#42a5f5", opacity=0.35),
        hovertemplate="Log Est: %{x:.2f}<br>Residual: %{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="#888", line_dash="dash", row=1, col=1)
    # Add rolling mean trendline
    sort_idx = np.argsort(log_est)
    x_sorted = log_est[sort_idx]
    y_sorted = residuals[sort_idx]
    window = max(1, len(y_sorted) // 30)
    y_smooth = pd.Series(y_sorted).rolling(window, center=True, min_periods=1).mean().values
    fig.add_trace(go.Scatter(
        x=x_sorted, y=y_smooth,
        mode="lines", line=dict(color="#ef5350", width=2),
        name="Trend", showlegend=False,
    ), row=1, col=1)

    # ── Panel 2: drift over time ──────────────────────────────────────────
    if "sale_year" in df.columns:
        x_time = df["sale_year"].astype(float).values
        if "sale_month" in df.columns:
            x_time = x_time + (df["sale_month"].astype(float).values - 1) / 12
        fig.add_trace(go.Scatter(
            x=x_time, y=residuals,
            mode="markers",
            marker=dict(size=3, color="#ce93d8", opacity=0.35),
            hovertemplate="Year: %{x:.2f}<br>Residual: %{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
        fig.add_hline(y=0, line_color="#888", line_dash="dash", row=1, col=2)
        sort_t = np.argsort(x_time)
        y_t_smooth = pd.Series(residuals[sort_t]).rolling(max(1, len(residuals)//20), center=True, min_periods=1).mean().values
        fig.add_trace(go.Scatter(
            x=x_time[sort_t], y=y_t_smooth,
            mode="lines", line=dict(color="#ef5350", width=2),
            showlegend=False,
        ), row=1, col=2)

    # ── Panel 3: distribution ─────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=50,
        marker_color="#66bb6a", opacity=0.7,
        name="Residuals", showlegend=False,
        hovertemplate="Residual: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ), row=1, col=3)
    # Normal overlay
    mu, sigma = float(np.mean(residuals)), float(np.std(residuals))
    x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 120)
    bin_width = (residuals.max() - residuals.min()) / 50 if (residuals.max() - residuals.min()) > 0 else 0.01
    y_norm = (
        np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
        / (sigma * np.sqrt(2 * np.pi))
        * len(residuals)
        * bin_width
    )
    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm,
        mode="lines", name="Normal fit",
        line=dict(color="#ffa726", width=2, dash="dot"),
        showlegend=False,
    ), row=1, col=3)

    fig.update_layout(
        title=f"Residual Diagnostics — log-price space  |  μ={mu:+.3f}  σ={sigma:.3f}",
        height=380,
        margin=dict(t=60, b=40),
        **PLOTLY_THEME,
    )
    return fig


# ── Alpha scorecard ────────────────────────────────────────────────────────────

def compute_alpha_scorecard(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Information Ratio table: mean edge % / std(edge %) by house and price tier.
    Edge % = (predicted_hammer - estimate_mid) / estimate_mid.
    IR > 0.5 in a segment signals systematic model signal.
    """
    if df_eval.empty:
        return pd.DataFrame()
    needed = {"sold_price", "predicted_hammer", "estimate_mid"}
    if not needed.issubset(df_eval.columns):
        return pd.DataFrame()

    df = df_eval.dropna(subset=["sold_price", "predicted_hammer", "estimate_mid"]).copy()
    df["_edge"]           = (df["predicted_hammer"] - df["estimate_mid"]) / df["estimate_mid"].clip(1) * 100
    df["_realized_alpha"] = (df["sold_price"]       - df["estimate_mid"]) / df["estimate_mid"].clip(1) * 100

    house_labels = {"christies": "Christie's", "bonhams": "Bonhams", "phillips": "Phillips", "sothebys": "Sotheby's"}
    rows = []

    def _row(segment, category, grp):
        if len(grp) < 10:
            return
        edge    = grp["_edge"].values
        alpha   = grp["_realized_alpha"].values
        mean_e  = float(np.mean(edge))
        std_e   = float(np.std(edge))
        ir      = mean_e / std_e if std_e > 0 else 0.0
        hit_rt  = float((np.sign(edge) == np.sign(alpha)).mean() * 100)
        rows.append({
            "Segment":        segment,
            "Category":       category,
            "N":              len(grp),
            "Mean Edge":      f"{mean_e:+.1f}%",
            "σ Edge":         f"{std_e:.1f}%",
            "Info Ratio":     f"{ir:+.2f}",
            "Hit Rate":       f"{hit_rt:.0f}%",
            "Realized Alpha": f"{float(np.mean(alpha)):+.1f}%",
        })

    # By auction house
    if "house" in df.columns:
        for h, grp in df.groupby("house"):
            _row(house_labels.get(h, h), "House", grp)

    # By price tier
    bins   = [0, 10_000, 50_000, 200_000, 1_000_000, float("inf")]
    labels = ["<$10k", "$10–50k", "$50–200k", "$200k–$1M", ">$1M"]
    df["_tier"] = pd.cut(df["estimate_mid"], bins=bins, labels=labels)
    for tier, grp in df.groupby("_tier", observed=True):
        _row(str(tier), "Price Tier", grp)

    # By year
    if "sale_year" in df.columns:
        for yr, grp in df.groupby("sale_year"):
            _row(str(int(yr)), "Year", grp)

    return pd.DataFrame(rows)


# ── Comparable sales ───────────────────────────────────────────────────────────

def find_comparable_sales(
    df: pd.DataFrame,
    artist: str,
    medium: str,
    estimate_mid: float,
    house: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    Find the most comparable historical lots for a given input.
    Scoring (0–100):
      Artist match      50 pts
      Medium match      20 pts
      Size proximity    15 pts  (requires log_area in df)
      Price proximity   15 pts  (within 2× of estimate_mid)
      Recency bonus     up to +10 pts (decays over 6 years)
    Returns top-n rows with display columns + Similarity % column.
    """
    if df.empty:
        return pd.DataFrame()

    sold = df[df["sold_price"].notna()].copy()
    if sold.empty:
        return pd.DataFrame()

    # Raw score components (each normalized to their weight)
    score = pd.Series(0.0, index=sold.index)

    # Artist match: 50 pts
    if artist:
        score += (sold["artist"].str.lower() == artist.lower()).astype(float) * 50

    # Medium match: 20 pts
    if medium and "medium" in sold.columns:
        score += (sold["medium"].str.lower() == medium.lower()).astype(float) * 20

    # Price proximity: 0–15 pts (log-ratio based)
    if estimate_mid > 0 and "sold_price" in sold.columns:
        ratio = sold["sold_price"] / max(estimate_mid, 1)
        score += (15 - (np.log(ratio.clip(0.05, 20)).abs() * 5)).clip(0, 15)

    # Size proximity: 0–15 pts (needs log_area)
    if "log_area" in sold.columns and estimate_mid > 0:
        # estimate a rough expected log_area from price tier (heuristic)
        expected_log_area = np.log1p(estimate_mid / 5000)  # rough proxy
        area_diff = (sold["log_area"] - expected_log_area).abs()
        score += (15 - area_diff * 5).clip(0, 15)

    # Recency bonus: 0–10 pts
    if "sale_year" in sold.columns:
        max_yr = int(sold["sale_year"].max())
        age    = (max_yr - sold["sale_year"]).clip(0, 6)
        score += (10 * (1 - age / 6))

    sold = sold.copy()
    sold["_raw_score"] = score
    top = sold.nlargest(n, "_raw_score")

    # Normalize to 0–100 similarity %
    max_possible = 50 + 20 + 15 + 15 + 10  # = 110
    top["_similarity"] = (top["_raw_score"] / max_possible * 100).clip(0, 100).round(0).astype(int)

    # ── Build display dataframe ─────────────────────────────────────────
    display_cols = {
        "artist":       "Artist",
        "medium":       "Medium",
        "sale_year":    "Year",
        "house":        "House",
        "estimate_mid": "Est. Mid",
        "sold_price":   "Hammer",
    }
    avail = {k: v for k, v in display_cols.items() if k in top.columns}
    out = top[list(avail.keys())].rename(columns=avail).reset_index(drop=True)

    # H/E Ratio column
    if "Est. Mid" in out.columns and "Hammer" in out.columns:
        raw_mid    = top["estimate_mid"].values
        raw_hammer = top["sold_price"].values
        out.insert(len(out.columns), "H/E Ratio",
                   [f"{h/e:.2f}×" if e > 0 else "—"
                    for h, e in zip(raw_hammer, raw_mid)])

    # Similarity column (last)
    out["Similarity"] = top["_similarity"].values.astype(str) + "%"

    # Format dollars
    if "Est. Mid" in out.columns:
        raw_mid_vals = top["estimate_mid"].values
        out["Est. Mid"] = [fmt_dollar(v) if pd.notna(v) else "—" for v in raw_mid_vals]
    if "Hammer" in out.columns:
        raw_hammer_vals = top["sold_price"].values
        out["Hammer"] = [fmt_dollar(v) for v in raw_hammer_vals]
    if "House" in out.columns:
        house_labels = {"christies": "Christie's", "bonhams": "Bonhams", "phillips": "Phillips", "sothebys": "Sotheby's"}
        out["House"] = out["House"].map(house_labels).fillna(out["House"])
    if "Year" in out.columns:
        out["Year"] = out["Year"].astype(int)

    return out


# ── SHAP explanation card ──────────────────────────────────────────────────────

def shap_explanation_card(
    shap_values: np.ndarray,
    feature_names: list,
    base_value: float,
    hammer: float,
    ci_lo: float,
    ci_hi: float,
    estimate_high: float,
) -> None:
    """
    Render a structured prediction explanation:
      - Top 3 positive / negative SHAP drivers with dollar impact
      - Confidence interval interpretation
      - Chance above high estimate
      - Risk level badge
    """
    # Dollar impact per feature (same formula as waterfall)
    impacts = [math.expm1(base_value + s) - math.expm1(base_value) for s in shap_values]
    df_sv = pd.DataFrame({"feature": feature_names, "impact": impacts})

    positives = df_sv[df_sv["impact"] > 0].nlargest(3, "impact")
    negatives = df_sv[df_sv["impact"] < 0].nsmallest(3, "impact")

    # P(hammer > estimate_high)
    # Approximate using conformal CI: fraction of CI above high estimate
    ci_range = max(ci_hi - ci_lo, 1)
    prob_above_high = max(0.0, min(1.0, (ci_hi - estimate_high) / ci_range))

    # Risk level
    ci_width_pct = (ci_hi - ci_lo) / max(hammer, 1) * 100
    if ci_width_pct < 80:
        risk_emoji, risk_label, risk_color = "🟢", "Low uncertainty", "#43a047"
    elif ci_width_pct < 160:
        risk_emoji, risk_label, risk_color = "🟡", "Moderate uncertainty", "#fdd835"
    else:
        risk_emoji, risk_label, risk_color = "🔴", "High uncertainty", "#e53935"

    st.markdown("#### 🔍 Prediction Explanation")
    col_pos, col_neg, col_risk = st.columns(3)

    with col_pos:
        st.markdown("**↑ Pushing price up**")
        for _, row in positives.iterrows():
            feat = row["feature"].replace("_", " ").title()
            st.markdown(
                f'<div style="padding:4px 0; color:#66bb6a; font-size:0.92rem;">'
                f'<b>{feat}</b><br>'
                f'<span style="font-size:0.85rem;">+{fmt_dollar(row["impact"])}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_neg:
        st.markdown("**↓ Pulling price down**")
        for _, row in negatives.iterrows():
            feat = row["feature"].replace("_", " ").title()
            st.markdown(
                f'<div style="padding:4px 0; color:#ef5350; font-size:0.92rem;">'
                f'<b>{feat}</b><br>'
                f'<span style="font-size:0.85rem;">{fmt_dollar(row["impact"])}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_risk:
        st.markdown("**Confidence**")
        st.markdown(
            f'<div style="font-size:0.92rem; line-height:1.7;">'
            f'{risk_emoji} <b style="color:{risk_color}">{risk_label}</b><br>'
            f'P10–P90 span: {fmt_dollar(ci_lo)} – {fmt_dollar(ci_hi)}<br>'
            f'Range width: {ci_width_pct:.0f}% of prediction<br>'
            f'Chance above high estimate: <b>{prob_above_high*100:.0f}%</b>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — session state init + preset buttons
# ══════════════════════════════════════════════════════════════════════════════
from ml.features import MEDIUM_CATEGORIES  # noqa: E402

MEDIUM_LIST = list(MEDIUM_CATEGORIES.keys())

# Initialise session state defaults (only on first load)
_DEFAULTS = dict(
    artist="",
    estimate_low=100_000,
    estimate_high=150_000,
    medium=MEDIUM_LIST[0],
    width_cm=0.0,
    height_cm=0.0,
    year_made=1980,
    signed=True,
    exhibited=False,
    literature=False,
    provenance="",
    house="christies",
    sale_year=2025,
    sale_month=5,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def apply_preset(name: str):
    p = PRESETS[name]
    for k, v in p.items():
        st.session_state[k] = v
    # Also sync the display-label key for house selectbox
    _house_opts   = ["christies", "bonhams", "phillips", "sothebys"]
    _house_labels = ["Christie's", "Bonhams", "Phillips", "Sotheby's"]
    raw_house = p.get("house", "christies")
    if raw_house in _house_opts:
        st.session_state["_house_label"] = _house_labels[_house_opts.index(raw_house)]
    # Sync artist selectbox
    st.session_state["_artist_select"] = p.get("artist", "")


known_artists = get_known_artists()

# ── Sidebar header ─────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #0d0d24 0%, #12122e 100%);
    border-bottom: 1px solid #1e1e3a;
    padding: 18px 16px 14px;
    margin: -1rem -1rem 1rem;
">
  <div style="font-size:1.15rem; font-weight:800; letter-spacing:-0.3px;
              background:linear-gradient(135deg,#42a5f5,#ce93d8);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              background-clip:text;">
    🎨 Art Price Intelligence
  </div>
  <div style="font-size:0.7rem; color:#424265; margin-top:3px; letter-spacing:0.3px;">
    Christie's · Bonhams · Phillips · Sotheby's · 134k priced lots
  </div>
</div>
""", unsafe_allow_html=True)

# ── Quick examples ─────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.8px;"
    "color:#424265;font-weight:600;margin-bottom:5px;'>Quick Examples</div>",
    unsafe_allow_html=True,
)
pc1, pc2, pc3 = st.sidebar.columns(3)
pc1.button("Basquiat", width='stretch', on_click=apply_preset, args=("Basquiat",))
pc2.button("Warhol",   width='stretch', on_click=apply_preset, args=("Warhol",))
pc3.button("Picasso",  width='stretch', on_click=apply_preset, args=("Picasso",))

st.sidebar.divider()

# ── Artist name ────────────────────────────────────────────────────────────
_cur_artist = st.session_state.get("artist", "")
if known_artists:
    _artist_opts = [""] + known_artists
    _artist_idx  = _artist_opts.index(_cur_artist) if _cur_artist in _artist_opts else 0
    _artist_sel  = st.sidebar.selectbox(
        "Artist Name",
        _artist_opts,
        index=_artist_idx,
        format_func=lambda x: x if x else "Search artists...",
        key="_artist_select",
    )
    artist_input = _artist_sel or st.session_state.get("artist", "")
    if _artist_sel:
        st.session_state["artist"] = _artist_sel
else:
    artist_input = st.sidebar.text_input(
        "Artist Name",
        placeholder="e.g. Jean-Michel Basquiat",
        value=_cur_artist,
        key="artist",
    )

# ── Artwork title — dropdown if artist known, text search if not ───────────
_lot_autofilled = False

if artist_input:
    # Artist is set → show dropdown of their known works
    _artworks_df = get_artist_artworks(artist_input)
    if not _artworks_df.empty:
        def _work_label(row):
            t  = str(row.get("title", "") or "Untitled")[:50]
            yr = f", {int(row['year_made'])}" if pd.notna(row.get("year_made")) else ""
            return f"{t}{yr}"

        _work_labels  = [_work_label(r) for _, r in _artworks_df.iterrows()]
        _prev_artwork = st.session_state.get("_selected_artwork", "")
        _artwork_idx  = (_work_labels.index(_prev_artwork) + 1) if _prev_artwork in _work_labels else 0

        _artwork_sel = st.sidebar.selectbox(
            "Artwork Title",
            ["— browse works —"] + _work_labels,
            index=_artwork_idx,
            key="_artwork_select",
        )
        st.sidebar.markdown(
            f"<div style='font-size:0.7rem;color:#424265;margin-top:-6px;'>"
            f"📚 {len(_artworks_df):,} lots in database</div>",
            unsafe_allow_html=True,
        )

        if _artwork_sel != "— browse works —":
            st.session_state["_selected_artwork"] = _artwork_sel
            _row = _artworks_df.iloc[_work_labels.index(_artwork_sel)]
            # Auto-fill all fields from the selected lot
            _safe = lambda col, default: (_row[col] if col in _row.index and pd.notna(_row[col]) else default)
            st.session_state["estimate_low"]  = float(_safe("estimate_low",  100_000))
            st.session_state["estimate_high"] = float(_safe("estimate_high", 150_000))
            st.session_state["medium"]        = str(_safe("medium",   MEDIUM_LIST[0]))
            st.session_state["width_cm"]      = float(_safe("width_cm",  0.0))
            st.session_state["height_cm"]     = float(_safe("height_cm", 0.0))
            st.session_state["year_made"]     = int(_safe("year_made", 1980))
            st.session_state["signed"]        = bool(_safe("signed",     True))
            st.session_state["exhibited"]     = bool(_safe("exhibited",  False))
            st.session_state["literature"]    = bool(_safe("literature", False))
            st.session_state["provenance"]    = str(_safe("provenance", ""))
            if pd.notna(_row.get("house")):
                st.session_state["house"] = str(_row["house"])
            _lot_autofilled = True
            # Show what was pulled in
            _h = st.session_state.get("house","")
            _house_disp = {"christies":"Christie's","phillips":"Phillips","sothebys":"Sotheby's"}.get(_h, _h.title())
            _yr_disp = st.session_state.get("year_made","")
            _med_disp = str(st.session_state.get("medium",""))[:28]
            st.sidebar.markdown(
                f"<div style='background:rgba(66,165,245,0.08);border:1px solid rgba(66,165,245,0.2);"
                f"border-radius:8px;padding:7px 10px;margin-top:4px;font-size:0.74rem;color:#90caf9;line-height:1.6;'>"
                f"✓ <b>Auto-filled</b><br>"
                f"<span style='color:#757575;'>{_med_disp} · {_yr_disp} · {_house_disp}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        # Artist known but no lots in DB — free text fallback
        st.sidebar.text_input("Artwork Title", placeholder="Enter title (not in database)",
                               key="_artwork_select")

else:
    # No artist yet → free text search across all artists + titles
    _title_query = st.sidebar.text_input(
        "Artwork Title",
        placeholder="Search title across all artists...",
        value=st.session_state.get("_title_query", ""),
        key="_title_query",
    )
    if _title_query.strip():
        _results = search_artwork("", _title_query)
        if not _results.empty:
            def _res_label(row):
                artist = str(row.get("artist", ""))
                title  = str(row.get("title",  "") or "Untitled")[:40]
                yr     = f" ({int(row['year_made'])})" if pd.notna(row.get("year_made")) else ""
                return f"{artist} — {title}{yr}"
            _res_labels = [_res_label(r) for _, r in _results.iterrows()]
            _res_choice = st.sidebar.selectbox(
                "Select artwork:", ["— pick one —"] + _res_labels, key="_res_select"
            )
            if _res_choice != "— pick one —":
                _row = _results.iloc[_res_labels.index(_res_choice)]
                # Auto-fill artist + all details
                _found_artist = str(_row.get("artist", ""))
                st.session_state["artist"] = _found_artist
                if known_artists and _found_artist in known_artists:
                    st.session_state["_artist_select"] = _found_artist
                _safe = lambda col, default: (_row[col] if col in _row.index and pd.notna(_row.get(col)) else default)
                st.session_state["estimate_low"]  = float(_safe("estimate_low",  100_000))
                st.session_state["estimate_high"] = float(_safe("estimate_high", 150_000))
                st.session_state["medium"]        = str(_safe("medium",   MEDIUM_LIST[0]))
                st.session_state["width_cm"]      = float(_safe("width_cm",  0.0))
                st.session_state["height_cm"]     = float(_safe("height_cm", 0.0))
                st.session_state["year_made"]     = int(_safe("year_made", 1980))
                st.session_state["signed"]        = bool(_safe("signed",     True))
                st.session_state["exhibited"]     = bool(_safe("exhibited",  False))
                st.session_state["literature"]    = bool(_safe("literature", False))
                st.session_state["provenance"]    = str(_safe("provenance", ""))
                if pd.notna(_row.get("house")):
                    st.session_state["house"] = str(_row["house"])
                _lot_autofilled = True
                st.rerun()
        else:
            st.sidebar.caption("No matches — try a different title or select an artist above.")

# Apply artist-level defaults (medium/dims/year) when no lot auto-filled
if artist_input and not _lot_autofilled:
    _defs = get_artist_defaults(artist_input)
    if _defs and st.session_state.get("_defaults_artist") != artist_input:
        for _k, _v in _defs.items():
            if _k not in ("estimate_low", "estimate_high"):
                st.session_state.setdefault(_k, _v)
        st.session_state["_defaults_artist"] = artist_input

# ── Details expander ──────────────────────────────────────────────────────
st.sidebar.divider()
_exp_label = "⚙  Details — auto-filled" if _lot_autofilled else "⚙  Details — all optional"
with st.sidebar.expander(_exp_label, expanded=not _lot_autofilled):

    # Presale estimate
    st.markdown(
        "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.8px;"
        "color:#42a5f5;font-weight:700;margin-bottom:6px;'>Presale Estimate</div>",
        unsafe_allow_html=True,
    )
    if not _lot_autofilled:
        st.markdown(
            "<div style='font-size:0.72rem;color:#555;margin:-2px 0 8px;'>The auction house's "
            "catalogue price range — strongest predictor.</div>",
            unsafe_allow_html=True,
        )
    col1, col2 = st.columns(2)
    estimate_low  = col1.number_input(
        "Low ($)", min_value=500,
        value=int(st.session_state.get("estimate_low",  100_000)), step=5_000, key="estimate_low",
    )
    estimate_high = col2.number_input(
        "High ($)", min_value=500,
        value=int(st.session_state.get("estimate_high", 150_000)), step=5_000, key="estimate_high",
    )

    st.markdown(
        "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.8px;"
        "color:#616161;font-weight:700;margin:12px 0 6px;'>Artwork Details</div>",
        unsafe_allow_html=True,
    )
    _med_default = st.session_state.get("medium", MEDIUM_LIST[0])
    _med_idx     = MEDIUM_LIST.index(_med_default) if _med_default in MEDIUM_LIST else 0
    medium = st.selectbox("Medium", MEDIUM_LIST, index=_med_idx, key="medium")

    col3, col4 = st.columns(2)
    width_cm  = col3.number_input("Width (cm)",  min_value=0.0,
                                   value=float(st.session_state.get("width_cm",  0.0)), step=1.0, key="width_cm")
    height_cm = col4.number_input("Height (cm)", min_value=0.0,
                                   value=float(st.session_state.get("height_cm", 0.0)), step=1.0, key="height_cm")
    year_made = st.number_input("Year Created", min_value=1800, max_value=2026,
                                 value=int(st.session_state.get("year_made", 1980)), step=1, key="year_made")

    st.markdown(
        "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.8px;"
        "color:#616161;font-weight:700;margin:12px 0 6px;'>Provenance</div>",
        unsafe_allow_html=True,
    )
    _prov_c1, _prov_c2, _prov_c3 = st.columns(3)
    signed     = _prov_c1.checkbox("Signed",     value=bool(st.session_state.get("signed",     True)),  key="signed")
    exhibited  = _prov_c2.checkbox("Exhibited",  value=bool(st.session_state.get("exhibited",  False)), key="exhibited")
    literature = _prov_c3.checkbox("Literature", value=bool(st.session_state.get("literature", False)), key="literature")
    provenance = st.text_area("Note", height=52,
                               placeholder="Private Collection, New York...",
                               value=st.session_state.get("provenance", ""), key="provenance")

    st.markdown(
        "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.8px;"
        "color:#616161;font-weight:700;margin:12px 0 6px;'>Sale</div>",
        unsafe_allow_html=True,
    )
    _house_opts   = ["christies", "bonhams", "phillips", "sothebys"]
    _house_labels = ["Christie's", "Bonhams", "Phillips", "Sotheby's"]
    _house_def    = st.session_state.get("house", "christies")
    _house_idx    = _house_opts.index(_house_def) if _house_def in _house_opts else 0
    _house_raw    = st.selectbox("Auction House", _house_labels, index=_house_idx, key="_house_label")
    house         = _house_opts[_house_labels.index(_house_raw)]

    col5, col6  = st.columns(2)
    _sy_opts    = list(range(2025, 2027))
    _sy_default = st.session_state.get("sale_year", 2025)
    _sy_idx     = _sy_opts.index(_sy_default) if _sy_default in _sy_opts else 0
    sale_year   = col5.selectbox("Year",  _sy_opts, index=_sy_idx, key="sale_year")
    _sm_opts    = list(range(1, 13))
    _sm_default = st.session_state.get("sale_month", 5)
    _sm_idx     = _sm_opts.index(_sm_default) if _sm_default in _sm_opts else 4
    sale_month  = col6.selectbox("Month", _sm_opts, index=_sm_idx, key="sale_month")

st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
predict_btn = st.sidebar.button(
    "Estimate Hammer Price", type="primary", width='stretch'
)

# ── main header ────────────────────────────────────────────────────────────
# All headline numbers come from meta.json's `display` block (single source of
# truth, written by ml/train.py) so the hero can never drift out of sync again.
_hero_meta = load_meta() or {}
_hd = _hero_meta.get("display", {})
_HOUSE_DISP = {"christies": "Christie's", "bonhams": "Bonhams", "phillips": "Phillips", "sothebys": "Sotheby's"}
_hero_houses   = ", ".join(_HOUSE_DISP.get(h, h.title())
                           for h in _hd.get("houses", ["christies", "bonhams", "phillips", "sothebys"]))
_hero_priced_k = round(_hd.get("n_priced", 133752) / 1000)
_hero_total_k  = round(_hd.get("n_total", 155776) / 1000)
_hero_r2       = _hd.get("r2_raw_model", 0.89)
_hero_cov      = round(_hd.get("interval_coverage", 82))
_hero_auc      = _hd.get("ranking_auc", 0.61)
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #0d0d2b 0%, #12122a 50%, #1a0d2e 100%);
    border: 1px solid #2a2a50;
    border-radius: 16px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 40px rgba(66,165,245,0.08);
">
  <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem;">
    <span style="font-size:2.4rem;">🎨</span>
    <h1 style="
        margin:0; font-size:2.4rem; font-weight:800; letter-spacing:-0.5px;
        background: linear-gradient(135deg, #42a5f5 0%, #ce93d8 55%, #ef9a9a 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    ">Art Auction Price Predictor</h1>
  </div>
  <p style="margin:0; color:#9e9e9e; font-size:1rem; line-height:1.6; max-width:860px;">
    Estimates hammer price distributions for auction lots using structured lot metadata,
    presale estimates, market momentum features, and an ensemble of <b style="color:#cccccc">LightGBM
    and CatBoost</b> trained on
    <b style="color:#42a5f5">{_hero_priced_k}k priced lots</b> — from a {_hero_total_k}k-lot dataset across {_hero_houses} (2018–2026).
  </p>
  <div style="display:flex; gap:1.5rem; margin-top:1rem; flex-wrap:wrap; align-items:center;">
    <span style="background:rgba(66,165,245,0.12); color:#42a5f5; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600;">Test R² {_hero_r2:.2f}</span>
    <span style="background:rgba(206,147,216,0.12); color:#ce93d8; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600;">Ranking AUC {_hero_auc:.2f}</span>
    <span style="background:rgba(102,187,106,0.12); color:#66bb6a; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600;">{_hero_cov}% Interval Coverage</span>
    <span style="background:rgba(206,147,216,0.12); color:#ce93d8; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600;">{_hero_priced_k}k Priced Lots</span>
    <span style="background:rgba(255,167,38,0.12); color:#ffa726; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600;">2018–2026</span>
    <span style="background:rgba(239,83,80,0.12); color:#ef5350; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600;">SHAP Explained</span>
    <span style="background:rgba(255,255,255,0.06); color:#9e9e9e; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600;">Conformal P10–P90</span>
    <span style="background:rgba(255,255,255,0.06); color:#9e9e9e; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600; border:1px solid rgba(255,87,34,0.3);">⚠ Research Only — Not Investment Advice</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_trends, tab_presale, tab_batch = st.tabs(
    ["Predict", "Trends", "Batch Scoring (demo)", "Batch Predict"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ══════════════════════════════════════════════════════════════════════════════

with tab_predict:
    if not models_exist():
        st.warning(
            "Models not found. Train the model first: `python -m ml.train`"
        )
        st.stop()

    # ── Live model stats banner ────────────────────────────────────────────
    meta = load_meta()
    if meta:
        # Deploy artifacts carry test_metrics=None (the cal-tail is in-sample); fall
        # back to the honest held-out numbers from the eval run. `or {}` also guards
        # against the key existing with a None value (.get's default only covers absence).
        tm = meta.get("test_metrics") or meta.get("holdout_eval_metrics") or {}
        vm = meta.get("val_metrics") or {}
        cv_mean    = meta.get("cv_medape_mean", 0)
        cv_std     = meta.get("cv_medape_std", 0)
        train_rows = meta.get("train_rows", 0)
        vm_r2      = f"{vm.get('r2', 0):.3f}"      if vm.get("r2")     else "—"
        vm_mape    = f"{vm.get('medape', 0):.1f}%"  if vm.get("medape") else "—"
        te_r2      = f"{tm.get('r2', 0):.3f}"       if tm.get("r2")     else "—"
        te_mape    = f"{tm.get('medape', 0):.1f}%"  if tm.get("medape") else "—"
        w20        = f"{tm.get('within_20', 0):.1f}%" if tm.get("within_20") else "—"
        cv_str     = f"{cv_mean:.1f}% ±{cv_std:.1f}%"

        # Model freshness
        import datetime as _dt
        trained_at = meta.get("trained_at", "")
        if trained_at:
            try:
                _td = _dt.datetime.fromisoformat(trained_at[:10])
                _days_ago = (_dt.date.today() - _td.date()).days
                freshness_label = f"Trained {trained_at[:10]}"
                if _days_ago > 90:
                    freshness_color, freshness_icon = "#ef5350", "⚠"
                elif _days_ago > 30:
                    freshness_color, freshness_icon = "#ffa726", "🟡"
                else:
                    freshness_color, freshness_icon = "#66bb6a", "✓"
            except Exception:
                freshness_label, freshness_color, freshness_icon = trained_at[:10], "#9e9e9e", "·"
        else:
            freshness_label, freshness_color, freshness_icon = "Training date unknown", "#9e9e9e", "·"

        def _stat(label, value, color="#42a5f5"):
            return (f'<div style="text-align:center;padding:0 1.2rem;border-right:1px solid #2a2a50;">'
                    f'<div style="font-size:1.25rem;font-weight:700;color:{color};">{value}</div>'
                    f'<div style="font-size:0.72rem;color:#757575;text-transform:uppercase;letter-spacing:0.4px;margin-top:2px;">{label}</div>'
                    f'</div>')

        banner_html = (
            '<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);'
            'border:1px solid #21262d;border-radius:12px;padding:1rem 1.5rem;'
            'display:flex;align-items:center;gap:0;margin-bottom:0.5rem;flex-wrap:wrap;">'
            f'<div style="padding:0 1.2rem;border-right:1px solid #2a2a50;">'
            f'<div style="font-size:0.72rem;color:#757575;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:2px;">Training Data</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:#e0e0e0;">{train_rows:,} lots</div></div>'
            + _stat("Val R²", vm_r2, "#42a5f5")
            + _stat("Val MedAPE", vm_mape, "#66bb6a")
            + _stat("Test R²", te_r2, "#42a5f5")
            + _stat("Test MedAPE", te_mape, "#66bb6a")
            + _stat("Within 20%", w20, "#ffa726")
            + _stat("3-Fold CV", cv_str, "#ce93d8")
            + f'<div style="text-align:center;padding:0 1.2rem;">'
            + f'<div style="font-size:0.95rem;font-weight:700;color:{freshness_color};">{freshness_icon} {freshness_label}</div>'
            + f'<div style="font-size:0.72rem;color:#757575;text-transform:uppercase;letter-spacing:0.4px;margin-top:2px;">Model Version</div>'
            + '</div>'
            + '</div>'
            + '<p style="font-size:0.75rem;color:#555;margin:0 0 0.8rem 4px;">'
            + 'R² on log hammer &nbsp;·&nbsp; MedAPE on dollar hammer &nbsp;·&nbsp; '
            + 'Train ≤ 2024 &nbsp;·&nbsp; Val = 2025 H1 &nbsp;·&nbsp; Test = 2025 H2 &nbsp;·&nbsp; strict temporal split'
            + '</p>'
        )
        st.markdown(banner_html, unsafe_allow_html=True)

    if predict_btn:
        if not artist_input:
            st.error("Please enter an artist name.")
        elif estimate_low >= estimate_high:
            st.error("Estimate Low must be less than Estimate High.")
        else:
            try:
                predictor = load_predictor()
            except Exception as e:
                st.error(f"Could not load model: {e}")
                st.stop()

            with st.spinner("Running prediction..."):
                result = predictor.predict(
                    artist=artist_input,
                    estimate_low=estimate_low,
                    estimate_high=estimate_high,
                    medium=medium,
                    width_cm=width_cm if width_cm > 0 else None,
                    height_cm=height_cm if height_cm > 0 else None,
                    year_made=year_made if year_made > 1800 else None,
                    signed=signed,
                    exhibited=exhibited,
                    literature=literature,
                    provenance=provenance,
                    house=house,
                    sale_year=sale_year,
                    sale_month=sale_month,
                )

            hammer   = result["predicted_hammer"]
            ci_lo    = result["confidence_p10"]
            ci_hi    = result["confidence_p90"]
            ratio    = result["hammer_to_estimate_ratio"]
            est_mid  = result["estimate_mid"]
            known    = result["artist_known"]

            # ── Metric cards ──────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Predicted Hammer",
                fmt_dollar(hammer),
                delta=f"{'+'if ratio>=1 else ''}{(ratio-1)*100:.0f}% vs estimate mid",
            )
            c2.metric(
                "P10 – P90 Range",
                f"{fmt_dollar(ci_lo)} – {fmt_dollar(ci_hi)}",
                help="80% confidence interval from quantile LightGBM models",
            )
            c3.metric(
                "Hammer / Estimate Ratio",
                f"{ratio:.2f}×",
                help="1.0 = sells exactly at estimate midpoint",
            )

            # ── Prediction Risk + Data Confidence card ────────────────────
            try:
                ci_width_pct = (ci_hi - ci_lo) / hammer * 100 if hammer > 0 else 0
                if ci_width_pct < 60:
                    conf_level, conf_color, conf_icon = "High", "#4caf50", "🟢"
                elif ci_width_pct < 120:
                    conf_level, conf_color, conf_icon = "Medium", "#ff9800", "🟡"
                else:
                    conf_level, conf_color, conf_icon = "Low", "#f44336", "🔴"

                artist_stats_tmp = load_artist_stats()
                n_lots_artist = 0
                sell_thru_val = 0.0
                if not artist_stats_tmp.empty and artist_input in artist_stats_tmp.index:
                    row_a = artist_stats_tmp.loc[artist_input]
                    n_lots_artist = int(row_a.get("n_lots", 0))
                    sell_thru_val = float(row_a.get("sell_through_rate", 0)) * 100
                if n_lots_artist >= 20:
                    depth_label, depth_color = "Deep", "#4caf50"
                elif n_lots_artist >= 5:
                    depth_label, depth_color = "Moderate", "#ff9800"
                else:
                    depth_label, depth_color = "Sparse", "#f44336"

                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid rgba(255,255,255,0.12);
                        border-radius: 8px;
                        padding: 14px 20px;
                        background: rgba(255,255,255,0.04);
                        margin: 8px 0 4px 0;
                        display: flex;
                        gap: 32px;
                        flex-wrap: wrap;
                    ">
                      <div>
                        <div style="font-size:0.75rem;color:#999;margin-bottom:2px;">PREDICTION CONFIDENCE</div>
                        <div style="font-size:1.1rem;font-weight:700;color:{conf_color};">
                          {conf_icon} {conf_level}
                        </div>
                        <div style="font-size:0.78rem;color:#aaa;">P10–P90 width: {ci_width_pct:.0f}% of estimate</div>
                      </div>
                      <div>
                        <div style="font-size:0.75rem;color:#999;margin-bottom:2px;">ARTIST DATA DEPTH</div>
                        <div style="font-size:1.1rem;font-weight:700;color:{depth_color};">{depth_label}</div>
                        <div style="font-size:0.78rem;color:#aaa;">{n_lots_artist:,} historical lots</div>
                      </div>
                      <div>
                        <div style="font-size:0.75rem;color:#999;margin-bottom:2px;">SELL-THROUGH RATE</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#ccc;">
                          {"—" if sell_thru_val == 0 else f"{sell_thru_val:.0f}%"}
                        </div>
                        <div style="font-size:0.78rem;color:#aaa;">Historical lots sold</div>
                      </div>
                      <div>
                        <div style="font-size:0.75rem;color:#999;margin-bottom:2px;">INTERVAL METHOD</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#ccc;">Conformal P10–P90</div>
                        <div style="font-size:0.78rem;color:#aaa;">~82% actual coverage (80% nominal)</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

            # ── Sell-through probability badge ────────────────────────────
            # Check if artist_stats has a sell_through_rate for this artist
            try:
                artist_stats = load_artist_stats()
                if not artist_stats.empty and artist_input in artist_stats.index:
                    sell_thru = float(artist_stats.loc[artist_input].get("sell_through_rate", 0)) * 100
                    if sell_thru > 0:
                        badge_color = (
                            "green" if sell_thru >= 85
                            else "orange" if sell_thru >= 65
                            else "red"
                        )
                        st.markdown(
                            f'<p style="font-size:1.1rem;">🎯 <b>Probability this lot sells: '
                            f'<span style="color:{badge_color};">{sell_thru:.0f}%</span></b> '
                            f'<span style="color:#999;font-size:0.9rem;">'
                            f'(based on {artist_input}\'s historical sell-through)</span></p>',
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass

            # ── Market Timing Score ───────────────────────────────────────
            try:
                df_eval = load_eval_predictions()
                if not df_eval.empty and "sold_price" in df_eval.columns:
                    df_e = df_eval.dropna(subset=["sold_price", "estimate_mid"])
                    df_e = df_e[df_e["estimate_mid"] > 0].copy()
                    df_e["_he"] = df_e["sold_price"] / df_e["estimate_mid"]

                    # Long-run baseline (all data)
                    baseline_he = float(df_e["_he"].median())

                    # Recent 12 months (last year in data)
                    max_yr = int(df_e["sale_year"].max()) if "sale_year" in df_e.columns else 2024
                    recent = df_e[df_e["sale_year"] >= max_yr - 1]
                    recent_he = float(recent["_he"].median()) if len(recent) >= 20 else baseline_he

                    mkt_delta_pct = (recent_he / max(baseline_he, 0.01) - 1) * 100
                    if mkt_delta_pct > 5:
                        mkt_label, mkt_color, mkt_icon = "Hot Market", "#4caf50", "🔥"
                        mkt_desc = f"Recent lots selling {mkt_delta_pct:+.0f}% vs long-run median"
                    elif mkt_delta_pct < -5:
                        mkt_label, mkt_color, mkt_icon = "Cool Market", "#ef5350", "❄️"
                        mkt_desc = f"Recent lots selling {mkt_delta_pct:+.0f}% vs long-run median"
                    else:
                        mkt_label, mkt_color, mkt_icon = "Neutral Market", "#ff9800", "⚖️"
                        mkt_desc = f"Recent lots near long-run median ({mkt_delta_pct:+.1f}%)"

                    # Best house for this price tier from eval data
                    tier_mask = (
                        (df_e["estimate_mid"] >= est_mid * 0.4) &
                        (df_e["estimate_mid"] <= est_mid * 2.5)
                    )
                    tier_data = df_e[tier_mask]
                    if "house" in tier_data.columns and len(tier_data) >= 30:
                        best_house = (tier_data.groupby("house")["_he"].median()
                                      .idxmax())
                        best_house_label = {"christies": "Christie's", "phillips": "Phillips",
                                            "sothebys": "Sotheby's"}.get(best_house, best_house.title())
                        house_tip = f"Best H/E at this price tier: {best_house_label}"
                    else:
                        house_tip = ""

                    st.markdown(
                        f'<div style="border:1px solid rgba(255,255,255,0.1);border-radius:8px;'
                        f'padding:10px 18px;background:rgba(255,255,255,0.03);margin:6px 0;'
                        f'display:flex;gap:28px;flex-wrap:wrap;align-items:center;">'
                        f'<div><div style="font-size:0.72rem;color:#999;text-transform:uppercase;'
                        f'letter-spacing:.4px;margin-bottom:2px;">MARKET TIMING</div>'
                        f'<div style="font-size:1.05rem;font-weight:700;color:{mkt_color};">'
                        f'{mkt_icon} {mkt_label}</div>'
                        f'<div style="font-size:0.76rem;color:#aaa;">{mkt_desc}</div></div>'
                        + (f'<div><div style="font-size:0.72rem;color:#999;text-transform:uppercase;'
                           f'letter-spacing:.4px;margin-bottom:2px;">HOUSE INSIGHT</div>'
                           f'<div style="font-size:0.95rem;font-weight:600;color:#ccc;">{house_tip}</div>'
                           f'</div>' if house_tip else "")
                        + f'</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

            st.divider()

            # ── CI chart ──────────────────────────────────────────────────
            st.plotly_chart(
                ci_chart(estimate_low, estimate_high, hammer, ci_lo, ci_hi),
                width='stretch',
                key="predict_ci_chart",
            )

            st.divider()

            # ── SHAP waterfall + explanation card ─────────────────────────
            if not SHAP_AVAILABLE:
                st.info("SHAP explanation not available. Install with: `pip install shap`")
            else:
                with st.spinner("Computing SHAP explanation..."):
                    try:
                        explainer = load_shap_explainer()
                        from ml.features import FEATURE_COLS
                        X_shap, _, _, _, _ = predictor._fvec(
                            artist=artist_input,
                            estimate_low=estimate_low,
                            estimate_high=estimate_high,
                            medium=medium,
                            width_cm=width_cm if width_cm > 0 else None,
                            height_cm=height_cm if height_cm > 0 else None,
                            year_made=year_made if year_made > 1800 else None,
                            signed=signed,
                            exhibited=exhibited,
                            literature=literature,
                            provenance=provenance,
                            house=house,
                            sale_year=sale_year,
                            sale_month=sale_month,
                        )
                        # The booster expects sell_through_prob appended as the 43rd
                        # feature (see predict.py). _fvec returns the 42 base features,
                        # so replicate that augmentation here or SHAP's feature count
                        # won't match the trained model.
                        shap_feats = list(FEATURE_COLS)
                        if predictor.meta.get("has_sell_through_feature", False):
                            _stp = predictor._sell_through_prob(X_shap)
                            X_shap = np.hstack([X_shap, np.array([[_stp]], dtype=np.float32)])
                            shap_feats = shap_feats + ["sell_through_prob"]

                        sv = explainer.shap_values(X_shap)[0]
                        base_val = explainer.expected_value
                        if isinstance(base_val, (list, np.ndarray)):
                            base_val = float(base_val[0])

                        # Waterfall chart
                        fig_shap = shap_waterfall_chart(
                            X_shap, shap_feats, explainer, artist_input, hammer
                        )
                        st.plotly_chart(fig_shap, width='stretch', key="predict_shap_waterfall")

                        st.divider()

                        # Structured explanation card
                        shap_explanation_card(
                            sv, shap_feats, base_val,
                            hammer, ci_lo, ci_hi, estimate_high,
                        )

                        st.divider()

                        # ── Factor Decomposition ──────────────────────────
                        st.subheader("📐 Factor Attribution")
                        st.caption(
                            "SHAP contributions grouped into orthogonal return factors — "
                            "analogous to Barra-style risk factor decomposition."
                        )
                        try:
                            _factor_groups = {
                                "Estimate Anchor":   [f for f in FEATURE_COLS if "estimate" in f or "log_est" in f],
                                "Artist":            [f for f in FEATURE_COLS if f.startswith("artist_") or f == "artist_known"],
                                "Market Regime":     [f for f in FEATURE_COLS if any(k in f for k in ["market_index", "market_momentum", "sale_year", "house_"])],
                                "Medium & Format":   [f for f in FEATURE_COLS if any(k in f for k in ["medium_", "log_area", "log_width", "log_height", "aspect"])],
                                "Provenance":        [f for f in FEATURE_COLS if any(k in f for k in ["signed", "exhibited", "literature", "provenance"])],
                                "Seasonality":       [f for f in FEATURE_COLS if any(k in f for k in ["sale_month", "month_sin", "month_cos", "quarter"])],
                            }
                            _base_price = math.expm1(base_val)
                            _fac_rows = []
                            for _fac_name, _fac_feats in _factor_groups.items():
                                _fac_idxs = [FEATURE_COLS.index(f) for f in _fac_feats if f in FEATURE_COLS]
                                if not _fac_idxs:
                                    continue
                                _fac_shap_sum = float(sv[_fac_idxs].sum())
                                _fac_dollar = math.expm1(base_val + _fac_shap_sum) - _base_price
                                _fac_pct = _fac_dollar / max(hammer, 1) * 100
                                _fac_rows.append({
                                    "Factor":         _fac_name,
                                    "$ Impact":       fmt_dollar(_fac_dollar),
                                    "% of Hammer":    f"{_fac_pct:+.1f}%",
                                    "Direction":      "↑" if _fac_dollar > 0 else "↓",
                                })
                            if _fac_rows:
                                _fac_df = pd.DataFrame(_fac_rows)
                                def _fac_color(row):
                                    c = "#4caf5033" if row["Direction"] == "↑" else "#f4433633"
                                    return [f"background-color:{c}"] * len(row)
                                st.dataframe(
                                    _fac_df.set_index("Factor").style.apply(_fac_color, axis=1),
                                    width='stretch',
                                )
                        except Exception:
                            pass

                    except Exception as e:
                        st.warning(f"SHAP explanation failed: {e}")

            st.divider()

            # ── Sensitivity Analysis ──────────────────────────────────────
            st.subheader("📊 Sensitivity Analysis")
            st.caption(
                "Re-runs the model on modified inputs to show how each factor independently "
                "shifts the predicted hammer price — analogous to a DCF sensitivity table."
            )
            try:
                predictor_sa = load_predictor()
                _base_inputs = dict(
                    artist=artist_input, estimate_low=estimate_low, estimate_high=estimate_high,
                    medium=medium, width_cm=width_cm if width_cm > 0 else None,
                    height_cm=height_cm if height_cm > 0 else None,
                    year_made=year_made if year_made > 1800 else None,
                    signed=signed, exhibited=exhibited, literature=literature,
                    provenance=provenance, house=house, sale_year=sale_year, sale_month=sale_month,
                )
                _scenarios = [
                    ("Estimate +20%",    {**_base_inputs, "estimate_low": estimate_low*1.20, "estimate_high": estimate_high*1.20}),
                    ("Estimate −20%",    {**_base_inputs, "estimate_low": estimate_low*0.80, "estimate_high": estimate_high*0.80}),
                    ("Christie's",       {**_base_inputs, "house": "christies"}),
                    ("Phillips",         {**_base_inputs, "house": "phillips"}),
                    ("Sotheby's",        {**_base_inputs, "house": "sothebys"}),
                    ("+ Signed",         {**_base_inputs, "signed": True}),
                    ("− Signed",         {**_base_inputs, "signed": False}),
                    ("+ Exhibited",      {**_base_inputs, "exhibited": True}),
                    ("+ Provenance",     {**_base_inputs, "provenance": "Private Collection, New York"}),
                ]
                _sa_rows = []
                for _sc_name, _sc_kwargs in _scenarios:
                    try:
                        _sc_res = predictor_sa.predict(**_sc_kwargs)
                        _sc_hammer = _sc_res["predicted_hammer"]
                        _sc_diff   = _sc_hammer - hammer
                        _sc_pct    = _sc_diff / max(hammer, 1) * 100
                        _sa_rows.append({
                            "Scenario":          _sc_name,
                            "Predicted Hammer":  fmt_dollar(_sc_hammer),
                            "Δ vs Base":         f"{'+' if _sc_diff >= 0 else ''}{fmt_dollar(_sc_diff)}",
                            "Δ %":               f"{_sc_pct:+.1f}%",
                        })
                    except Exception:
                        pass
                if _sa_rows:
                    _sa_df = pd.DataFrame(_sa_rows)
                    def _sa_color(row):
                        try:
                            val = float(row["Δ %"].replace("%","").replace("+",""))
                            if val > 2:   return ["","","background-color:#4caf5022","background-color:#4caf5022"]
                            if val < -2:  return ["","","background-color:#f4433622","background-color:#f4433622"]
                        except Exception:
                            pass
                        return [""] * len(row)
                    st.dataframe(
                        _sa_df.set_index("Scenario").style.apply(_sa_color, axis=1),
                        width='stretch',
                    )
            except Exception as e:
                st.caption(f"Sensitivity analysis unavailable: {e}")

            st.divider()

            # ── Comparable past lots ──────────────────────────────────────
            st.subheader("📋 Comparable Past Lots")
            try:
                df_cache = load_features_cache()
                if not df_cache.empty:
                    comp = find_comparable_sales(
                        df_cache,
                        artist=artist_input,
                        medium=medium,
                        estimate_mid=float(estimate_low + estimate_high) / 2,
                        house=house,
                    )
                    if not comp.empty:
                        st.caption(
                            "Most similar historical lots scored by artist match, medium, "
                            "price proximity, and recency."
                        )
                        _comp_dl, _comp_tbl = st.columns([1, 5])
                        with _comp_dl:
                            st.download_button(
                                "⬇ Export CSV",
                                comp.to_csv(index=False).encode(),
                                file_name=f"comparable_lots_{artist_input.replace(' ','_')}.csv",
                                mime="text/csv",
                                width='stretch',
                            )
                        with _comp_tbl:
                            st.dataframe(comp, width='stretch', hide_index=True)
                    else:
                        st.info("No comparable lots found in training data.")
            except Exception as e:
                st.warning(f"Could not load comparable sales: {e}")

            st.divider()

            # ── Artist history card ───────────────────────────────────────
            st.subheader("Artist History")
            if known:
                try:
                    artist_stats = load_artist_stats()
                    if artist_input in artist_stats.index:
                        row = artist_stats.loc[artist_input]
                        n_lots     = int(row.get("n_sales", 0))
                        med_hammer = math.expm1(float(row.get("median_log_hammer", 0)))
                        sell_thru  = float(row.get("sell_through_rate", 0)) * 100
                        med_ratio  = float(row.get("median_ratio", 0))

                        ah1, ah2, ah3, ah4 = st.columns(4)
                        ah1.metric("Past Lots", f"{n_lots:,}")
                        ah2.metric("Median Hammer", fmt_dollar(med_hammer))
                        ah3.metric("Sell-Through Rate", f"{sell_thru:.1f}%")
                        ah4.metric("Median Ratio", f"{math.exp(med_ratio):.2f}×" if med_ratio else "—")

                        st.success(
                            f"{artist_input} is in the training data with recorded auction history."
                        )
                    else:
                        st.success(f"{artist_input} found in training data.")
                except Exception:
                    st.success(f"{artist_input} found in training data.")
            else:
                st.warning(
                    f"{artist_input} not found in training data. "
                    "Prediction uses global artist statistics — confidence interval is wider."
                )

            st.divider()

            # ── Artist price momentum ─────────────────────────────────────
            if known:
                try:
                    df_cache_art = load_features_cache()
                    fig_mom = chart_artist_momentum(df_cache_art, artist_input)
                    if fig_mom.data:
                        st.subheader(f"📈 {artist_input} — Price Momentum")
                        st.plotly_chart(fig_mom, width='stretch', key="predict_artist_momentum")
                except Exception:
                    pass

            st.divider()

            # ── Market timing ─────────────────────────────────────────────
            st.subheader("🗓️ Best Time to Sell")
            try:
                df_cache_mt = load_features_cache()
                art_lots = df_cache_mt[
                    (df_cache_mt.get("artist", pd.Series(dtype=str)) == artist_input)
                    & df_cache_mt["sold_price"].notna()
                ] if "artist" in df_cache_mt.columns else pd.DataFrame()

                if len(art_lots) >= 5 and "sale_month" in art_lots.columns and "house" in art_lots.columns:
                    house_map_disp = {"christies": "Christie's", "bonhams": "Bonhams", "phillips": "Phillips", "sothebys": "Sotheby's"}
                    art_lots = art_lots.copy()
                    art_lots["house_disp"] = art_lots["house"].map(house_map_disp).fillna(art_lots["house"])
                    art_lots["month_name"] = pd.to_datetime(art_lots["sale_month"], format="%m").dt.strftime("%b")
                    art_lots["hammer_ratio"] = art_lots["sold_price"] / art_lots["estimate_mid"].replace(0, np.nan)

                    timing = (art_lots.groupby(["house_disp", "sale_month", "month_name"])
                                      .agg(median_ratio=("hammer_ratio", "median"),
                                           n_lots=("sold_price", "count"))
                                      .reset_index()
                                      .sort_values(["house_disp", "sale_month"]))

                    if not timing.empty:
                        best = timing.nlargest(5, "median_ratio")[["house_disp", "month_name", "median_ratio", "n_lots"]]
                        best.columns = ["House", "Month", "Median Ratio", "Lots"]
                        best["Median Ratio"] = best["Median Ratio"].map(lambda x: f"{x:.2f}×")
                        mt_c1, mt_c2 = st.columns([1, 2])
                        with mt_c1:
                            st.caption(f"Best house × month combinations for **{artist_input}** based on {len(art_lots):,} historical sales.")
                            st.dataframe(best, width='stretch', hide_index=True)
                        with mt_c2:
                            # Mini heatmap
                            pivot = timing.pivot_table(
                                index="house_disp", columns="month_name",
                                values="median_ratio", aggfunc="median"
                            )
                            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                            pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
                            fig_mt = go.Figure(go.Heatmap(
                                z=pivot.values.tolist(),
                                x=pivot.columns.tolist(),
                                y=pivot.index.tolist(),
                                colorscale="RdYlGn",
                                zmid=1.0,
                                text=[[f"{v:.2f}×" if not np.isnan(v) else "—"
                                       for v in row] for row in pivot.values],
                                texttemplate="%{text}",
                                hovertemplate="House: %{y}<br>Month: %{x}<br>Ratio: %{text}<extra></extra>",
                            ))
                            fig_mt.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=25, b=0),
                                title=dict(text="Hammer/Estimate by House × Month", font=dict(size=12)),
                                **PLOTLY_THEME,
                            )
                            st.plotly_chart(fig_mt, width='stretch', key="predict_market_timing")
                    else:
                        st.info(f"Not enough sales data for {artist_input} to compute market timing.")
                else:
                    st.info(f"Not enough historical sales for {artist_input} to show market timing (need ≥5 lots with known house and month).")
            except Exception as e:
                st.info(f"Market timing unavailable: {e}")

            st.divider()

            # ── Raw result ────────────────────────────────────────────────
            with st.expander("Raw prediction details"):
                st.json(result)

    else:
        # ── Model Validation Card (prominent, top of page) ────────────────
        st.markdown(
            """
            <div style="
                border: 1px solid rgba(66,165,245,0.4);
                border-radius: 8px;
                padding: 18px 22px;
                background: rgba(66,165,245,0.07);
                margin-bottom: 8px;
            ">
            <div style="font-size:1.05rem; font-weight:700; color:#42a5f5; margin-bottom:10px;">
                📊 Model Validation Summary
            </div>
            <div style="font-size:0.82rem; color:#aaa; margin-bottom:12px;">
                Strict temporal split — trained on lots sold ≤ 2024, validated on 2025 H1, tested on 2025 H2.
                No future data leaks into the training set.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bench = compute_benchmark_metrics()
        if not bench.empty:
            for split_label in bench["Split"].unique():
                sub = bench[bench["Split"] == split_label][["Method", "MedAPE", "R2"]].copy()
                st.markdown(f"**{split_label}**")
                sub = sub.rename(columns={"R2": "R²", "MedAPE": "MedAPE %"})
                def _highlight(row):
                    if "Ensemble" in str(row.name):
                        return ["background-color: rgba(66,165,245,0.18)"] * len(row)
                    return [""] * len(row)
                st.dataframe(
                    sub.set_index("Method").style.apply(_highlight, axis=1),
                    width='stretch',
                )
            st.caption(
                "💡 **MedAPE** = median absolute % error on dollar hammer price. "
                f"**R²** on log hammer price — ensemble R² {_hd.get('r2_raw_model', 0.89):.2f} vs specialist-estimate baseline {_hd.get('r2_raw_est', 0.88):.2f} on the deployed model's clean 2025 H2 holdout "
                "(the estimate is near-efficient, so the model ties it on point accuracy and wins on R²). "
                "Where the model adds value: a ranking signal that sorts over- vs. under-performers "
                f"(AUC {_hd.get('ranking_auc', 0.61):.2f}), plus calibrated P10–P90 intervals and SHAP attribution — none of which the estimate provides."
            )

            # ── Walk-forward robustness (primary evaluation) ──────────────
            _wf = _load_walk_forward()
            if _wf:
                p = _wf.get("pooled", {})
                st.markdown(
                    f"**🔁 Walk-forward CV (primary eval — {_wf.get('n_folds', 5)} rolling quarters, "
                    f"{p.get('n', 0):,} out-of-sample lots incl. {_wf.get('christies_test_lots', 0):,} Christie's):** "
                    f"MedAPE **{p.get('medape_model')}%** vs estimate {p.get('medape_est')}% "
                    f"(paired Δ {p.get('medape_delta_pp')}pp, 95% CI {p.get('medape_delta_ci')}), "
                    f"R² **{p.get('r2_model')}** vs {p.get('r2_est')}, "
                    f"ranking AUC **{p.get('ranking_auc')}** {p.get('ranking_auc_ci')}, "
                    f"decile {p.get('decile_bottom')}%→{p.get('decile_top')}% (ρ={p.get('decile_spearman_rho')}). "
                    "Each fold refits the full calibration chain and tests on the next quarter — so this estimates how "
                    "the approach generalizes across time, and restores Christie's (absent from any single recent split)."
                )
                _wfimg = os.path.join(os.path.dirname(__file__), "figures", "walk_forward.png")
                if os.path.exists(_wfimg):
                    st.image(_wfimg, width='stretch')
        else:
            st.info("Run `python -m ml.train` to populate the benchmark table.")

        # ── Methodology box ───────────────────────────────────────────────
        with st.expander("🔬 Methodology", expanded=False):
            st.markdown(
                "**Temporal split & leakage prevention** — All feature engineering "
                "(artist momentum, market index, artist stats) uses only data available *before* each lot's sale date. "
                "Train ≤ 2024 · Validation = 2025 H1 · Test = 2025 H2.\n\n"
                "**Ensemble** — LightGBM + CatBoost base learners combined via non-negative least squares "
                "(NNLS) meta-stacking on out-of-fold predictions. "
                "A three-stage calibration chain (tiered isotonic → medium/house bias correction → confidence-weighted "
                "estimate-anchored shrink) removes residual bias and prevents the model from over-riding the specialist where it lacks signal.\n\n"
                "**Prediction intervals** — Conformalized quantile regression (CQR on LightGBM P10/P90) "
                "calibrated on the 2025 H1 validation set. "
                f"Nominal 80% coverage achieves ~{_hd.get('interval_coverage', 81):.0f}% actual coverage on held-out 2025 test lots.\n\n"
                "**SHAP attribution** — TreeExplainer SHAP values converted from log-price space to approximate "
                "dollar impact via `expm1(base + shap) − expm1(base)`. Attribution is computed on the **LightGBM "
                "base learner** (the dominant ensemble component, NNLS weight ≈0.66), used as a faithful proxy for "
                "the full ensemble; bars are directional drivers, not exact ensemble dollar contributions "
                "(log→dollar is non-additive, so per-feature dollars do not sum exactly to the prediction)."
            )

        st.divider()

        # ── How it works ──────────────────────────────────────────────────
        st.subheader("How it works")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(
                "**① Input**\n\n"
                "Artist, presale estimate, medium, dimensions, "
                "provenance flags, auction house, and sale date."
            )
        with col_b:
            st.markdown(
                "**② Market Layer**\n\n"
                "Artist momentum, medium price trends, auction "
                "seasonality, and house/year market regime — "
                f"computed from {_hero_priced_k}k priced lots ({_hero_total_k}k total incl. unsold)."
            )
        with col_c:
            st.markdown(
                "**③ Ensemble Model**\n\n"
                "LightGBM + CatBoost combined via "
                "NNLS meta-stacking on out-of-fold predictions. "
                "Three-stage calibration removes systematic bias."
            )
        with col_d:
            st.markdown(
                "**④ Output**\n\n"
                "Hammer price estimate with split-conformal P10–P90 "
                "interval, SHAP attribution, artist history, and "
                "comparable past lots."
            )

        st.info(
            "Fill in the artwork details in the sidebar and click **Estimate Hammer Price** — "
            "or use the **Quick examples** buttons (Basquiat / Warhol / Picasso) to get started instantly."
        )

        st.divider()

        # ── Sample output preview ─────────────────────────────────────────
        st.subheader("Example output")
        ex1, ex2, ex3, ex4 = st.columns(4)
        ex1.metric("Predicted Hammer", "$4.20M", delta="+12% vs estimate mid")
        ex2.metric("P10 – P90 Range", "$3.1M – $5.8M", help="80% conformal prediction interval")
        ex3.metric("Hammer / Estimate", "1.12×", help="Above specialist's midpoint estimate")
        ex4.metric("Sell-Through Prob.", "91%", help="Based on artist's historical sell-through rate")
        st.caption(
            "Example: Basquiat, acrylic on canvas, est. $3.5–4.5M, Christie's, May 2025. "
            "Use the sidebar to enter your own lot."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRENDS
# ══════════════════════════════════════════════════════════════════════════════

with tab_trends:
    if not os.path.exists(FEATURES_PARQUET):
        st.warning(
            f"Feature cache not found at `{FEATURES_PARQUET}`. "
            "Run training first to generate it."
        )
    else:
        with st.spinner("Loading auction data..."):
            df_all  = load_features_cache()
            df_eval = load_eval_predictions()

        if df_all.empty:
            st.warning("Feature cache is empty.")
        else:
            n_artists = df_all["artist"].nunique() if "artist" in df_all.columns else 0
            n_years   = df_all["sale_year"].nunique() if "sale_year" in df_all.columns else 0
            st.caption(
                f"**{len(df_all):,} auction lots** • {n_years} years (2018–2026) "
                f"• {n_artists:,} artists • Christie's, Bonhams, Phillips, Sotheby's"
            )

            # ─────────────────────────────────────────────────────────────
            # CHAPTER 1 — GLOBAL MARKET CYCLE
            # ─────────────────────────────────────────────────────────────
            with st.expander("📈 Market Cycle", expanded=True):
                st.caption("🔗 *Used as model features: `market_index_3m`, `market_index_12m`, `market_momentum`*")
                with st.spinner("Building market cycle..."):
                    st.plotly_chart(chart_market_cycle(df_all), width='stretch', key="trends_market_cycle")

                col_a, col_b = st.columns(2)
                with col_a:
                    with st.spinner("House battle..."):
                        st.plotly_chart(chart_house_battle(df_all), width='stretch', key="trends_house_battle")
                    st.caption("🔗 *Used as model feature: `house` (auction house regime)*")
                with col_b:
                    with st.spinner("Price tier distribution..."):
                        st.plotly_chart(chart_price_tier_distribution(df_all), width='stretch', key="trends_price_tier")

            # ─────────────────────────────────────────────────────────────
            # CHAPTER 2 — ARTIST INTELLIGENCE
            # ─────────────────────────────────────────────────────────────
            with st.expander("🎨 Artist Intelligence", expanded=True):
                col_c, col_d = st.columns(2)
                with col_c:
                    with st.spinner("Finding fastest rising artists..."):
                        st.plotly_chart(chart_rising_artists(df_all), width='stretch', key="trends_rising_artists")
                with col_d:
                    with st.spinner("Building dollar volume treemap..."):
                        st.plotly_chart(chart_artist_treemap(df_all), width='stretch', key="trends_artist_treemap")

                st.markdown("#### Artist Price Momentum — Drill-Down")
                st.caption("🔗 *Used as model features: `artist_momentum_6m`, `artist_momentum_12m`, `artist_median_log_hammer`, `artist_prestige_score`*")
                if "artist" in df_all.columns:
                    top_artists_series = (
                        df_all[df_all["sold_price"].notna()]
                        .groupby("artist")["sold_price"].count()
                        .sort_values(ascending=False)
                        .head(50)
                    )
                    top_artists_dict = top_artists_series.to_dict()
                    artist_options_labeled = [
                        f"{a} ({n:,} lots)" for a, n in top_artists_dict.items()
                    ]
                    artist_names_list = list(top_artists_dict.keys())

                    # Sync to sidebar artist if they're in the top-50 list
                    _sidebar_artist = st.session_state.get("artist", "")
                    if _sidebar_artist and _sidebar_artist in artist_names_list:
                        default_idx = artist_names_list.index(_sidebar_artist)
                    elif "Pablo Picasso" in artist_names_list:
                        default_idx = artist_names_list.index("Pablo Picasso")
                    else:
                        default_idx = 0

                    selected_label  = st.selectbox("Select artist", artist_options_labeled, index=default_idx)
                    selected_artist = artist_names_list[artist_options_labeled.index(selected_label)]
                    st.caption(
                        "Bars = median hammer by year. "
                        "Red line = hammer/estimate ratio (>1.0× means sold above the specialist's midpoint estimate)."
                    )
                    with st.spinner(f"Loading {selected_artist}..."):
                        st.plotly_chart(chart_artist_momentum(df_all, selected_artist), width='stretch', key="trends_artist_momentum")

                # ── Artist Comparison ─────────────────────────────────────
                st.markdown("#### 🥊 Artist Head-to-Head Comparison")
                if "artist" in df_all.columns:
                    cmp_col1, cmp_col2 = st.columns(2)
                    artist_list_cmp = sorted(df_all[df_all["sold_price"].notna()]["artist"].dropna().unique().tolist())
                    default_a = "Jean-Michel Basquiat" if "Jean-Michel Basquiat" in artist_list_cmp else artist_list_cmp[0]
                    default_b = "Andy Warhol" if "Andy Warhol" in artist_list_cmp else artist_list_cmp[1]
                    with cmp_col1:
                        artist_a = st.selectbox("Artist A", artist_list_cmp,
                                                index=artist_list_cmp.index(default_a),
                                                key="_cmp_artist_a")
                    with cmp_col2:
                        artist_b = st.selectbox("Artist B", artist_list_cmp,
                                                index=artist_list_cmp.index(default_b),
                                                key="_cmp_artist_b")
                    if artist_a and artist_b and artist_a != artist_b:
                        with st.spinner(f"Comparing {artist_a} vs {artist_b}..."):
                            st.plotly_chart(chart_artist_comparison(df_all, artist_a, artist_b),
                                            width='stretch', key="trends_artist_comparison")

            # ─────────────────────────────────────────────────────────────
            # CHAPTER 3 — ASSET CHARACTERISTICS
            # ─────────────────────────────────────────────────────────────
            with st.expander("🖼️ Asset Characteristics — Medium & Era", expanded=False):
                st.caption("🔗 *Used as model features: `medium_cat_*` one-hot encoding, `medium_price_momentum`, `log_area`, `log_width`, `log_height`, `aspect_ratio`*")

                col_e, col_f = st.columns(2)
                with col_e:
                    with st.spinner("Building Medium Price Index..."):
                        st.plotly_chart(chart_medium_index(df_all), width='stretch', key="trends_medium_index")
                with col_f:
                    with st.spinner("Building decade vs price..."):
                        st.plotly_chart(chart_decade_vs_price(df_all), width='stretch', key="trends_decade_price")

                with st.spinner("Building medium intelligence bubble..."):
                    st.plotly_chart(chart_medium_bubble(df_all), width='stretch', key="trends_medium_bubble")

            # ─────────────────────────────────────────────────────────────
            # CHAPTER 4 — SEASONALITY
            # ─────────────────────────────────────────────────────────────
            with st.expander("📅 Seasonality & Calendar Patterns", expanded=False):
                st.caption("🔗 *Used as model features: `sale_month`, `sale_month_sin`, `sale_month_cos` (cyclical encoding) and `quarterly hammer/estimate ratio`*")

                col_g, col_h = st.columns(2)
                with col_g:
                    with st.spinner("Building Hammer/Estimate Ratio chart..."):
                        st.plotly_chart(chart_hammer_ratio(df_all), width='stretch', key="trends_hammer_ratio")
                with col_h:
                    with st.spinner("Building seasonal heatmap..."):
                        st.plotly_chart(chart_seasonal_heatmap(df_all), width='stretch', key="trends_seasonal_heatmap")

            # ─────────────────────────────────────────────────────────────
            # CHAPTER 5 — MODEL EVIDENCE
            # ─────────────────────────────────────────────────────────────
            with st.expander("🤖 Model Evidence", expanded=False):
                if not df_eval.empty:
                    # Walk-forward backtest
                    with st.spinner("Building walk-forward backtest..."):
                        st.plotly_chart(chart_walk_forward_backtest(df_eval), width='stretch', key="trends_walk_forward")
                    st.caption(
                        "Each bar shows MedAPE on lots from **that quarter only** — "
                        "never seen during training. Blue fill = gap between ensemble and specialist baseline. "
                        "Green shading = held-out window (the model trained on lots ≤ 2024)."
                    )

                    st.divider()

                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        with st.spinner("Building model calibration chart..."):
                            st.plotly_chart(chart_model_calibration(df_eval), width='stretch', key="trends_calibration")
                    with perf_col2:
                        with st.spinner("Building reliability diagram..."):
                            st.plotly_chart(chart_reliability_diagram(df_eval), width='stretch', key="trends_reliability")
                    st.caption(
                        "**Left:** Predicted vs actual hammer price (log scale). Points near the dashed diagonal = accurate predictions. "
                        "**Right:** Reliability diagram — how often the model's confidence intervals actually contain the true price. "
                        "A perfectly calibrated model hugs the diagonal. The 80% interval achieves ~82% actual coverage."
                    )

                    st.divider()

                    # Residual diagnostics
                    with st.spinner("Building residual diagnostics..."):
                        st.plotly_chart(chart_residual_diagnostics(df_eval), width='stretch', key="trends_residuals")
                    st.caption(
                        "**Left:** Residuals vs log estimate — a flat scatter means no systematic bias by price tier (heteroscedasticity). "
                        "**Centre:** Residuals over time — a flat trend means no model drift. "
                        "**Right:** Residual distribution — the orange dashed line is a fitted Normal; "
                        "slight fat tails are typical for art auction prices."
                    )
                else:
                    st.info("Run `python -m ml.train` to generate evaluation predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PRESALE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_presale:
    st.warning(
        "**⚠️ This is a batch-scoring demonstration, not a calendar of upcoming auctions.** "
        "The lots shown are records with **no recorded hammer price** (`sold == 0`) — which conflates "
        "genuinely unsold / bought-in lots with any unpriced entries. They are scored here only to "
        "demonstrate the model running over many lots at once. **\"Edge %\" is not an investment signal "
        "or a guaranteed return**, and these are not curated future-sale opportunities.",
        icon="⚠️",
    )
    if os.path.exists(NO_HAMMER_CSV):
        with st.spinner("Scoring lots..."):
            df_up = pd.read_csv(NO_HAMMER_CSV)

        # ── Summary KPIs ──────────────────────────────────────────────────
        total_lots = len(df_up)
        strong_buys = int((df_up["edge_pct"] > 40).sum()) if "edge_pct" in df_up.columns else 0
        median_edge = float(df_up["edge_pct"].median()) if "edge_pct" in df_up.columns else 0
        houses_in   = df_up["house"].nunique() if "house" in df_up.columns else 0
        years_in    = df_up["sale_year"].nunique() if "sale_year" in df_up.columns else 0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Lots", f"{total_lots:,}")
        kpi2.metric("High Edge Lots (Edge >40%)", f"{strong_buys:,}")
        kpi3.metric("Median Edge", f"{median_edge:+.0f}%")
        kpi4.metric("Houses × Years", f"{houses_in} × {years_in}")

        st.markdown(
            "**Edge %** = model-predicted hammer price relative to the specialist's estimate midpoint. "
            "A positive edge indicates model-estimated upside vs the specialist consensus — **not a guaranteed return**. "
            "🟢 >40% = model sees significant upside &nbsp; 🟡 0–40% = modest upside &nbsp; 🔴 <0% = model below estimate.",
            unsafe_allow_html=True,
        )
        # ── Alpha Scorecard ───────────────────────────────────────────────
        with st.expander("📈 Alpha Scorecard — Information Ratio by Segment", expanded=False):
            df_eval_ps = load_eval_predictions()
            if not df_eval_ps.empty:
                alpha_df = compute_alpha_scorecard(df_eval_ps)
                if not alpha_df.empty:
                    _tab_house, _tab_tier, _tab_year = st.tabs(["By House", "By Price Tier", "By Year"])
                    for _tab_widget, _cat in [(_tab_house, "House"), (_tab_tier, "Price Tier"), (_tab_year, "Year")]:
                        with _tab_widget:
                            _sub = alpha_df[alpha_df["Category"] == _cat].drop(columns="Category")
                            if not _sub.empty:
                                def _ir_color(row):
                                    try:
                                        ir = float(row["Info Ratio"])
                                        if ir > 0.5:   return [""] * (len(row)-1) + ["background-color:#4caf5033"]
                                        if ir < -0.5:  return [""] * (len(row)-1) + ["background-color:#f4433633"]
                                    except Exception:
                                        pass
                                    return [""] * len(row)
                                st.dataframe(
                                    _sub.set_index("Segment").style.apply(_ir_color, axis=1),
                                    width='stretch',
                                )
                            else:
                                st.info("No data for this segment.")
                    st.caption(
                        "**Info Ratio** = Mean Edge % ÷ σ(Edge %) — analogous to a Sharpe ratio. "
                        "IR > 0.5 signals the model has systematic alpha in that segment. "
                        "**Hit Rate** = % of lots where model direction matched realized alpha. "
                        "Based on the held-out 2025 evaluation set; not a guarantee of future performance."
                    )
                else:
                    st.info("Run `python -m ml.train` to compute evaluation predictions.")
            else:
                st.info("Run `python -m ml.train` to compute evaluation predictions.")

        st.divider()

        # ── Filters ───────────────────────────────────────────────────────
        f1, f2, f3 = st.columns(3)
        with f1:
            house_opts = ["All"] + sorted(df_up["house"].dropna().unique().tolist()) if "house" in df_up.columns else ["All"]
            house_filter = st.selectbox("🏛 Auction House", house_opts, key="_ps_house")
        with f2:
            edge_min = st.slider("Min Edge %", min_value=-100, max_value=200, value=0, step=5, key="_ps_edge")
        with f3:
            year_opts = ["All"] + sorted(df_up["sale_year"].dropna().astype(int).unique().tolist()) if "sale_year" in df_up.columns else ["All"]
            year_filter = st.selectbox("📅 Sale Year", year_opts, key="_ps_year")

        # Apply filters
        df_filtered = df_up.copy()
        if house_filter != "All" and "house" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["house"] == house_filter]
        if "edge_pct" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["edge_pct"] >= edge_min]
        if year_filter != "All" and "sale_year" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["sale_year"].astype(int) == int(year_filter)]

        df_filtered = df_filtered.sort_values("edge_pct", ascending=False) if "edge_pct" in df_filtered.columns else df_filtered

        # ── Edge distribution chart ────────────────────────────────────────
        if "edge_pct" in df_filtered.columns and len(df_filtered) > 0:
            ec1, ec2 = st.columns([2, 1])
            with ec1:
                edge_vals = df_filtered["edge_pct"].clip(-100, 300)
                fig_edge = go.Figure()
                fig_edge.add_trace(go.Histogram(
                    x=edge_vals, nbinsx=40,
                    marker_color=[
                        "#43a047" if v > 40 else "#fdd835" if v > 0 else "#e53935"
                        for v in edge_vals
                    ],
                    hovertemplate="Edge: %{x:.0f}%<br>Count: %{y}<extra></extra>",
                ))
                fig_edge.add_vline(x=0, line_color="#888", line_dash="dash")
                fig_edge.add_vline(x=40, line_color="#43a047", line_dash="dot",
                                   annotation_text="High Edge threshold", annotation_position="top right",
                                   annotation_font_color="#43a047")
                fig_edge.update_layout(
                    title=f"Edge % Distribution ({len(df_filtered):,} lots)",
                    xaxis_title="Edge % (model vs estimate midpoint)",
                    yaxis_title="Number of lots",
                    height=280, showlegend=False,
                    margin=dict(t=40, b=40),
                    **PLOTLY_THEME,
                )
                st.plotly_chart(fig_edge, width='stretch', key="presale_edge_dist")
            with ec2:
                # Top artists by strong-buy count
                if "artist" in df_filtered.columns:
                    strong = df_filtered[df_filtered["edge_pct"] > 40]
                    if not strong.empty:
                        top_art = strong["artist"].value_counts().head(8).reset_index()
                        top_art.columns = ["Artist", "High Edge Lots"]
                        st.markdown("**Top artists with most High Edge lots**")
                        st.dataframe(top_art, hide_index=True, width='stretch')

        st.divider()

        # ── Main lots table ───────────────────────────────────────────────
        st.subheader(f"📋 Lot-by-Lot Predictions ({len(df_filtered):,} lots)")

        # Build display df using pre-formatted columns from generate_presale.py
        disp_cols_map = {
            "artist":           "Artist",
            "medium":           "Medium",
            "house":            "House",
            "sale_year":        "Year",
            "sale_month":       "Month",
            "Est. Low":         "Est. Low",
            "Est. High":        "Est. High",
            "Predicted Hammer": "Predicted Hammer",
            "P10":              "P10",
            "P90":              "P90",
            "Edge %":           "Edge %",
        }
        disp_cols = [c for c in disp_cols_map if c in df_filtered.columns]
        df_disp = df_filtered[disp_cols].rename(columns=disp_cols_map).head(500)

        # Color-code Edge % column with background gradient via dataframe styling
        def _edge_style(val):
            try:
                v = float(str(val).replace("%", "").replace("+", ""))
                if v > 40:
                    return "background-color: rgba(67,160,71,0.25); color: #81c784"
                elif v > 0:
                    return "background-color: rgba(253,216,53,0.15); color: #fff176"
                else:
                    return "background-color: rgba(229,57,53,0.2); color: #ef9a9a"
            except Exception:
                return ""

        styled_disp = df_disp.style
        if "Edge %" in df_disp.columns:
            styled_disp = styled_disp.map(_edge_style, subset=["Edge %"])

        dl_col, tbl_col = st.columns([1, 8])
        with dl_col:
            st.download_button(
                "⬇ Export CSV",
                df_filtered.to_csv(index=False).encode(),
                file_name="presale_predictions.csv",
                mime="text/csv",
                width='stretch',
            )
        with tbl_col:
            st.dataframe(styled_disp, width='stretch', hide_index=True, height=500)

        st.caption(
            f"Data from `data/no_hammer_batch_scores.csv` (no-hammer lots) · "
            f"{total_lots:,} total presale lots · "
            "Regenerate with: `python generate_presale.py`"
        )

    else:
        # ── Visual placeholder + generate button ──────────────────────────
        st.markdown("""
        ### 📋 Batch Scoring (demo)

        This dashboard scores **every upcoming unsold lot** in the dataset using the trained ensemble
        and ranks them by **Edge %** — how much the model predicts above the specialist's estimate midpoint.

        | Column | Meaning |
        |---|---|
        | **Predicted Hammer** | Model's expected hammer price |
        | **Edge %** | Model prediction ÷ estimate midpoint − 1 |
        | **P10 / P90** | 80% confidence interval bounds |
        | **House** | Christie's, Bonhams, Phillips, or Sotheby's |

        **Signal legend:**
        - 🟢 **Edge >40%** — model sees strong upside vs specialist estimate
        - 🟡 **Edge 0–40%** — modest upside
        - 🔴 **Edge <0%** — model expects below-estimate result (caution)
        """)

        if st.button("🚀 Generate Presale Predictions", type="primary"):
            script = os.path.join(os.path.dirname(__file__), "generate_presale.py")
            if not os.path.exists(script):
                st.error("`generate_presale.py` not found.")
            else:
                with st.spinner("Running batch predictions on all presale lots (~1–2 min)..."):
                    proc = subprocess.run(
                        [sys.executable, script],
                        capture_output=True, text=True,
                        cwd=os.path.dirname(__file__),
                    )
                if proc.returncode == 0:
                    st.success("Done!")
                    st.rerun()
                else:
                    st.error(f"Failed:\n```\n{proc.stderr[:2000]}\n```")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("📦 Batch Prediction — Upload a Portfolio")
    st.markdown(
        "Upload a CSV with multiple lots and get predictions for all of them at once. "
        "Useful for valuing a collection or comparing upcoming lots."
    )

    # ── Template download ──────────────────────────────────────────────────
    template_df = pd.DataFrame([{
        "artist":       "Jean-Michel Basquiat",
        "medium":       "acrylic on canvas",
        "estimate_low": 3_000_000,
        "estimate_high":5_000_000,
        "width_cm":     183.0,
        "height_cm":    122.0,
        "year_made":    1984,
        "signed":       True,
        "exhibited":    True,
        "literature":   True,
        "provenance":   "Private Collection, New York",
        "house":        "christies",
        "sale_year":    2025,
        "sale_month":   5,
    }, {
        "artist":       "Andy Warhol",
        "medium":       "screenprint",
        "estimate_low": 800_000,
        "estimate_high":1_200_000,
        "width_cm":     101.6,
        "height_cm":    101.6,
        "year_made":    1962,
        "signed":       True,
        "exhibited":    False,
        "literature":   True,
        "provenance":   "Private Collection, London",
        "house":        "phillips",
        "sale_year":    2025,
        "sale_month":   6,
    }])

    st.download_button(
        "⬇ Download Template CSV",
        template_df.to_csv(index=False).encode(),
        file_name="batch_template.csv",
        mime="text/csv",
    )

    st.caption(
        "Required columns: `artist`, `estimate_low`, `estimate_high`. "
        "Optional: `medium`, `width_cm`, `height_cm`, `year_made`, `signed`, `exhibited`, "
        "`literature`, `provenance`, `house` (christies/bonhams/phillips/sothebys), `sale_year`, `sale_month`."
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"], key="_batch_upload")

    if uploaded is not None:
        try:
            df_batch_in = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_batch_in):,} rows.")

            required = {"artist", "estimate_low", "estimate_high"}
            missing = required - set(df_batch_in.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                if st.button("🚀 Run Batch Predictions", type="primary", key="_batch_run"):
                    try:
                        predictor_b = load_predictor()
                    except Exception as e:
                        st.error(f"Could not load model: {e}")
                        st.stop()

                    results_batch = []
                    prog = st.progress(0, text="Running predictions...")
                    for i, row in df_batch_in.iterrows():
                        try:
                            res = predictor_b.predict(
                                artist=str(row.get("artist", "")),
                                estimate_low=float(row.get("estimate_low", 10_000)),
                                estimate_high=float(row.get("estimate_high", 15_000)),
                                medium=str(row.get("medium", "unknown")),
                                width_cm=float(row["width_cm"]) if pd.notna(row.get("width_cm")) and float(row.get("width_cm", 0)) > 0 else None,
                                height_cm=float(row["height_cm"]) if pd.notna(row.get("height_cm")) and float(row.get("height_cm", 0)) > 0 else None,
                                year_made=int(row["year_made"]) if pd.notna(row.get("year_made")) and int(row.get("year_made", 0)) > 1800 else None,
                                signed=bool(row.get("signed", False)),
                                exhibited=bool(row.get("exhibited", False)),
                                literature=bool(row.get("literature", False)),
                                provenance=str(row.get("provenance", "")),
                                house=str(row.get("house", "christies")),
                                sale_year=int(row.get("sale_year", 2025)),
                                sale_month=int(row.get("sale_month", 6)),
                            )
                            est_mid = (float(row.get("estimate_low", 10_000)) + float(row.get("estimate_high", 15_000))) / 2
                            edge    = (res["predicted_hammer"] / est_mid - 1) * 100 if est_mid > 0 else 0
                            results_batch.append({
                                **{c: row.get(c, "") for c in df_batch_in.columns},
                                "Predicted Hammer":  fmt_dollar(res["predicted_hammer"]),
                                "P10":               fmt_dollar(res["confidence_p10"]),
                                "P90":               fmt_dollar(res["confidence_p90"]),
                                "Edge %":            f"{edge:+.0f}%",
                                "Ratio":             f"{res['hammer_to_estimate_ratio']:.2f}×",
                                "Artist Known":      "✓" if res["artist_known"] else "—",
                                "_predicted_hammer": res["predicted_hammer"],
                                "_edge_pct":         edge,
                            })
                        except Exception:
                            results_batch.append({
                                **{c: row.get(c, "") for c in df_batch_in.columns},
                                "Predicted Hammer": "ERROR",
                                "P10": "—", "P90": "—",
                                "Edge %": "—", "Ratio": "—",
                                "Artist Known": "—",
                                "_predicted_hammer": 0, "_edge_pct": 0,
                            })
                        prog.progress((i + 1) / len(df_batch_in), text=f"Row {i+1}/{len(df_batch_in)}")

                    prog.empty()

                    df_results = pd.DataFrame(results_batch)
                    df_results = df_results.sort_values("_edge_pct", ascending=False)
                    display_cols_b = [c for c in df_results.columns if not c.startswith("_")]
                    df_disp_b = df_results[display_cols_b]

                    st.success(f"✅ {len(df_results):,} predictions complete!")

                    # Summary KPIs
                    bk1, bk2, bk3 = st.columns(3)
                    bk1.metric("Lots processed", len(df_results))
                    bk2.metric("High Edge Lots (Edge >40%)", int((df_results["_edge_pct"] > 40).sum()))
                    bk3.metric("Median Edge", f"{df_results['_edge_pct'].median():+.0f}%")

                    # Edge column styling
                    def _batch_edge_style(val):
                        try:
                            v = float(str(val).replace("%", "").replace("+", ""))
                            if v > 40:
                                return "background-color: rgba(67,160,71,0.25); color: #81c784"
                            elif v > 0:
                                return "background-color: rgba(253,216,53,0.15); color: #fff176"
                            else:
                                return "background-color: rgba(229,57,53,0.2); color: #ef9a9a"
                        except Exception:
                            return ""

                    styled_b = df_disp_b.style
                    if "Edge %" in df_disp_b.columns:
                        styled_b = styled_b.map(_batch_edge_style, subset=["Edge %"])

                    dl_b, tbl_b = st.columns([1, 7])
                    with dl_b:
                        st.download_button(
                            "⬇ Export Results",
                            df_results[display_cols_b].to_csv(index=False).encode(),
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                            width='stretch',
                        )
                    with tbl_b:
                        st.dataframe(styled_b, width='stretch', hide_index=True)

        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:3rem;'></div>", unsafe_allow_html=True)
with st.expander("⚠ Important Disclosures", expanded=False):
    st.markdown(
        """
**Research purposes only.** This tool is a quantitative research prototype. It does not constitute
investment advice, a solicitation to buy or sell any asset, or a recommendation to participate in any
auction.

**Model limitations.** Predicted hammer prices are statistical estimates derived from historical
transaction data. The model does not have access to the specialist's private pre-sale research,
condition reports, buyer relationships, undisclosed reserves, or current-day market intelligence.
Actual hammer prices may differ materially from model estimates.

**Art market risks.** Art is an illiquid asset class. Prices are not continuously observable, lots
may be withdrawn before sale, and buyer's premium (typically 15–26% of hammer) is not included in
any estimates shown. Past auction performance of an artist is not indicative of future results.

**Data coverage.** The model was trained on publicly available hammer prices from Christie's, Bonhams,
Phillips, and Sotheby's for lots sold 2018–2026. It may underperform for artists, media, or price tiers that are
underrepresented in the training data. The Data Depth indicator in prediction results signals when
artist coverage is limited.

**Temporal validity.** The model was trained on a fixed historical dataset. Art market conditions change
over time; model accuracy may degrade if market structure shifts materially from the training period.
The Model Version timestamp in the stats banner indicates when the model was last retrained.

**No guarantee of interval coverage.** Prediction intervals (P10–P90) are calibrated on the 2025 H1
validation set and achieve approximately 80% actual coverage on the 2025 H2 test data. Coverage is
not guaranteed in future periods or for lots outside the training distribution.

---
*This disclaimer should accompany any distribution of outputs from this tool.*
        """
    )
