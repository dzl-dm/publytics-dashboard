"""
DZG Publication Dashboard: interactive analysis of PubMed publication data for all DZGs in the yaml file.

Reads four CSV files (produced by 02_preprocessing.py) and displays five tabs:
Overview, Citations, Journal Metrics, Collaboration, MeSH Terms.
The active DZG is read from dzg_search_terms.yaml (active_dzg key).

Requirements: pip install streamlit plotly pandas matplotlib venn
"""

import locale
import math
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

matplotlib.use("Agg")

# Locale names differ between operating systems, so several spellings are tried in order
for _loc in ("en_US.UTF-8", "en_US", "English_United States.1252", "English"):
    try:
        locale.setlocale(locale.LC_ALL, _loc)
        break
    except locale.Error:
        continue


# Folder paths are resolved relative to this script, so the project runs from any
# location and on any machine without editing hardcoded paths.
# Expected layout:  <project>/code/  contains the scripts and the YAML config
#                   <project>/data/  holds all CSV input and output files
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_ARTICLES            = DATA_DIR / "pubmed_articles_processed.csv"
CSV_AUTHORS             = DATA_DIR / "pubmed_authors_processed.csv"
CSV_MESH                = DATA_DIR / "pubmed_mesh.csv"
CSV_METADATA            = DATA_DIR / "metadata_processing.csv"
CSV_METADATA_EXTRACTION = DATA_DIR / "metadata_extraction.csv"
YAML_PATH               = CODE_DIR / "dzg_search_terms.yaml"
MESH_STOPLIST_PATH      = CODE_DIR / "mesh_stoplist.yaml"

DZG_COLUMNS = ["DZIF", "DZNE", "DZPG", "DZKJ", "DZHK", "DKTK", "DZL", "DZD"]

# Colour maps are resolved at runtime from the YAML config, see resolve_colors()
DZG_COLORS: dict[str, str]  = {}
SITE_COLORS: dict[str, str] = {}

BG      = "#0f1117"
SURFACE = "#1a1d27"
BORDER  = "#2a2d3e"
BLUE    = "#4f8ef7"
ORANGE  = "#f5a623"
PURPLE  = "#9b7cf5"
GREEN   = "#3ecfaa"
MUTED   = "#8b8fa8"
TEXT    = "#e8eaf0"

# Fixed mapping from best to lowest journal quartile. SCImago writes "-" for journals it
# ranks without assigning a quartile, which is relabelled for display only.
QUARTILE_COLORS = {"Q1": GREEN, "Q2": BLUE, "Q3": ORANGE, "Q4": "#e8534f"}
QUARTILE_ORDER  = ["Q1", "Q2", "Q3", "Q4"]
NO_QUARTILE_LABEL = "No quartile"

# Used for categories with no colour configured in the YAML, and for charts whose
# categories are not named entities (for example "1 Site", "2 Sites")
FALLBACK_PALETTE = [BLUE, GREEN, ORANGE, PURPLE, "#e05c8a", "#e8534f", "#8fcc3a", "#f0c040"]


def plot_layout(**kwargs) -> dict:
    """Return a base Plotly layout dict with consistent dark-theme styling; extra kwargs are merged in."""
    base = dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="system-ui, sans-serif", size=12),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        margin=dict(l=48, r=16, t=40, b=48),
        separators=".,",
    )
    base.update(kwargs)
    return base


def format_number(n, decimals: int = 0) -> str:
    """Format a number using the active locale for thousands and decimal separators; returns '–' for None or NaN."""
    if n is None:
        return "–"
    if isinstance(n, str):
        return n
    try:
        if isinstance(n, float) and math.isnan(n):
            return "–"
    except TypeError:
        pass
    return locale.format_string(f"%.{decimals}f", n, grouping=True)


st.set_page_config(
    page_title="DZG Publication Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE}; border-right: 1px solid {BORDER};
    }}
    .metric-card {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
        padding: 18px 20px;
    }}
    .metric-label {{
        color: {MUTED}; font-size: 10px; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 6px;
    }}
    .metric-value {{ color: {TEXT}; font-size: 26px; font-weight: 700; }}
    .metric-sub   {{ color: {MUTED}; font-size: 11px; margin-top: 2px; }}
    .section-label {{
        color: {MUTED}; font-size: 10px; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 4px;
    }}
    .data-note {{ color: {MUTED}; font-size: 10px; font-style: italic; margin-top: 4px; }}
    /* Streamlit's built-in top-right "running man" indicator cycles through a fixed set of
       icons (several are sports emoji) that isn't officially configurable, so it's hidden
       here instead; the existing st.spinner below still gives loading feedback. */
    div[data-testid="stStatusWidget"] {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


def empty_figure(title: str, message: str) -> go.Figure:
    """Return a styled figure carrying only a centred message, used when a chart has no data to show."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(color=MUTED))
    fig.update_layout(**plot_layout(title=title))
    return fig


def kpi(col, label, value, sub=""):
    """Render a styled KPI card with a label, value, and optional subtitle into a Streamlit column."""
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)


def data_note(text: str) -> None:
    """Render a small italic data-source note below a chart."""
    st.markdown(f'<div class="data-note">ℹ️ {text}</div>', unsafe_allow_html=True)


def info_box(title: str, body_html: str) -> None:
    """Render a bordered explanatory box above a section.

    Leading indentation is stripped from every line because Markdown treats four or more
    leading spaces as a code block, which would print the raw HTML instead of rendering it.
    """
    body = " ".join(line.strip() for line in body_html.strip().splitlines())
    html = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;'
        f'padding:18px 22px;margin-bottom:20px;font-size:12px;color:{MUTED}">'
        f'<b style="color:{TEXT}">{title}</b><br><br>{body}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data
def load_active_dzg(path: Path) -> str:
    """Read the active_dzg key from the YAML config; falls back to first DZG in DZG_COLUMNS if missing."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        dzg = raw.get("active_dzg")
        if dzg and dzg in DZG_COLUMNS:
            return dzg
    except Exception:
        pass
    return DZG_COLUMNS[0]


@st.cache_data
def load_color_config(path: Path) -> tuple[dict, dict]:
    """Read the colors block from the YAML config and return (dzg_colors, site_colors_by_dzg)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        colors = raw.get("colors") or {}
        return colors.get("dzg") or {}, colors.get("sites") or {}
    except Exception:
        return {}, {}


def resolve_colors(names: list[str], configured: dict) -> dict[str, str]:
    """Map every name to a colour, preferring the configured value.

    Names without a configured colour receive one from the fallback palette in alphabetical
    order, so an unconfigured name keeps the same colour across every chart in the dashboard.
    """
    resolved = {}
    fallback_index = 0
    for name in sorted(names):
        if configured.get(name):
            resolved[name] = configured[name]
        else:
            resolved[name] = FALLBACK_PALETTE[fallback_index % len(FALLBACK_PALETTE)]
            fallback_index += 1
    return resolved


@st.cache_data
def load_last_update(path: Path) -> str:
    """Read the timestamp of the most recent pipeline run from a metadata CSV.

    Falls back to raw line parsing when the header no longer matches the appended rows,
    which happens after new metadata columns were introduced in the pipeline.
    """
    try:
        meta = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        if not meta.empty and "timestamp" in meta.columns:
            return str(meta.iloc[-1]["timestamp"])
    except Exception:
        pass
    # Fallback: read the last non-empty line and take its first field
    try:
        with open(path, encoding="utf-8-sig") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) > 1:
            return lines[-1].split(";")[0]
    except Exception:
        pass
    return "unknown"


@st.cache_data
def load_mesh_stoplist(path: Path) -> frozenset[str]:
    """Load MeSH stoplist from YAML and return all terms as a frozenset; returns empty set if file is missing."""
    if not path.exists():
        return frozenset()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    terms = set()
    for category_terms in data.values():
        if isinstance(category_terms, list):
            terms.update(category_terms)
    return frozenset(terms)


@st.cache_data
def load_data():
    """Load and type-cast article, author and MeSH CSVs; returns all DataFrames plus last-update timestamps."""
    df_articles = pd.read_csv(CSV_ARTICLES, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    df_authors  = pd.read_csv(CSV_AUTHORS,  sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)

    # pubmed_mesh.csv is optional, it may not exist if the pipeline has not been re-run yet
    try:
        df_mesh = pd.read_csv(CSV_MESH, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    except FileNotFoundError:
        df_mesh = pd.DataFrame(columns=["pmid", "publication_year", "mesh_term"])

    for col in ["n_authors", "n_dzg_authors", "publication_year", "cited_by_count", "rcr", "sjr"]:
        if col in df_articles.columns:
            df_articles[col] = pd.to_numeric(df_articles[col], errors="coerce")

    if "author_position" in df_authors.columns:
        df_authors["author_position"] = pd.to_numeric(df_authors["author_position"], errors="coerce")

    bool_map = {"True": True, "False": False, True: True, False: False}

    for col in ["is_first_author", "is_last_author"]:
        if col in df_authors.columns:
            df_authors[col] = df_authors[col].map(bool_map)

    # DZG affiliation columns and their site sub-columns
    for dzg in DZG_COLUMNS:
        for df in [df_articles, df_authors]:
            for col in df.columns:
                if col == dzg or col.startswith(f"{dzg}_"):
                    df[col] = df[col].map(bool_map)

    if "publication_year" in df_mesh.columns:
        df_mesh["publication_year"] = pd.to_numeric(df_mesh["publication_year"], errors="coerce")

    last_prepared  = load_last_update(CSV_METADATA)
    last_extracted = load_last_update(CSV_METADATA_EXTRACTION)
    return df_articles, df_authors, df_mesh, last_prepared, last_extracted


def h_index(s: pd.Series) -> int:
    """Compute the h-index from a Series of citation counts."""
    counts = sorted(s.dropna().astype(int).tolist(), reverse=True)
    return sum(1 for i, c in enumerate(counts, 1) if c >= i)


def filter_by_dzg(df: pd.DataFrame, dzg: str) -> pd.DataFrame:
    """Return only rows where the given DZG column is True; returns df unchanged if the column is missing."""
    if dzg not in df.columns:
        return df
    return df[df[dzg] == True]


def has_citations(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame contains at least one non-null citation count."""
    return "cited_by_count" in df.columns and df["cited_by_count"].notna().any()


def has_sjr(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame contains at least one non-null SJR value."""
    return "sjr" in df.columns and df["sjr"].notna().any()


def has_rcr(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame contains at least one non-null RCR value."""
    return "rcr" in df.columns and df["rcr"].notna().any()


def year_as_category(df: pd.DataFrame, column: str = "publication_year") -> pd.DataFrame:
    """Cast a numeric year column to string so Plotly renders it as a discrete category axis."""
    # Plotly treats a numeric x-axis as continuous, casting the
    # year to a string keeps it as a clean category axis regardless of selection size
    df = df.copy()
    df[column] = df[column].astype(int).astype(str)
    return df


def leadership_pmids(df_au: pd.DataFrame, dzg: str) -> set:
    """Return the set of PMIDs where at least one DZG-affiliated author is first or last author."""
    if "is_first_author" not in df_au.columns or "is_last_author" not in df_au.columns:
        return set()
    if dzg not in df_au.columns:
        return set()
    mask = (df_au[dzg] == True) & ((df_au["is_first_author"] == True) | (df_au["is_last_author"] == True))
    return set(df_au[mask]["pmid"])


def fig_pubs_per_year(df: pd.DataFrame, df_au: pd.DataFrame, dzg: str) -> go.Figure:
    """Area chart of publications per year with the first or last author publications drawn as a subset."""
    d_pubs = df.groupby("publication_year").size().rename("total")

    lead_pmids = leadership_pmids(df_au, dzg)
    d_lead = (
        df[df["pmid"].isin(lead_pmids)]
        .groupby("publication_year").size()
        .rename("n_lead")
    )
    d = pd.concat([d_pubs, d_lead], axis=1).fillna(0).reset_index()
    d["share"] = (d["n_lead"] / d["total"] * 100).round(1)
    d = year_as_category(d)

    fig = go.Figure()
    # Total publications drawn first so the subset area sits visually inside it
    fig.add_scatter(
        x=d["publication_year"], y=d["total"],
        name="Total Publications",
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.18)",
        line=dict(color=BLUE, width=2),
        marker=dict(color=BLUE, size=6),
        hovertemplate="<b>%{x}</b><br>Total Publications: %{y:,}<extra></extra>",
    )
    fig.add_scatter(
        x=d["publication_year"], y=d["n_lead"],
        name=f"{dzg} First or Last Author",
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(245,166,35,0.30)",
        line=dict(color=ORANGE, width=2),
        marker=dict(color=ORANGE, size=5),
        customdata=d["share"],
        hovertemplate="<b>%{x}</b><br>First or Last Author: %{y:,}"
                      "<br>Share of Total: %{customdata:.1f}%<extra></extra>",
    )
    fig.update_layout(**plot_layout(
        title="Publications per Year",
        yaxis=dict(title="Publications", gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=48, r=16, t=40, b=80),
        hovermode="x unified",
    ))
    fig.update_xaxes(type="category")
    return fig


def fig_pubs_per_site_year(df: pd.DataFrame, site_columns: list[str]) -> go.Figure:
    """Stacked bar chart showing publications per site and year."""
    if not site_columns:
        return None
    rows = []
    for col in site_columns:
        site = col.split("_", 1)[1]
        counts = df[df[col] == True].groupby("publication_year").size()
        for year, n in counts.items():
            rows.append({"publication_year": year, "site": site, "n": n})
    if not rows:
        return empty_figure("Publications per Site and Year", "No site data available")

    d = pd.DataFrame(rows)
    d = year_as_category(d)
    sites = sorted(d["site"].unique())

    fig = go.Figure()
    for site in sites:
        sub = d[d["site"] == site]
        fig.add_bar(
            x=sub["publication_year"], y=sub["n"],
            name=site,
            marker_color=SITE_COLORS.get(site, BLUE),
            marker_line_width=0,
            hovertemplate=f"<b>{site}</b><br>%{{x}}<br>Publications: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(**plot_layout(
        title="Publications per Site and Year",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=48, r=16, t=40, b=80),
    ))
    fig.update_xaxes(type="category")
    return fig


def fig_citations_per_year(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing total citations grouped by the publication year of the cited articles."""
    if not has_citations(df):
        return empty_figure("Citations by Publication Year", "No citation data available")
    d = df.groupby("publication_year")["cited_by_count"].sum().reset_index()
    d = year_as_category(d)
    fig = go.Figure()
    fig.add_bar(x=d["publication_year"], y=d["cited_by_count"],
                marker_color=PURPLE, marker_line_width=1, marker_line_color=SURFACE,
                hovertemplate="<b>%{x}</b><br>Citations: %{y:,}<extra></extra>")
    fig.update_layout(**plot_layout(title="Citations by Publication Year"))
    fig.update_xaxes(type="category")
    return fig


def fig_sjr_quartile(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing the distribution of publications across SJR quartiles."""
    if not has_sjr(df) or "sjr_quartile" not in df.columns:
        return empty_figure("Publications by SJR Quartile", "No SJR data available")

    counts  = df["sjr_quartile"].value_counts()
    unknown = [q for q in counts.index if q not in QUARTILE_ORDER]
    keys    = [k for k in QUARTILE_ORDER + unknown if k in counts.index]
    values  = counts.reindex(keys).values
    colors  = [QUARTILE_COLORS.get(q, MUTED) for q in keys]
    labels  = [q if q in QUARTILE_COLORS else NO_QUARTILE_LABEL for q in keys]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        sort=False,
        marker=dict(colors=colors, line=dict(color=SURFACE, width=2)),
        textfont=dict(color=TEXT, size=12),
        texttemplate="%{percent:.1%}",
        hovertemplate="<b>%{label}</b><br>Publications: %{value:,}<br>Share: %{percent:.1%}<extra></extra>",
    ))
    fig.update_layout(**plot_layout(
        title="Publications by SJR Quartile",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=16, r=16, t=40, b=48),
    ))
    return fig


def fig_sjr_quartile_per_year(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing the SJR quartile distribution per publication year."""
    if not has_sjr(df) or "sjr_quartile" not in df.columns:
        return empty_figure("SJR Quartile per Year", "No SJR data available")

    d = (
        df.dropna(subset=["sjr_quartile"])
        .groupby(["publication_year", "sjr_quartile"])
        .size()
        .reset_index(name="n")
    )
    if d.empty:
        return empty_figure("SJR Quartile per Year", "No SJR data available")

    d = year_as_category(d)
    present  = [q for q in QUARTILE_ORDER if q in d["sjr_quartile"].values]
    present += [q for q in d["sjr_quartile"].unique() if q not in QUARTILE_ORDER]

    fig = go.Figure()
    for q in present:
        sub   = d[d["sjr_quartile"] == q]
        label = q if q in QUARTILE_COLORS else NO_QUARTILE_LABEL
        fig.add_bar(
            x=sub["publication_year"], y=sub["n"],
            name=label,
            marker_color=QUARTILE_COLORS.get(q, MUTED),
            marker_line_width=0,
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Publications: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(**plot_layout(
        title="SJR Quartile per Year",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=48, r=16, t=40, b=80),
    ))
    fig.update_xaxes(type="category")
    return fig


def fig_sjr_per_year(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing the average SJR value of published journals per year."""
    if not has_sjr(df):
        return empty_figure("Average SJR per Year", "No SJR data available")
    d = df.dropna(subset=["sjr"]).groupby("publication_year")["sjr"].mean().round(2).reset_index()
    d = year_as_category(d)
    fig = go.Figure()
    fig.add_bar(x=d["publication_year"], y=d["sjr"],
                marker_color=BLUE, marker_line_width=1, marker_line_color=SURFACE,
                hovertemplate="<b>%{x}</b><br>Average SJR: %{y:.2f}<extra></extra>")
    fig.update_layout(**plot_layout(title="Average SJR per Year"))
    fig.update_xaxes(type="category")
    return fig


def fig_co_affiliation(df: pd.DataFrame, selected_dzg: str) -> go.Figure:
    """Bar chart showing how many publications of the selected DZG are also affiliated with each other DZG."""
    other_dzgs = [d for d in DZG_COLUMNS if d != selected_dzg and d in df.columns]
    counts = {d: int((df[d] == True).sum()) for d in other_dzgs}
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        return empty_figure("Co-affiliation with Other DZGs", "No co-affiliations with other DZGs")
    dzgs   = list(counts.keys())
    values = list(counts.values())
    colors = [DZG_COLORS.get(d, BLUE) for d in dzgs]
    fig = go.Figure()
    fig.add_bar(
        x=dzgs, y=values,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Publications: %{y:,}<extra></extra>",
    )
    fig.update_layout(**plot_layout(showlegend=False, title="Co-affiliation with Other DZGs"))
    return fig


def fig_authors_per_article_year(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing the average total number of authors per publication, grouped by year."""
    avg_authors = df.groupby("publication_year")["n_authors"].mean().reset_index()
    avg_authors = year_as_category(avg_authors)
    fig = go.Figure()
    fig.add_bar(x=avg_authors["publication_year"], y=avg_authors["n_authors"].round(1),
                marker_color=BLUE, marker_line_width=1, marker_line_color=SURFACE,
                hovertemplate="<b>%{x}</b><br>Average Authors: %{y:.1f}<extra></extra>")
    fig.update_layout(**plot_layout(title="Average Authors per Publication by Year"))
    fig.update_xaxes(type="category")
    return fig


def get_site_columns(df: pd.DataFrame, dzg: str) -> list[str]:
    """Return site-level column names for a DZG (e.g. DZL_ARCN, DZL_BREATH)."""
    return [c for c in df.columns if c.startswith(f"{dzg}_")]


def fig_sites_per_article(df: pd.DataFrame, site_columns: list[str]) -> go.Figure:
    """Pie chart showing how many publications involve 0, 1, 2, and more sites."""
    if not site_columns:
        return None
    n_sites = df[site_columns].apply(lambda row: (row == True).sum(), axis=1)
    counts  = n_sites.value_counts().sort_index()
    labels  = [f"{n} Site{'s' if n != 1 else ''}" for n in counts.index]
    colors  = [FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)] for i in range(len(labels))]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=0,
        sort=False,
        marker=dict(colors=colors, line=dict(color=SURFACE, width=2)),
        textfont=dict(color=TEXT, size=12),
        texttemplate="%{percent:.1%}",
        hovertemplate="<b>%{label}</b><br>Publications: %{value:,}<br>Share: %{percent:.1%}<extra></extra>",
    ))
    fig.update_layout(**plot_layout(
        title="Number of Sites per Publication",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=16, r=16, t=40, b=48),
    ))
    return fig


def fig_site_heatmap(df: pd.DataFrame, site_columns: list[str]) -> go.Figure:
    """Heatmap of shared publications between site pairs; the diagonal is excluded from the colour scale."""
    if not site_columns or len(site_columns) < 2:
        return None
    site_names = [c.split("_", 1)[1] for c in site_columns]
    n = len(site_columns)

    # z drives the colour scale and leaves the diagonal empty so pair values stay distinguishable
    z    = np.full((n, n), np.nan)
    text = np.empty((n, n), dtype=object)

    for i, col_i in enumerate(site_columns):
        for j, col_j in enumerate(site_columns):
            if i == j:
                total = int((df[col_i] == True).sum())
                text[i][j] = f"{total:,}"          # own total shown but not colour scaled
            else:
                shared = int(((df[col_i] == True) & (df[col_j] == True)).sum())
                z[i][j]    = shared
                text[i][j] = f"{shared:,}"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=site_names, y=site_names,
        text=text,
        texttemplate="%{text}",
        textfont=dict(color=TEXT, size=11),
        colorscale=[[0, "#4a1010"], [0.5, "#c0392b"], [1, "#ff6b6b"]],
        hovertemplate="<b>%{y} × %{x}</b><br>Shared publications: %{text}<extra></extra>",
        showscale=True,
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="Shared", font=dict(color=TEXT)),
            tickfont=dict(color=TEXT),
        ),
    ))
    fig.update_layout(**plot_layout(
        title="Site Co-affiliation Heatmap",
        margin=dict(l=80, r=16, t=40, b=80),
    ))
    return fig


def fig_site_venn(df: pd.DataFrame, site_columns: list[str]) -> plt.Figure | None:
    """Venn diagram showing publication overlap between sites; supports 2 to 6 sites via the venn library."""
    if not site_columns or len(site_columns) < 2:
        return None
    try:
        from venn import venn as venn_plot
    except ImportError:
        return None

    # The venn library supports at most 6 sets, so the largest sites are kept
    # rather than whichever ones happen to come first in the CSV column order
    ranked = sorted(site_columns, key=lambda c: int((df[c] == True).sum()), reverse=True)
    cols   = ranked[:6]

    sets    = {c.split("_", 1)[1]: set(df[df[c] == True].index) for c in cols}
    palette = [SITE_COLORS.get(c.split("_", 1)[1], BLUE) for c in cols]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    from matplotlib.colors import ListedColormap
    venn_plot(sets, ax=ax, cmap=ListedColormap(palette), alpha=0.5, fontsize=9)

    ax.set_title("Site Co-authorship (Venn)", color=TEXT, fontsize=12, pad=12)
    for text in ax.texts:
        text.set_color(TEXT)

    if len(site_columns) > 6:
        omitted = [c.split("_", 1)[1] for c in ranked[6:]]
        ax.set_xlabel(
            f"Showing the 6 largest of {len(site_columns)} sites. Not shown: {', '.join(omitted)}",
            color=MUTED, fontsize=8,
        )

    plt.tight_layout()
    return fig


def fig_mesh_top_n(mesh_counts: pd.Series, top_n: int) -> go.Figure:
    """Horizontal bar chart of the top N most frequent MeSH descriptors."""
    if mesh_counts.empty:
        return empty_figure(f"Top {top_n} MeSH Terms", "No MeSH data available, re-run 02_preprocessing.py")
    top = mesh_counts.head(top_n).sort_values()
    fig = go.Figure()
    fig.add_bar(
        x=top.values, y=top.index,
        orientation="h",
        marker_color=GREEN,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Publications: %{x:,}<extra></extra>",
    )
    fig.update_layout(**plot_layout(
        title=f"Top {top_n} MeSH Terms",
        margin=dict(l=280, r=16, t=40, b=48),
    ))
    return fig


def fig_mesh_trend(df_mesh_filtered: pd.DataFrame, terms: list[str]) -> go.Figure:
    """Line chart showing the publication count per year for each selected MeSH term."""
    if df_mesh_filtered.empty or not terms:
        return empty_figure("MeSH Term Trend over Time", "Select at least one MeSH term above")

    trend = (
        df_mesh_filtered[df_mesh_filtered["mesh_term"].isin(terms)]
        .groupby(["publication_year", "mesh_term"])
        .size()
        .reset_index(name="n")
    )
    if trend.empty:
        return empty_figure("MeSH Term Trend over Time", "No data for selected terms in this time range")

    trend = year_as_category(trend, column="publication_year")
    fig = px.line(trend, x="publication_year", y="n", color="mesh_term",
                  markers=True,
                  labels={"publication_year": "Year", "n": "Publications", "mesh_term": "MeSH Term"})
    fig.update_layout(**plot_layout(title="MeSH Term Trend over Time"))
    fig.update_xaxes(type="category")
    return fig


# Application

with st.spinner("Loading data ..."):
    df_articles, df_authors, df_mesh, last_prepared, last_extracted = load_data()

selected_dzg = load_active_dzg(YAML_PATH)

# Resolve colour maps once so every chart uses the same colour per DZG and per site
_cfg_dzg_colors, _cfg_site_colors = load_color_config(YAML_PATH)
DZG_COLORS = resolve_colors(DZG_COLUMNS, _cfg_dzg_colors)
SITE_COLORS = resolve_colors(
    [c[len(selected_dzg) + 1:] for c in df_articles.columns if c.startswith(f"{selected_dzg}_")],
    _cfg_site_colors.get(selected_dzg, {}),
)

with st.sidebar:
    st.markdown(f'<div style="color:{MUTED};font-size:10px;letter-spacing:0.15em;'
                f'text-transform:uppercase;margin-bottom:4px">Filter</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # DZG is fixed from the YAML config and shown as info rather than a selector
    st.markdown(
        f'<div style="color:{MUTED};font-size:10px">Active DZG</div>'
        f'<div style="color:{TEXT};font-size:16px;font-weight:700;margin-bottom:12px">{selected_dzg}</div>',
        unsafe_allow_html=True,
    )

    # site-level columns follow the "DZG_site" naming pattern from the data pipeline
    site_columns = [c for c in df_articles.columns if c.startswith(f"{selected_dzg}_")]
    sites = sorted(c[len(selected_dzg) + 1:] for c in site_columns)
    if sites:
        selected_site = st.selectbox("Site", ["All Sites"] + sites)
    else:
        selected_site = "All Sites"

    years_in_data = df_articles["publication_year"].dropna().astype(int)
    year_min = int(years_in_data.min()) if not years_in_data.empty else 2005
    year_max = int(years_in_data.max()) if not years_in_data.empty else 2026
    year_range = st.slider("Time Range", year_min, year_max, (2010, year_max))

    st.markdown("---")
    st.markdown(
        f'<div style="color:{MUTED};font-size:10px">Data Extracted<br>'
        f'<span style="color:{TEXT}">{last_extracted}</span></div>'
        f'<div style="color:{MUTED};font-size:10px;margin-top:8px">Data Processed<br>'
        f'<span style="color:{TEXT}">{last_prepared}</span></div>',
        unsafe_allow_html=True,
    )

# Filtering

df_filtered = filter_by_dzg(df_articles, selected_dzg)
if selected_site != "All Sites":
    site_column = f"{selected_dzg}_{selected_site}"
    if site_column in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[site_column] == True]
df_filtered = df_filtered[df_filtered["publication_year"].between(year_range[0], year_range[1], inclusive="both")]

df_authors_filtered     = df_authors[df_authors["pmid"].isin(df_filtered["pmid"])]
df_authors_filtered_dzg = df_authors_filtered[df_authors_filtered[selected_dzg] == True] if selected_dzg in df_authors.columns else df_authors_filtered

df_mesh_filtered = df_mesh[
    df_mesh["pmid"].isin(df_filtered["pmid"]) &
    df_mesh["publication_year"].between(year_range[0], year_range[1])
] if not df_mesh.empty else df_mesh

# Site columns for the selected DZG (only present if sub-networks are defined in the YAML)
site_cols_filtered = get_site_columns(df_filtered, selected_dzg)

# MeSH stoplist and term counts
mesh_stoplist = load_mesh_stoplist(MESH_STOPLIST_PATH)
mesh_counts_full = (
    df_mesh_filtered["mesh_term"].value_counts()
    if not df_mesh_filtered.empty and "mesh_term" in df_mesh_filtered.columns
    else pd.Series(dtype=int)
)

# Shared statistics used across tabs
total_pubs   = len(df_filtered)
n_coauthored = int((df_filtered["n_authors"].fillna(0) > 1).sum())
n_years      = max(1, year_range[1] - year_range[0] + 1)

n_leadership     = len(leadership_pmids(df_authors_filtered_dzg, selected_dzg))
share_leadership = round(n_leadership / total_pubs * 100, 1) if total_pubs > 0 else None

# Page header

st.markdown(
    f'<h2 style="color:{TEXT};font-weight:600;margin-bottom:0">Publication Analysis</h2>'
    f'<div style="color:{MUTED};font-size:12px;margin-bottom:24px">'
    f'{selected_dzg}{" – " + selected_site if selected_site != "All Sites" else ""} &nbsp;·&nbsp; '
    f'{year_range[0]}–{year_range[1]} &nbsp;·&nbsp; {format_number(total_pubs)} publications</div>',
    unsafe_allow_html=True,
)

tab_overview, tab_citations, tab_journal, tab_collaboration, tab_mesh = st.tabs(
    ["Overview", "Citations", "Journal Metrics", "Collaboration", "MeSH Terms"]
)

# Tab: Overview

with tab_overview:
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Total Publications",           format_number(total_pubs))
    kpi(c2, "Average Publications per Year", format_number(total_pubs / n_years))
    kpi(c3, "First or Last Author",
        f"{format_number(share_leadership, 1)}%" if share_leadership is not None else "–",
        sub=f"publications with {selected_dzg} as first or last author")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.plotly_chart(fig_pubs_per_year(df_filtered, df_authors_filtered_dzg, selected_dzg), width="stretch")

    if site_cols_filtered:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.plotly_chart(fig_pubs_per_site_year(df_filtered, site_cols_filtered), width="stretch")
        data_note("A publication is counted once for every site involved, so publications written jointly "
                  "by several sites appear in more than one bar segment and the stacked totals can exceed "
                  "the actual number of publications in that year.")

# Tab: Citations

with tab_citations:
    info_box("About the Citation Data", """
        Citation counts come from NIH iCite, an open service run by the National Institutes of Health.
        It indexes every article listed in PubMed and recalculates the citation figures on a regular
        schedule, so each publication in this dashboard is matched to its iCite record by PMID.
    """)

    if has_citations(df_filtered):
        total_citations = int(df_filtered["cited_by_count"].sum())
        avg_citations   = round(df_filtered["cited_by_count"].mean(), 1)
        h               = h_index(df_filtered["cited_by_count"])
        m_quotient      = round(h / n_years, 2)
    else:
        total_citations = avg_citations = h = m_quotient = None

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Citations",               format_number(total_citations))
    kpi(c2, "Average Citations per Publication", format_number(avg_citations, 1))
    kpi(c3, "h-Index",                 format_number(h), sub="h publications with at least h citations each")
    kpi(c4, "m-Quotient",              format_number(m_quotient, 2), sub="h-Index divided by years in range")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.plotly_chart(fig_citations_per_year(df_filtered), width="stretch")

    if has_citations(df_filtered):
        st.subheader("Most-Cited Publications")
        top_n = st.number_input("Number of publications shown", min_value=1, max_value=200, value=20, step=1)
        top_list = (df_filtered.nlargest(top_n, "cited_by_count")
                 [["pmid", "article_title", "publication_year", "cited_by_count", "pubmed_url"]]
                 .copy())
        top_list["cited_by_count"] = top_list["cited_by_count"].apply(format_number)
        top_list.columns = ["PMID", "Title", "Year", "Citations", "Link"]
        st.dataframe(top_list, width="stretch", hide_index=True,
                     column_config={
                         "Title": st.column_config.TextColumn("Title", width="large"),
                         "Link":  st.column_config.LinkColumn("Link"),
                     })
    else:
        st.info("No citation data available, run the data preparation pipeline first.")

# Tab: Journal Metrics

with tab_journal:
    info_box("Definitions", f"""
        <b style="color:{BLUE}">SJR</b> (SCImago Journal Rank) rates a journal's prestige.
        It weights citations by the standing of the citing journal. Published per journal and year by SCImago/Scopus.<br>
        <b style="color:{BLUE}">RCR</b> (Relative Citation Ratio) rates the impact of a single
        publication. It compares its citation rate to the average for its specific field, where
        1.0 equals the NIH field average. Published per publication by NIH iCite.
    """)

    if has_sjr(df_filtered):
        avg_sjr    = round(df_filtered["sjr"].mean(), 2)
        median_sjr = round(df_filtered["sjr"].median(), 2)
    else:
        avg_sjr = median_sjr = None

    if has_rcr(df_filtered):
        avg_rcr          = round(df_filtered["rcr"].mean(), 2)
        share_rcr_above1 = round((df_filtered["rcr"] > 1).sum() / df_filtered["rcr"].notna().sum() * 100, 2)
    else:
        avg_rcr = share_rcr_above1 = None

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Average SJR", format_number(avg_sjr, 2))
    kpi(c2, "Median SJR",  format_number(median_sjr, 2))
    kpi(c3, "Average RCR", format_number(avg_rcr, 2))
    kpi(c4, "RCR above 1", f"{format_number(share_rcr_above1, 2)}%" if share_rcr_above1 is not None else "–",
        sub="share of publications above the NIH field average")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_sjr_quartile(df_filtered), width="stretch")
    with col2:
        st.plotly_chart(fig_sjr_quartile_per_year(df_filtered), width="stretch")
    data_note("Publications whose journal could not be matched by ISSN are not shown.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.plotly_chart(fig_sjr_per_year(df_filtered), width="stretch")

# Tab: Collaboration

with tab_collaboration:
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Total Publications",       format_number(total_pubs))
    kpi(c2, "Co-Authored Publications", format_number(n_coauthored),
        sub="publications written by more than one author")
    kpi(c3, "First or Last Author",
        f"{format_number(share_leadership, 1)}%" if share_leadership is not None else "–",
        sub=f"publications with {selected_dzg} as first or last author")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Section 1: sites
    if site_cols_filtered:
        st.markdown('<div class="section-label">Site Collaboration</div>', unsafe_allow_html=True)

        venn_fig = fig_site_venn(df_filtered, site_cols_filtered)
        if venn_fig:
            _, venn_col, _ = st.columns([1, 2, 1])
            with venn_col:
                st.pyplot(venn_fig, use_container_width=True)
            plt.close(venn_fig)
        else:
            data_note("Install the venn library to enable the Venn diagram: pip install venn")

        col1, col2 = st.columns(2)
        with col1:
            f = fig_sites_per_article(df_filtered, site_cols_filtered)
            if f:
                st.plotly_chart(f, width="stretch")
        with col2:
            heat_fig = fig_site_heatmap(df_filtered, site_cols_filtered)
            if heat_fig:
                st.plotly_chart(heat_fig, width="stretch")
        data_note("The heatmap diagonal shows each site's own total and stays outside the colour scale "
                  "so the shared counts between different sites remain distinguishable.")

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Section 2: other DZGs
    st.markdown('<div class="section-label">Collaboration with Other DZGs</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_co_affiliation(df_filtered, selected_dzg), width="stretch")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Section 3: team size
    st.markdown('<div class="section-label">Team Size</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_authors_per_article_year(df_filtered), width="stretch")
    data_note("Includes all authors on each publication, not only DZG-affiliated authors.")

# Tab: MeSH Terms

with tab_mesh:
    info_box("About MeSH Terms", """
        MeSH data is pre-processed during data preparation and stored in pubmed_mesh.csv.<br>
        Non-specific terms such as <i>Humans</i>, <i>Male</i> or <i>Female</i> can be hidden
        via the stoplist defined in <code>mesh_stoplist.yaml</code>.
    """)

    if mesh_counts_full.empty:
        st.info("No MeSH data available, re-run 02_preprocessing.py to generate pubmed_mesh.csv.")
    else:
        filter_stopwords = st.checkbox(
            f"Hide non-specific terms ({len(mesh_stoplist)} terms in stoplist from mesh_stoplist.yaml)",
            value=True,
        )
        mesh_counts = (
            mesh_counts_full[~mesh_counts_full.index.isin(mesh_stoplist)]
            if filter_stopwords else mesh_counts_full
        )

        top_n_mesh = st.slider("Number of top terms to display", min_value=5, max_value=50,
                               value=20, step=5)
        st.plotly_chart(fig_mesh_top_n(mesh_counts, top_n_mesh), width="stretch")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="section-label">MeSH Term Trend Analysis</div>',
                    unsafe_allow_html=True)
        top_100_terms = mesh_counts.head(100).index.tolist()
        selected_terms = st.multiselect(
            "Select MeSH terms to track over time (top 100 shown):",
            options=top_100_terms,
            default=top_100_terms[:5],
        )
        st.plotly_chart(fig_mesh_trend(df_mesh_filtered, selected_terms), width="stretch")
        data_note("Only publications with MeSH descriptors assigned by PubMed are included.")