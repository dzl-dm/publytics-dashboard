"""
DZG Publication Dashboard: interactive analysis of PubMed publication data for one DZG.

Reads the CSV files produced by 02_preprocessing.py and displays six tabs:
Overview, Citations, Journal Metrics, Collaboration, MeSH Terms and Raw Data.
The active DZG is read from dzg_search_terms.yaml (active_dzg key).

Requirements: see requirements.txt
"""

import locale
import logging
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

# Progress and problems are logged rather than printed so they appear in the terminal
# that runs Streamlit, where print output from cached functions is easy to miss
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(Path(__file__).stem)

# Locale names differ between operating systems, so several spellings are tried in order
for _loc in ("en_US.UTF-8", "en_US", "English_United States.1252", "English"):
    try:
        locale.setlocale(locale.LC_ALL, _loc)
        break
    except locale.Error:
        continue


# Paths resolve relative to this script, so the project runs from any location.
# The scripts and the YAML config sit in the repository root, the data subfolder
# holds every CSV and is created on first run.
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
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

# Site filter options that are not an actual site name
SITE_ALL  = "All"
SITE_NONE = "No Sites"

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


def table_column_config(df: pd.DataFrame) -> dict:
    """Build a Streamlit column configuration so links open and numbers stay readable."""
    config = {}
    for col in df.columns:
        if col in ("url", "pubmed_url"):
            config[col] = st.column_config.LinkColumn(col, display_text="open", width="small")
        elif col in ("article_title", "journal_title", "affiliation", "mesh_descriptor"):
            config[col] = st.column_config.TextColumn(col, width="large")
        elif col in ("cited_by_count", "n_authors", "n_dzg_authors", "author_position"):
            config[col] = st.column_config.NumberColumn(col, format="%d")
        elif col in ("rcr", "sjr"):
            config[col] = st.column_config.NumberColumn(col, format="%.2f")
    return config


def filter_table(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Narrow a table one column at a time.

    The widget follows the column content. Columns with few distinct values get a
    multiselect, numeric columns with a wide spread get a range slider, and anything
    else gets a case insensitive substring search.
    """
    with st.expander("Filter columns"):
        chosen = st.multiselect(
            "Pick the columns you want to filter on",
            options=list(df.columns),
            key=f"{key_prefix}_filter_cols",
        )
        for col in chosen:
            series = df[col]
            values = series.dropna()
            if values.empty:
                continue

            numeric        = pd.to_numeric(series, errors="coerce")
            mostly_numeric = numeric.notna().sum() >= 0.9 * values.size
            has_spread     = mostly_numeric and numeric.min() < numeric.max()

            if values.nunique() <= 25:
                options  = sorted(values.astype(str).unique().tolist())
                selected = st.multiselect(col, options, default=options, key=f"{key_prefix}_{col}")
                df = df[series.astype(str).isin(selected)]
            elif has_spread:
                low, high = float(numeric.min()), float(numeric.max())
                lo, hi = st.slider(col, low, high, (low, high), key=f"{key_prefix}_{col}")
                df = df[numeric.between(lo, hi)]
            else:
                text = st.text_input(f"{col} contains", key=f"{key_prefix}_{col}")
                if text:
                    df = df[series.astype(str).str.contains(text, case=False, na=False)]
    return df


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
        log.warning("active_dzg is missing or unknown in %s, falling back to %s",
                    path.name, DZG_COLUMNS[0])
    except Exception as error:
        log.warning("Could not read %s (%s), falling back to %s", path.name, error, DZG_COLUMNS[0])
    return DZG_COLUMNS[0]


@st.cache_data
def load_color_config(path: Path) -> tuple[dict, dict]:
    """Read the colors block from the YAML config and return (dzg_colors, site_colors_by_dzg)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as error:
        log.warning("Could not read the colours from %s (%s), using the built-in palette",
                    path.name, error)
        return {}, {}

    colors = raw.get("colors") or {}
    return colors.get("dzg") or {}, colors.get("sites") or {}


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
    """Return the timestamp of the most recent pipeline run.

    The file is read line by line instead of as a table, because the pipeline appends to it
    over time and rows written by an older version can carry fewer columns than the header.
    """
    try:
        rows = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except OSError as error:
        log.warning("Could not read %s: %s", path.name, error)
        return "unknown"

    if len(rows) < 2:          # only the header, no run recorded yet
        return "unknown"
    return rows[-1].split(";")[0]


@st.cache_data
def load_mesh_stoplist(path: Path) -> frozenset[str]:
    """Load MeSH stoplist from YAML and return all terms as a frozenset; returns empty set if file is missing."""
    if not path.exists():
        log.warning("%s not found, no MeSH terms will be filtered", path.name)
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
        log.warning("%s not found, the MeSH tab stays empty", CSV_MESH.name)
        df_mesh = pd.DataFrame(columns=["pmid", "publication_year", "mesh_term"])

    for col in ["n_authors", "n_dzg_authors", "publication_year", "cited_by_count", "rcr", "sjr"]:
        if col in df_articles.columns:
            df_articles[col] = pd.to_numeric(df_articles[col], errors="coerce")

    if "author_position" in df_authors.columns:
        df_authors["author_position"] = pd.to_numeric(df_authors["author_position"], errors="coerce")

    # The pipeline writes the flags as True/False text. Turning them into real booleans
    # here means every later check can use the column directly instead of comparing values.
    TRUE_FALSE = {"True": True, "False": False, True: True, False: False}
    flag_columns = ["is_first_author", "is_last_author"] + DZG_COLUMNS
    for df in (df_articles, df_authors):
        for column in df.columns:
            is_site_flag = any(column.startswith(f"{dzg}_") for dzg in DZG_COLUMNS)
            if column in flag_columns or is_site_flag:
                df[column] = df[column].map(TRUE_FALSE).fillna(False).astype(bool)

    if "publication_year" in df_mesh.columns:
        df_mesh["publication_year"] = pd.to_numeric(df_mesh["publication_year"], errors="coerce")

    last_prepared  = load_last_update(CSV_METADATA)
    last_extracted = load_last_update(CSV_METADATA_EXTRACTION)
    log.info("Loaded %s articles, %s author rows and %s MeSH rows",
             f"{len(df_articles):,}", f"{len(df_authors):,}", f"{len(df_mesh):,}")
    return df_articles, df_authors, df_mesh, last_prepared, last_extracted


def filter_by_dzg(df: pd.DataFrame, dzg: str) -> pd.DataFrame:
    """Return only rows where the given DZG column is True; returns df unchanged if the column is missing."""
    if dzg not in df.columns:
        return df
    return df[df[dzg]]


def has_citations(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame contains at least one non-null citation count."""
    return "cited_by_count" in df.columns and df["cited_by_count"].notna().any()


def has_sjr(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame contains at least one non-null SJR value."""
    return "sjr" in df.columns and df["sjr"].notna().any()


def has_rcr(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame contains at least one non-null RCR value."""
    return "rcr" in df.columns and df["rcr"].notna().any()


def site_of(column: str) -> str:
    """Return the site name behind a column such as DZL_ARCN."""
    return column.split("_", 1)[1]


def get_site_columns(df: pd.DataFrame, dzg: str) -> list[str]:
    """Return site-level column names for a DZG (e.g. DZL_ARCN, DZL_BREATH)."""
    return [column for column in df.columns if column.startswith(f"{dzg}_")]


def use_year_axis(fig: go.Figure) -> go.Figure:
    """Draw the x axis as one sorted category per year.

    A category axis avoids decimal tick labels such as 2020.5 when only one or two years
    are selected. Plotly otherwise arranges categories by first appearance, which in a
    stacked chart depends on the order the traces were added and puts years out of
    sequence, so the order is stated explicitly.
    """
    fig.update_xaxes(type="category", categoryorder="category ascending")
    return fig


def leadership_pmids(df_authors: pd.DataFrame, dzg: str) -> set:
    """Return the PMIDs where at least one DZG-affiliated author is first or last author."""
    required = {"is_first_author", "is_last_author", dzg}
    if not required.issubset(df_authors.columns):
        return set()
    leading = df_authors[dzg] & (df_authors["is_first_author"] | df_authors["is_last_author"])
    return set(df_authors.loc[leading, "pmid"])


def authors_by_mesh(df_mesh: pd.DataFrame, df_authors: pd.DataFrame, df_articles: pd.DataFrame,
                    terms: list[str], leading_only: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find the authors who published on every selected MeSH term, plus their publications.

    The terms are combined per author, not per publication. Someone who covered one term in
    one paper and another term in a second paper still counts.

    Authors are grouped by the author_key column that preprocessing writes, which joins
    last name and initials. Returns the author table, the publication table, and the pairs
    of author and publication that connect the two, so the interface can narrow the
    publications to one author.
    """
    empty = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not terms or df_mesh.empty or df_authors.empty:
        return empty

    if "author_key" not in df_authors.columns:
        log.warning("The author file has no author_key column")
        return empty

    authors = df_authors
    if leading_only and {"is_first_author", "is_last_author"}.issubset(authors.columns):
        authors = authors[authors["is_first_author"] | authors["is_last_author"]]
    if authors.empty:
        return empty

    # One author has to appear under every single term, so the sets are intersected
    qualified = None
    for term in terms:
        term_pmids = set(df_mesh.loc[df_mesh["mesh_term"] == term, "pmid"])
        term_keys  = set(authors.loc[authors["pmid"].isin(term_pmids), "author_key"])
        qualified  = term_keys if qualified is None else qualified & term_keys
    if not qualified:
        return empty

    # Their publications are those carrying at least one of the terms
    all_pmids = set(df_mesh.loc[df_mesh["mesh_term"].isin(terms), "pmid"])
    hits = authors[authors["author_key"].isin(qualified) & authors["pmid"].isin(all_pmids)]

    leading = pd.Series(False, index=hits.index)
    if {"is_first_author", "is_last_author"}.issubset(hits.columns):
        leading = hits["is_first_author"] | hits["is_last_author"]

    years = df_articles.set_index("pmid")["publication_year"]
    hits  = hits.assign(publication_year=hits["pmid"].map(years), leading=leading)

    # Every series below shares the author_key index, so the columns line up on their own
    grouped        = hits.groupby("author_key")
    leading_counts = hits[hits["leading"]].groupby("author_key")["pmid"].nunique()

    summary = pd.DataFrame({
        "Author":                  grouped["last_name"].first() + ", " + grouped["initials"].first(),
        "Publications":            grouped["pmid"].nunique(),
        "As first or last author": leading_counts.reindex(grouped.size().index).fillna(0).astype(int),
        "From":                    grouped["publication_year"].min(),
        "To":                      grouped["publication_year"].max(),
        "ORCID":                   grouped["author_identifier"].first(),
    })
    summary = (summary.sort_values("Publications", ascending=False)
                      .reset_index()
                      .rename(columns={"index": "author_key"}))

    publications = df_articles[df_articles["pmid"].isin(hits["pmid"].unique())]
    return summary, publications, hits[["author_key", "pmid"]]


def fig_pubs_per_year(df: pd.DataFrame, df_authors: pd.DataFrame, dzg: str) -> go.Figure:
    """Area chart of publications per year with the first or last author publications as a subset."""
    total_per_year = df.groupby("publication_year").size().rename("total")
    leading        = df[df["pmid"].isin(leadership_pmids(df_authors, dzg))]
    leading_per_year = leading.groupby("publication_year").size().rename("leading")

    per_year = pd.concat([total_per_year, leading_per_year], axis=1).fillna(0).reset_index()
    per_year["share"] = (per_year["leading"] / per_year["total"] * 100).round(1)

    fig = go.Figure()
    # Total publications drawn first so the subset area sits visually inside it
    fig.add_scatter(
        x=per_year["publication_year"], y=per_year["total"],
        name="Total Publications",
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.18)",
        line=dict(color=BLUE, width=2),
        marker=dict(color=BLUE, size=6),
        hovertemplate="<b>%{x}</b><br>Total Publications: %{y:,}<extra></extra>",
    )
    fig.add_scatter(
        x=per_year["publication_year"], y=per_year["leading"],
        name=f"{dzg} First or Last Author",
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(245,166,35,0.30)",
        line=dict(color=ORANGE, width=2),
        marker=dict(color=ORANGE, size=5),
        customdata=per_year["share"],
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
    return use_year_axis(fig)


def fig_pubs_per_site_year(df: pd.DataFrame, site_columns: list[str]) -> go.Figure:
    """Stacked bar chart showing publications per site and year, including those without a site."""
    if not site_columns:
        return empty_figure("Publications per Site and Year", "No sites are defined for this DZG")
    rows = []
    for col in site_columns:
        site   = site_of(col)
        counts = df[df[col]].groupby("publication_year").size()
        rows += [{"publication_year": y, "site": site, "n": n} for y, n in counts.items()]

    # Publications the DZG claims but no individual site does
    no_site = df[~df[site_columns].any(axis=1)].groupby("publication_year").size()
    rows += [{"publication_year": y, "site": SITE_NONE, "n": n} for y, n in no_site.items()]

    if not rows:
        return empty_figure("Publications per Site and Year", "No site data available")

    per_site_year = pd.DataFrame(rows)
    # Named sites alphabetically, the residual category always last
    order = sorted(site for site in per_site_year["site"].unique() if site != SITE_NONE)
    if SITE_NONE in per_site_year["site"].values:
        order.append(SITE_NONE)

    fig = go.Figure()
    for site in order:
        sub = per_site_year[per_site_year["site"] == site]
        fig.add_bar(
            x=sub["publication_year"], y=sub["n"],
            name=site,
            marker_color=MUTED if site == SITE_NONE else SITE_COLORS.get(site, BLUE),
            marker_line_width=0,
            hovertemplate=f"<b>{site}</b><br>%{{x}}<br>Publications: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(**plot_layout(
        title="Publications per Site and Year",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=48, r=16, t=40, b=80),
    ))
    return use_year_axis(fig)


def fig_citations_per_year(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing total citations grouped by the publication year of the cited articles."""
    if not has_citations(df):
        return empty_figure("Citations by Publication Year", "No citation data available")
    per_year = df.groupby("publication_year")["cited_by_count"].sum().reset_index()
    fig = go.Figure()
    fig.add_bar(x=per_year["publication_year"], y=per_year["cited_by_count"],
                marker_color=PURPLE, marker_line_width=1, marker_line_color=SURFACE,
                hovertemplate="<b>%{x}</b><br>Citations: %{y:,}<extra></extra>")
    fig.update_layout(**plot_layout(title="Citations by Publication Year"))
    return use_year_axis(fig)


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

    per_quartile_year = (
        df.dropna(subset=["sjr_quartile"])
        .groupby(["publication_year", "sjr_quartile"])
        .size()
        .reset_index(name="n")
    )
    if per_quartile_year.empty:
        return empty_figure("SJR Quartile per Year", "No SJR data available")

    quartiles  = [q for q in QUARTILE_ORDER if q in per_quartile_year["sjr_quartile"].values]
    quartiles += [q for q in per_quartile_year["sjr_quartile"].unique() if q not in QUARTILE_ORDER]

    fig = go.Figure()
    for q in quartiles:
        sub   = per_quartile_year[per_quartile_year["sjr_quartile"] == q]
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
    return use_year_axis(fig)


def fig_sjr_per_year(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing the average SJR value of published journals per year."""
    if not has_sjr(df):
        return empty_figure("Average SJR per Year", "No SJR data available")
    per_year = df.dropna(subset=["sjr"]).groupby("publication_year")["sjr"].mean().round(2).reset_index()
    fig = go.Figure()
    fig.add_bar(x=per_year["publication_year"], y=per_year["sjr"],
                marker_color=BLUE, marker_line_width=1, marker_line_color=SURFACE,
                hovertemplate="<b>%{x}</b><br>Average SJR: %{y:.2f}<extra></extra>")
    fig.update_layout(**plot_layout(title="Average SJR per Year"))
    return use_year_axis(fig)


def fig_co_affiliation(df: pd.DataFrame, selected_dzg: str) -> go.Figure:
    """Bar chart showing how many publications of the selected DZG are also affiliated with each other DZG."""
    other_dzgs = [dzg for dzg in DZG_COLUMNS if dzg != selected_dzg and dzg in df.columns]
    counts     = {dzg: int(df[dzg].sum()) for dzg in other_dzgs}
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        return empty_figure("Co-affiliation with Other DZGs", "No co-affiliations with other DZGs")
    dzgs   = list(counts.keys())
    values = list(counts.values())
    colors = [DZG_COLORS.get(dzg, BLUE) for dzg in dzgs]
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
    fig = go.Figure()
    fig.add_bar(x=avg_authors["publication_year"], y=avg_authors["n_authors"].round(1),
                marker_color=BLUE, marker_line_width=1, marker_line_color=SURFACE,
                hovertemplate="<b>%{x}</b><br>Average Authors: %{y:.1f}<extra></extra>")
    fig.update_layout(**plot_layout(title="Average Authors per Publication by Year"))
    return use_year_axis(fig)


def fig_sites_per_article(df: pd.DataFrame, site_columns: list[str]) -> go.Figure:
    """Pie chart showing how many publications involve 0, 1, 2 and more sites."""
    if not site_columns:
        return empty_figure("Number of Sites per Publication", "No sites are defined for this DZG")
    n_sites = df[site_columns].sum(axis=1)
    counts  = n_sites.value_counts().sort_index()
    labels  = [SITE_NONE if n == 0 else f"{n} Site{'s' if n != 1 else ''}" for n in counts.index]
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
    """Heatmap of shared publications between site pairs, with the diagonal outside the colour scale."""
    if len(site_columns) < 2:
        return empty_figure("Site Co-affiliation Heatmap", "At least two sites are needed for a comparison")
    site_columns = sorted(site_columns, key=lambda column: site_of(column).lower())
    site_names   = [site_of(column) for column in site_columns]
    n = len(site_columns)

    # z drives the colour scale and leaves the diagonal empty so pair values stay distinguishable
    z    = np.full((n, n), np.nan)
    text = np.empty((n, n), dtype=object)

    for i, col_i in enumerate(site_columns):
        for j, col_j in enumerate(site_columns):
            if i == j:
                total = int(df[col_i].sum())
                text[i][j] = f"{total:,}"          # own total shown but not colour scaled
            else:
                shared = int((df[col_i] & df[col_j]).sum())
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


def fig_site_venn(df: pd.DataFrame, site_columns: list[str]) -> tuple[plt.Figure | None, str]:
    """Venn diagram of the publication overlap between sites, for 2 to 6 sites.

    Returns the figure together with a message, so the caller can say why nothing is drawn
    instead of guessing a reason. Exactly one of the two is ever filled.
    """
    if len(site_columns) < 2:
        return None, "At least two sites are needed for a Venn diagram."
    try:
        from venn import venn as venn_plot
    except ImportError:
        return None, "Install the venn library to enable the Venn diagram: pip install venn"

    # The venn library supports at most 6 sets, so the largest sites are kept rather than
    # whichever come first in the CSV, then ordered alphabetically for a predictable layout
    ranked = sorted(site_columns, key=lambda column: int(df[column].sum()), reverse=True)
    shown  = sorted(ranked[:6], key=lambda column: site_of(column).lower())

    sets    = {site_of(column): set(df[df[column]].index) for column in shown}
    palette = [SITE_COLORS.get(site_of(column), BLUE) for column in shown]

    # The venn library divides by the size of the union when it computes the percentages
    if not set().union(*sets.values()):
        return None, "No publication in the current selection is assigned to a site."

    fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    from matplotlib.colors import ListedColormap
    venn_plot(sets, ax=ax, cmap=ListedColormap(palette), alpha=0.5, fontsize=9)

    ax.set_title("Site Co-authorship (Venn)", color=TEXT, fontsize=12, pad=12)
    for text in ax.texts:
        text.set_color(TEXT)

    if len(site_columns) > 6:
        omitted = [site_of(column) for column in ranked[6:]]
        ax.set_xlabel(
            f"Showing the 6 largest of {len(site_columns)} sites. Not shown: {', '.join(omitted)}",
            color=MUTED, fontsize=8,
        )

    plt.tight_layout()
    return fig, ""


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

    fig = px.line(trend, x="publication_year", y="n", color="mesh_term",
                  markers=True,
                  labels={"publication_year": "Year", "n": "Publications", "mesh_term": "MeSH Term"})
    fig.update_layout(**plot_layout(title="MeSH Term Trend over Time"))
    return use_year_axis(fig)


# Application

with st.spinner("Loading data ..."):
    df_articles, df_authors, df_mesh, last_prepared, last_extracted = load_data()

selected_dzg = load_active_dzg(YAML_PATH)

# Resolve colour maps once so every chart uses the same colour per DZG and per site
_cfg_dzg_colors, _cfg_site_colors = load_color_config(YAML_PATH)
DZG_COLORS = resolve_colors(DZG_COLUMNS, _cfg_dzg_colors)
SITE_COLORS = resolve_colors(
    [site_of(column) for column in get_site_columns(df_articles, selected_dzg)],
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
    site_columns = get_site_columns(df_articles, selected_dzg)
    sites        = sorted(site_of(column) for column in site_columns)
    if sites:
        selected_site = st.selectbox("Site", [SITE_ALL, SITE_NONE] + sites)
    else:
        selected_site = SITE_ALL

    years_in_data = df_articles["publication_year"].dropna().astype(int)
    year_min = int(years_in_data.min()) if not years_in_data.empty else 2005
    year_max = int(years_in_data.max()) if not years_in_data.empty else 2026

    year_range = st.slider("Time Range", year_min, year_max, (max(year_min, 2010), year_max))

    st.markdown("---")
    st.markdown(
        f'<div style="color:{MUTED};font-size:10px">Data Extracted<br>'
        f'<span style="color:{TEXT}">{last_extracted}</span></div>'
        f'<div style="color:{MUTED};font-size:10px;margin-top:8px">Data Processed<br>'
        f'<span style="color:{TEXT}">{last_prepared}</span></div>',
        unsafe_allow_html=True,
    )

# Filtering

# The sidebar defines what the whole dashboard shows
df_filtered = filter_by_dzg(df_articles, selected_dzg)
if selected_site == SITE_NONE:
    # Publications of the DZG that no individual site claims
    if site_columns:
        df_filtered = df_filtered[~df_filtered[site_columns].any(axis=1)]
elif selected_site != SITE_ALL:
    site_column = f"{selected_dzg}_{selected_site}"
    if site_column in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[site_column]]
df_filtered = df_filtered[df_filtered["publication_year"].between(year_range[0], year_range[1], inclusive="both")]

df_authors_filtered = df_authors[df_authors["pmid"].isin(df_filtered["pmid"])]
if selected_dzg in df_authors_filtered.columns:
    df_authors_filtered_dzg = df_authors_filtered[df_authors_filtered[selected_dzg]]
else:
    df_authors_filtered_dzg = df_authors_filtered

df_mesh_filtered = df_mesh[
    df_mesh["pmid"].isin(df_filtered["pmid"]) &
    df_mesh["publication_year"].between(year_range[0], year_range[1])
] if not df_mesh.empty else df_mesh

# Site columns for the selected DZG (only present if sub-networks are defined in the YAML).
# With the no-site filter active every site chart would be empty, so they are skipped.
site_cols_filtered = get_site_columns(df_filtered, selected_dzg)
show_site_charts   = bool(site_cols_filtered) and selected_site != SITE_NONE

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
    f'{selected_dzg}{" – " + selected_site if selected_site != SITE_ALL else ""} &nbsp;·&nbsp; '
    f'{year_range[0]}–{year_range[1]} &nbsp;·&nbsp; {format_number(total_pubs)} publications</div>',
    unsafe_allow_html=True,
)

# key plus on_change makes the tab bar remember which tab is open across reruns,
# so changing a filter in the sidebar does not throw the user back to the first tab
tab_overview, tab_citations, tab_journal, tab_collaboration, tab_mesh, tab_authors, tab_data = st.tabs(
    ["Overview", "Citations", "Journal Metrics", "Collaboration", "MeSH Terms",
     "Author Search", "Raw Data"],
    key="active_tab",
    on_change="rerun",
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

    st.plotly_chart(fig_pubs_per_year(df_filtered, df_authors_filtered_dzg, selected_dzg),
                    width="stretch")

    if show_site_charts:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.plotly_chart(fig_pubs_per_site_year(df_filtered, site_cols_filtered), width="stretch")
        data_note("A publication is counted once for every site involved, so work written jointly by "
                  f"several sites appears in more than one segment and a stacked bar can exceed the "
                  f"real number of publications that year. \"{SITE_NONE}\" covers publications that "
                  "carry the DZG affiliation without naming one of its sites.")

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
    else:
        total_citations = avg_citations = None

    c1, c2 = st.columns(2)
    kpi(c1, "Cumulative Citations",             format_number(total_citations))
    kpi(c2, "Average Citations per Publication", format_number(avg_citations, 1))

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

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    info_box("Definitions", f"""
        <b style="color:{BLUE}">SJR</b> (SCImago Journal Rank) rates a journal's prestige.
        It weights citations by the standing of the citing journal. Published per journal and year by SCImago/Scopus.<br>
        <b style="color:{BLUE}">RCR</b> (Relative Citation Ratio) rates the impact of a single
        publication. It compares its citation rate to the average for its specific field, where
        1.0 equals the NIH field average. Published per publication by NIH iCite.
    """)


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
    if selected_site == SITE_NONE:
        st.info(f'The site filter is set to "{SITE_NONE}", so the site charts are hidden. '
                f'Switch it back to "{SITE_ALL}" to compare the sites.')
    elif site_cols_filtered:
        st.markdown('<div class="section-label">Site Collaboration</div>', unsafe_allow_html=True)

        venn_fig, venn_message = fig_site_venn(df_filtered, site_cols_filtered)
        if venn_fig:
            _, venn_col, _ = st.columns([1, 2, 1])
            with venn_col:
                st.pyplot(venn_fig, width="stretch")
            plt.close(venn_fig)
        else:
            data_note(venn_message)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_sites_per_article(df_filtered, site_cols_filtered), width="stretch")
        with col2:
            st.plotly_chart(fig_site_heatmap(df_filtered, site_cols_filtered), width="stretch")
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


# Tab: Author Search

with tab_authors:
    info_box("Finding Colleagues by Topic", """
        Lists the DZG affiliated authors who have published on the selected MeSH terms.
        Terms are combined per author rather than per publication, so someone who covered
        one term in one paper and another term in a second paper still appears.
    """)

    if mesh_counts_full.empty:
        st.info("No MeSH data available, re-run 02_preprocessing.py to generate pubmed_mesh.csv.")
    else:
        hide_generic = st.checkbox(
            "Hide non-specific terms in the list below", value=True, key="author_search_stoplist"
        )
        offered = (mesh_counts_full[~mesh_counts_full.index.isin(mesh_stoplist)]
                   if hide_generic else mesh_counts_full)

        chosen_terms = st.multiselect(
            "MeSH terms, combined with AND",
            options=offered.index.tolist(),
            help="An author has to have published on every selected term, not necessarily "
                 "within the same publication.",
            key="author_search_terms",
        )
        leading_only = st.checkbox(
            "Only first or last author",
            value=False,
            key="author_search_leading",
        )

        if not chosen_terms:
            st.info("Select at least one MeSH term to search for.")
        else:
            found_authors, found_publications, author_links = authors_by_mesh(
                df_mesh_filtered, df_authors_filtered_dzg, df_filtered, chosen_terms, leading_only
            )

            if found_authors.empty:
                st.info("No author of the centre has published on all of the selected terms "
                        "within the current selection.")
            else:
                st.markdown('<div class="section-label">Authors</div>', unsafe_allow_html=True)
                st.caption(f"{format_number(len(found_authors))} authors, "
                           f"{format_number(len(found_publications))} publications. "
                           "Select a row to narrow the publications below to that author.")

                # The key carries the chosen terms, so a new search starts without an old
                # row still being selected
                selection = st.dataframe(
                    found_authors,
                    width="stretch",
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=f"author_table_{'_'.join(sorted(chosen_terms))}_{leading_only}",
                    column_order=[c for c in found_authors.columns if c != "author_key"],
                    column_config={
                        "Publications":            st.column_config.NumberColumn(format="%d"),
                        "As first or last author": st.column_config.NumberColumn(format="%d"),
                        "From":                    st.column_config.NumberColumn(format="%d"),
                        "To":                      st.column_config.NumberColumn(format="%d"),
                    },
                )
                data_note("Authors are identified by last name and initials, because PubMed "
                          "carries no reliable author identifier. Namesakes therefore share a "
                          "row, and one person can appear twice when a publisher abbreviates "
                          "the name differently. The ORCID helps to tell the cases apart.")

                chosen_rows = selection.selection.rows if selection else []
                if chosen_rows:
                    picked      = found_authors.iloc[chosen_rows[0]]
                    picked_ids  = author_links.loc[author_links["author_key"] == picked["author_key"], "pmid"]
                    shown       = found_publications[found_publications["pmid"].isin(picked_ids)]
                    heading     = f"Publications of {picked['Author']}"
                else:
                    shown   = found_publications
                    heading = "Their Publications"

                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown(f'<div class="section-label">{heading}</div>', unsafe_allow_html=True)
                columns = [c for c in ["pmid", "article_title", "publication_year",
                                       "cited_by_count", "pubmed_url"]
                           if c in shown.columns]
                table = shown[columns].sort_values("publication_year", ascending=False)
                st.caption(f"{format_number(len(table))} publications")
                st.dataframe(
                    table, width="stretch", hide_index=True,
                    column_config=table_column_config(table),
                )
                st.download_button(
                    "Download the authors as CSV",
                    data=found_authors.drop(columns=["author_key"]).to_csv(index=False, sep=";").encode("utf-8-sig"),
                    file_name=f"{selected_dzg.lower()}_authors_by_topic.csv",
                    mime="text/csv",
                )


# Tab: Raw Data

with tab_data:
    table_choice = st.radio(
        "Table",
        ["Publications", "Authors", "MeSH Terms"],
        horizontal=True,
    )
    apply_filters = st.checkbox("Apply the sidebar filters", value=True)

    if table_choice == "Publications":
        df_table  = df_filtered if apply_filters else df_articles
        file_stem = "publications"
    elif table_choice == "Authors":
        df_table  = df_authors_filtered if apply_filters else df_authors
        file_stem = "authors"
    else:
        df_table  = df_mesh_filtered if apply_filters else df_mesh
        file_stem = "mesh_terms"

    if df_table.empty:
        st.info("No rows match the current selection.")
    else:
        df_view = filter_table(df_table, key_prefix=file_stem)

        st.caption(
            f"{format_number(len(df_view))} of {format_number(len(df_table))} rows, "
            f"{format_number(len(df_view.columns))} columns. "
            "Click a column header to sort."
        )

        if df_view.empty:
            st.info("No rows match the column filters.")
        else:
            # Rendering every row of a large table is slow and rarely useful, so the
            # preview is capped while the download still covers the full selection
            PREVIEW_ROWS = 1000
            st.dataframe(
                df_view.head(PREVIEW_ROWS),
                width="stretch",
                hide_index=True,
                column_config=table_column_config(df_view),
            )
            if len(df_view) > PREVIEW_ROWS:
                data_note(f"Showing the first {format_number(PREVIEW_ROWS)} rows. "
                          "The download contains all of them.")

            suffix = "filtered" if apply_filters else "full"
            st.download_button(
                "Download as CSV",
                data=df_view.to_csv(index=False, sep=";").encode("utf-8-sig"),
                file_name=f"{selected_dzg.lower()}_{file_stem}_{suffix}.csv",
                mime="text/csv",
            )