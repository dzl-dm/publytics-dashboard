"""
Data preparation: transforms raw PubMed CSVs into analysis-ready output.

Produces three CSV files:
  - pubmed_articles_processed.csv : one row per article, enriched with DZG affiliation, citations and SJR
  - pubmed_authors_processed.csv  : one row per author, enriched with DZG affiliation and first/last author flags
  - pubmed_mesh.csv               : one row per article × MeSH term (exploded from mesh_descriptor)
Also appends one row to metadata_processing.csv per run, logging citation coverage and runtime.

Requirements: see requirements.txt
"""

import logging
import time
import re
import asyncio
import aiohttp
import requests
import yaml
import pandas as pd
from pathlib import Path
from io import StringIO
from datetime import datetime

from search_terms import affiliation_matches, dzg_terms


# All progress output goes through the logging module so the verbosity can be changed
# in one place and messages carry a timestamp and severity level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(Path(__file__).stem)


# Paths resolve relative to this script, so the project runs from any location.
# The scripts and the YAML config sit in the repository root, the data subfolder
# holds every CSV and is created on first run.
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
YAML_PATH           = CODE_DIR / "dzg_search_terms.yaml"
MESH_STOPLIST_PATH  = CODE_DIR / "mesh_stoplist.yaml"
CSV_ARTICLES        = DATA_DIR / "pubmed_articles.csv"
CSV_AUTHORS         = DATA_DIR / "pubmed_authors.csv"

CSV_OUTPUT_ARTICLES = DATA_DIR / "pubmed_articles_processed.csv"
CSV_OUTPUT_AUTHORS  = DATA_DIR / "pubmed_authors_processed.csv"
CSV_MESH            = DATA_DIR / "pubmed_mesh.csv"
CSV_METADATA        = DATA_DIR / "metadata_processing.csv"
SJR_CACHE_FOLDER    = DATA_DIR / "sjr_data"
SJR_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

ICITE_BATCH_SIZE       = 200   # PMIDs per request, the maximum iCite accepts
REQUEST_CONCURRENCY    = 10    # requests running at the same time
ICITE_TIMEOUT          = 30    # seconds before a request is abandoned
RETRIES                = 3     # attempts per batch before it is given up on
HTTP_OK                = 200   # the request succeeded
HTTP_TOO_MANY_REQUESTS = 429   # rate limit reached, waiting and retrying helps


def load_yaml(path: Path) -> dict:
    """Load and return a YAML file as a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dzg_affiliation_articles(df_authors: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Determine DZG affiliation at article level by matching keywords against all author affiliations combined."""
    affil_per_pmid = (
        df_authors.groupby("pmid")["affiliation"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
        .rename(columns={"affiliation": "affil_combined"})
    )
    result = affil_per_pmid[["pmid"]].copy()
    for dzg, networks in config.items():
        for network, terms in networks.items():
            column = dzg if network == "general" else f"{dzg}_{network}"
            result[column] = affiliation_matches(affil_per_pmid["affil_combined"], terms)
    return result


def dzg_affiliation_authors(df_authors: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Determine DZG affiliation per author by matching the terms against their affiliation.

    Only terms that identify the DZG itself count here. A {nosearch} is excluded.
    """
    result       = df_authors[["pmid", "author_position"]].copy()
    affiliations = df_authors["affiliation"].fillna("")
    for dzg, networks in config.items():
        result[dzg] = affiliation_matches(affiliations, dzg_terms(networks))
    return result


def mark_first_last_author(df_authors: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns indicating whether each author is the first or last author of their article."""
    df = df_authors.copy()
    df["author_position"] = df["author_position"].astype(int)
    last_position = df.groupby("pmid")["author_position"].transform("max")
    df["is_first_author"] = df["author_position"] == 1
    df["is_last_author"]  = df["author_position"] == last_position
    return df


def count_dzg_authors(df_authors: pd.DataFrame, dzg_columns: list) -> pd.DataFrame:
    """Count how many authors per article are affiliated with any DZG."""
    is_dzg_author = df_authors[dzg_columns].any(axis=1)
    return (
        df_authors.assign(is_dzg_author=is_dzg_author)
        .groupby("pmid")["is_dzg_author"]
        .sum()
        .reset_index(name="n_dzg_authors")
    )


def strip_html_tags(series: pd.Series) -> pd.Series:
    """Remove HTML formatting tags from a text column; PubMed titles sometimes contain tags like <i>."""
    return series.astype(str).str.replace(r"<[^>]+>", "", regex=True)


async def icite_batch(session: aiohttp.ClientSession,
                      semaphore: asyncio.Semaphore,
                      pmid_list: list) -> dict:
    """Fetch the citation data for one batch of PMIDs from iCite, keyed by PMID.

    Returns an empty dict when the batch cannot be retrieved.
    """
    url    = "https://icite.od.nih.gov/api/pubs"
    params = {"pmids": ",".join(pmid_list)}

    async with semaphore:
        for attempt in range(1, RETRIES + 1):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == HTTP_OK:
                        data = await response.json()
                        return {str(record["pmid"]): record for record in data.get("data", [])}

                    if response.status == HTTP_TOO_MANY_REQUESTS:
                        # Sending too fast, wait longer with every attempt and try again
                        await asyncio.sleep(2 ** attempt)
                        continue

                    # Any other status means iCite will not serve this batch at all,
                    # so repeating the request would not help
                    log.warning(f"iCite answered with status {response.status}, batch skipped")
                    return {}

            # A timeout is not a ClientError, without it here one slow request would
            # abort the whole run instead of costing a single batch
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                if attempt == RETRIES:
                    log.warning(f"iCite batch given up on after {RETRIES} attempts: {error}")
                else:
                    await asyncio.sleep(2 ** attempt)

    # Reached when every attempt ran into the rate limit. Without this line the batch
    # would disappear silently and the missing citations would be hard to notice.
    log.warning(f"iCite kept rate limiting a batch of {len(pmid_list)} PMIDs, giving up on it")
    return {}


async def icite_all(pmids: list) -> dict:
    """Dispatch all iCite batch requests concurrently and merge results into a single PMID lookup dict."""
    semaphore = asyncio.Semaphore(REQUEST_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=REQUEST_CONCURRENCY, ssl=False)
    timeout   = aiohttp.ClientTimeout(total=ICITE_TIMEOUT)
    lookup = {}
    total = len(pmids)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i in range(0, total, ICITE_BATCH_SIZE):
            batch_list = pmids[i:i + ICITE_BATCH_SIZE]
            tasks.append(icite_batch(session, semaphore, batch_list))

        for i in range(0, len(tasks), 100):
            group   = tasks[i:i + 100]
            results = await asyncio.gather(*group)
            for result in results:
                lookup.update(result)
            done = min((i + 100) * ICITE_BATCH_SIZE, total)
            log.info("iCite: %s/%s PMIDs processed", f"{done:,}", f"{total:,}")
    return lookup


def enrich_citations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add cited_by_count and RCR columns to the article DataFrame by querying the iCite API."""
    df    = df.copy()
    pmids = df["pmid"].dropna().astype(str).tolist()
    log.info("%s PMIDs, batch size %s, %s concurrent requests",
             f"{len(pmids):,}", ICITE_BATCH_SIZE, REQUEST_CONCURRENCY)

    lookup = asyncio.run(icite_all(pmids))

    pmid_str = df["pmid"].astype(str)
    df["cited_by_count"] = pmid_str.map({k: v.get("citation_count") for k, v in lookup.items()})
    df["rcr"]            = pmid_str.map({k: v.get("relative_citation_ratio") for k, v in lookup.items()})

    found = len(lookup.keys() & set(pmids))
    share = found / max(1, len(pmids)) * 100
    if share < 90:
        log.warning(f"iCite only returned data for {share:.1f} percent of the PMIDs. "
                    f"Citation figures are incomplete, consider running this step again.")

    stats = {
        "source":  "iCite",
        "queried": len(pmids),
        "found":   found,
        "share":   f"{found / max(1, len(df)) * 100:.1f}%",
    }
    return df, stats


def normalize_issn(issn) -> str | None:
    """Strip non-alphanumeric characters from an ISSN and return it uppercased, or None if empty."""
    if pd.isna(issn):
        return None
    cleaned = re.sub(r"[^0-9Xx]", "", str(issn)).upper()
    return cleaned or None


def load_sjr_year(year: int) -> pd.DataFrame:
    """Download or load from cache the SCImago SJR data for a given year; returns one row per ISSN."""
    # One row per (issn, year); a journal with print+online ISSN is exploded into two rows
    cache_path = SJR_CACHE_FOLDER / f"sjr_{year}.csv"

    if cache_path.exists():
        raw = pd.read_csv(cache_path, sep=";", dtype=str, encoding="utf-8-sig")
    else:
        # scimagojr.com returns 403 without a browser-like header, plain urllib/requests
        # user agents get blocked
        url = f"https://www.scimagojr.com/journalrank.php?year={year}&out=xls"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "text/csv,application/vnd.ms-excel,*/*",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            "Referer": "https://www.scimagojr.com/journalrank.php",
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text), sep=";", decimal=",", dtype=str)
        raw.to_csv(cache_path, index=False, sep=";", encoding="utf-8-sig")
        time.sleep(1.5)

    raw = raw.rename(columns={
        "Issn": "issn_raw",
        "SJR":  "sjr",
        "SJR Best Quartile": "sjr_quartile",
    })
    raw = raw[["issn_raw", "sjr", "sjr_quartile"]].copy()
    raw["sjr"]  = pd.to_numeric(raw["sjr"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    raw["year"] = year

    # one journal can list several ISSNs separated by comma, one row per ISSN after the split
    raw["issn_list"] = raw["issn_raw"].astype(str).str.split(",")
    sjr_long = raw.explode("issn_list")
    sjr_long["issn"] = sjr_long["issn_list"].apply(normalize_issn)
    sjr_long = sjr_long.dropna(subset=["issn"])
    sjr_long = sjr_long.drop_duplicates(subset=["issn", "year"], keep="first")

    return sjr_long[["issn", "year", "sjr", "sjr_quartile"]]


def load_sjr_data(years: set) -> pd.DataFrame:
    """Load SJR data for all publication years present in the dataset and concatenate into one DataFrame."""
    parts = []
    for year in sorted(years):
        log.info("Loading SJR data for %s", year)
        try:
            parts.append(load_sjr_year(int(year)))
        except Exception as error:
            log.warning("Could not load SJR data for %s: %s", year, error)
    if not parts:
        empty = pd.DataFrame({
            "issn": pd.Series(dtype="object"),
            "year": pd.Series(dtype="int64"),
            "sjr":  pd.Series(dtype="float64"),
            "sjr_quartile": pd.Series(dtype="object"),
        })
        return empty
    return pd.concat(parts, ignore_index=True)


def match_sjr(df_articles: pd.DataFrame, sjr_data: pd.DataFrame) -> pd.DataFrame:
    """Join SJR values onto articles by ISSN, using the closest preceding year when an exact match is missing."""
    df = df_articles.copy()

    if sjr_data.empty:
        log.warning("No SJR data available, sjr columns remain empty")
        df["sjr"]           = pd.NA
        df["sjr_quartile"]  = pd.NA
        return df

    df["issn_norm"]  = df["issn"].apply(normalize_issn)
    df["_year_sort"] = pd.to_numeric(df["publication_year"], errors="coerce")

    sjr_sorted = sjr_data.rename(columns={"issn": "issn_norm"}).sort_values("year")
    df_sorted  = df.sort_values("_year_sort")

    matched = pd.merge_asof(
        df_sorted,
        sjr_sorted,
        left_on="_year_sort",
        right_on="year",
        by="issn_norm",
        direction="backward",
    )
    matched = matched.drop(columns=["issn_norm", "_year_sort", "year"])
    return matched.sort_index()


def load_mesh_stoplist(path: Path) -> set[str]:
    """Load MeSH stoplist from YAML and return all terms as a flat set; returns empty set if file is missing."""
    if not path.exists():
        log.warning("mesh_stoplist.yaml not found at %s, no terms will be filtered", path)
        return set()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    terms = set()
    for category_terms in data.values():
        if isinstance(category_terms, list):
            terms.update(category_terms)
    return terms


def build_mesh_long(df_articles: pd.DataFrame) -> pd.DataFrame:
    """Explode the semicolon-separated mesh_descriptor column into one row per article × MeSH term."""
    if "mesh_descriptor" not in df_articles.columns:
        return pd.DataFrame(columns=["pmid", "publication_year", "mesh_term"])
    mesh = (
        df_articles[["pmid", "publication_year", "mesh_descriptor"]]
        .dropna(subset=["mesh_descriptor"])
        .copy()
    )
    mesh["mesh_term"] = mesh["mesh_descriptor"].str.split("; ")
    mesh = mesh.explode("mesh_term")
    mesh["mesh_term"] = mesh["mesh_term"].str.strip()
    mesh = mesh[mesh["mesh_term"] != ""].drop(columns=["mesh_descriptor"])
    return mesh.reset_index(drop=True)


def write_metadata(path: Path, stats: dict, runtime: float, df_articles: pd.DataFrame) -> None:
    """Append one summary row per run to the metadata CSV so past pipeline runs stay traceable."""
    total_citations = int(df_articles["cited_by_count"].sum())         if "cited_by_count" in df_articles.columns else None
    avg_rcr         = round(df_articles["rcr"].mean(), 3)               if "rcr"            in df_articles.columns else None
    avg_sjr         = round(df_articles["sjr"].mean(), 3)               if "sjr"            in df_articles.columns else None
    n_q1            = int((df_articles["sjr_quartile"] == "Q1").sum())  if "sjr_quartile"   in df_articles.columns else None

    row = pd.DataFrame([{
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":           stats["source"],
        "n_queried":        stats["queried"],
        "n_found":          stats["found"],
        "share_found":      stats["share"],
        "total_citations":  total_citations,
        "avg_rcr":          avg_rcr,
        "avg_sjr":          avg_sjr,
        "n_q1_articles":    n_q1,
        "runtime_seconds":  round(runtime, 1),
    }])
    file_exists = path.exists()
    row.to_csv(path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig", sep=";")


if __name__ == "__main__":
    t_start = time.time()
    config = load_yaml(YAML_PATH)
    # active_dzg only controls extraction; all DZG columns are computed here
    # so the dashboard can show co-affiliation with other DZGs
    config.pop("active_dzg", None)
    config.pop("colors", None)
    dzg_columns = list(config.keys())

    log.info("Loading input files")
    df_articles = pd.read_csv(CSV_ARTICLES, sep=";", dtype=str, encoding="utf-8-sig")
    df_authors  = pd.read_csv(CSV_AUTHORS,  sep=";", dtype=str, encoding="utf-8-sig")
    log.info("%s articles, %s author rows", f"{len(df_articles):,}", f"{len(df_authors):,}")

    if "article_title" in df_articles.columns:
        df_articles["article_title"] = strip_html_tags(df_articles["article_title"])

    log.info("Step 1: DZG affiliation at article level")
    df_dzg_articles = dzg_affiliation_articles(df_authors, config)
    df_articles = df_articles.merge(df_dzg_articles, on="pmid", how="left")
    log.info("%s columns added", len(df_dzg_articles.columns) - 1)

    log.info("Step 2: DZG affiliation at author level")
    df_dzg_authors = dzg_affiliation_authors(df_authors, config)
    df_authors = df_authors.merge(df_dzg_authors, on=["pmid", "author_position"], how="left")

    log.info("Step 3: Number of DZG authors per article")
    dzg_count   = count_dzg_authors(df_authors, dzg_columns)
    df_articles = df_articles.merge(dzg_count, on="pmid", how="left")
    df_articles["n_dzg_authors"] = df_articles["n_dzg_authors"].fillna(0).astype(int)

    log.info("Step 4: Marking first and last authors")
    df_authors = mark_first_last_author(df_authors)

    log.info("Step 5: Citation data from iCite")
    df_articles, stats = enrich_citations(df_articles)

    log.info("Step 6: Matching SJR journal metric by ISSN")
    years_in_data = set(pd.to_numeric(df_articles["publication_year"], errors="coerce").dropna().astype(int))
    sjr_data    = load_sjr_data(years_in_data)
    df_articles = match_sjr(df_articles, sjr_data)
    sjr_found   = df_articles["sjr"].notna().sum()
    log.info("SJR found for %s/%s articles", f"{sjr_found:,}", f"{len(df_articles):,}")


    log.info("Step 7: Building MeSH long table")
    stoplist   = load_mesh_stoplist(MESH_STOPLIST_PATH)
    df_mesh    = build_mesh_long(df_articles)
    df_mesh["is_stopword"] = df_mesh["mesh_term"].isin(stoplist)
    n_stop = df_mesh["is_stopword"].sum()
    log.info("%s article by MeSH term rows, %s marked as stopwords",
             f"{len(df_mesh):,}", f"{n_stop:,}")

    try:
        df_articles.to_csv(CSV_OUTPUT_ARTICLES, index=False, encoding="utf-8-sig", sep=";")
        df_authors.to_csv(CSV_OUTPUT_AUTHORS,   index=False, encoding="utf-8-sig", sep=";")
        df_mesh.to_csv(CSV_MESH,                index=False, encoding="utf-8-sig", sep=";")
    except PermissionError:
        log.error("Output file is open, please close it and restart the script")
        raise SystemExit(1)

    runtime = time.time() - t_start
    write_metadata(CSV_METADATA, stats, runtime, df_articles)

    log.info("Citation source: %s", stats["source"])
    log.info("Queried:         %s PMIDs", f"{stats['queried']:,}")
    log.info("With citations:  %s (%s of all articles)", f"{stats['found']:,}", stats["share"])
    log.info("Runtime:         %.0fs (%.1f minutes)", runtime, runtime / 60)
    log.info("Saved: %s, %s, %s", CSV_OUTPUT_ARTICLES.name, CSV_OUTPUT_AUTHORS.name, CSV_MESH.name)
    log.info("Metadata written to %s", CSV_METADATA.name)