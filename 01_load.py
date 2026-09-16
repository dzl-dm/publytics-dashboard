"""
PubMed data extraction for the DZG selected via active_dzg in the yaml file.

Produces two CSV files:
  - pubmed_articles.csv : one row per article (primary key: PMID)
  - pubmed_authors.csv  : one row per author on an article (primary key: PMID + author_position, foreign key: PMID)
Also appends one row to metadata_extraction.csv per run, logging when the data was last extracted.

Requirements: see requirements.txt
"""

import logging
import time
import yaml
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from Bio import Entrez

from search_terms import build_query


# All progress output goes through the logging module so the verbosity can be changed
# in one place and messages carry a timestamp and severity level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(Path(__file__).stem)

# Email for notifications in case of failures
Entrez.email = "your.email@example.com"  # Adjust!
# Entrez.api_key = "YOUR_API_KEY"  # Optional, for faster requests


FETCH_ALL       = True  # Set to True to fetch all results, otherwise capped at MAX_RESULTS per DZG (test run)
MAX_RESULTS     = 500
BATCH_SIZE      = 100   # articles per efetch call
RETRIES         = 3     # attempts per batch before it is given up on
YEAR_FROM       = 2009  # start year for splitting into annual slices
SLEEP_SECONDS   = 0.4   # delay between fetches


# Paths resolve relative to this script, so the project runs from any location.
# The scripts and the YAML config sit in the repository root, the data subfolder
# holds every CSV and is created on first run.
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
YAML_PATH     = CODE_DIR / "dzg_search_terms.yaml"
CSV_ARTICLES  = DATA_DIR / "pubmed_articles.csv"
CSV_AUTHORS   = DATA_DIR / "pubmed_authors.csv"
CSV_METADATA  = DATA_DIR / "metadata_extraction.csv"


def load_yaml(path: Path) -> dict:
    """Load and return a YAML file as a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def xml_attribute(element, name: str) -> str:
    """Read an attribute off a Biopython XML element, or an empty string when it is absent."""
    return str(getattr(element, "attributes", {}).get(name, ""))


def write_metadata(path: Path, n_dzg: int, n_articles: int, n_authors: int, runtime: float) -> None:
    """Append one summary row per run to the metadata CSV so past extractions stay traceable."""
    row = pd.DataFrame([{
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_dzg":           n_dzg,
        "n_articles":      n_articles,
        "n_authors":       n_authors,
        "runtime_seconds": round(runtime, 1),
    }])
    file_exists = path.exists()
    row.to_csv(path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig", sep=";")


def start_search(search_term: str, year: int) -> tuple[str, str, int]:
    """Run an Entrez esearch for a given term and year. Returns WebEnv, QueryKey, and total hit count."""
    term   = f"{search_term} AND {year}[pdat]"
    handle = Entrez.esearch(db="pubmed", term=term, retmax=0, usehistory="y")
    try:
        record = Entrez.read(handle)
    finally:
        handle.close()
    return record["WebEnv"], record["QueryKey"], int(record["Count"])


def fetch_one_batch(webenv: str, query_key: str, start: int, batch_size: int) -> list:
    """Fetch a single batch of PubMed records, one network call.

    The lowest of three levels: fetch_one_batch gets one block, fetch_all_batches gets
    every block of one search, and fetch_articles_by_year runs one search per year.
    """
    handle = Entrez.efetch(
        db="pubmed",
        query_key=query_key,
        WebEnv=webenv,
        rettype="xml",
        retmode="xml",
        retstart=start,
        retmax=batch_size
    )
    try:
        records = Entrez.read(handle)
    finally:
        handle.close()
    return records.get("PubmedArticle", [])


def fetch_all_batches(webenv: str, query_key: str, target: int) -> list:
    """Fetch every article of one search, block by block.
    """
    all_articles = []

    for start in range(0, target, BATCH_SIZE):
        page_size = min(BATCH_SIZE, target - start)

        for attempt in range(1, RETRIES + 1):
            try:
                all_articles.extend(fetch_one_batch(webenv, query_key, start, page_size))
                break
            except Exception as error:
                if attempt == RETRIES:
                    log.error(f"Batch at {start} given up on after {RETRIES} attempts: {error}")
                else:
                    wait = 2 ** attempt
                    log.warning(f"Batch at {start} failed ({error}), retrying in {wait}s")
                    time.sleep(wait)

        time.sleep(SLEEP_SECONDS)

    return all_articles


def fetch_articles_by_year(search_term: str, max_results: int | None) -> tuple[list, int]:
    """Search year by year from YEAR_FROM to today and collect the matching articles.
    """
    all_articles = []
    total_hits   = 0

    for year in range(YEAR_FROM, date.today().year + 1):
        webenv, query_key, hits = start_search(search_term, year)
        if hits == 0:
            continue
        time.sleep(SLEEP_SECONDS)
        total_hits += hits

        wanted = hits
        if max_results is not None:
            remaining = max_results - len(all_articles)
            if remaining <= 0:
                break
            wanted = min(hits, remaining)

        log.info(f"{year}: {hits:,} hits, loading {wanted:,}")
        all_articles.extend(fetch_all_batches(webenv, query_key, wanted))

    return all_articles, total_hits


def parse_article(raw: dict) -> dict | None:
    """Extract article-level metadata from a raw PubMed XML record. Returns None if the record is malformed."""
    try:
        medline = raw["MedlineCitation"]
        art     = medline["Article"]
    except KeyError:
        return None

    pmid           = str(medline.get("PMID", ""))
    medline_status = xml_attribute(medline, "Status")

    journal  = art.get("Journal", {})
    issn     = str(journal.get("ISSN", ""))
    title    = str(journal.get("Title", ""))
    iso_abbr = str(journal.get("ISOAbbreviation", ""))

    article_title = str(art.get("ArticleTitle", ""))
    language      = "; ".join(str(s) for s in art.get("Language", []))

    # Find DOI in ArticleIdList
    doi = ""
    for id_obj in raw.get("PubmedData", {}).get("ArticleIdList", []):
        if xml_attribute(id_obj, "IdType") == "doi":
            doi = str(id_obj)
            break
    url        = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    # PublicationType: text and UI collected in parallel, stored separately
    pub_type_texts = []
    pub_type_uis   = []
    for pt in art.get("PublicationTypeList", []):
        pub_type_texts.append(str(pt))
        pub_type_uis.append(xml_attribute(pt, "UI"))

    # Electronic publication date. Only the first entry is used, further ones repeat the
    # same publication in another format.
    article_date      = ""
    article_date_year = ""
    article_dates     = art.get("ArticleDate", [])
    if article_dates:
        first             = article_dates[0]
        article_date_year = str(first.get("Year", ""))
        month             = str(first.get("Month", "")).zfill(2)
        day               = str(first.get("Day", "")).zfill(2)
        article_date      = f"{article_date_year}-{month}-{day}"

    # PubDate as a fallback for the year (ArticleDate isn't always present)
    pubdate_year = ""
    pubdate_raw  = journal.get("JournalIssue", {}).get("PubDate", {})
    if pubdate_raw:
        pubdate_year = str(pubdate_raw.get("Year", ""))
        if not pubdate_year:
            medline_date = str(pubdate_raw.get("MedlineDate", ""))
            pubdate_year = medline_date[:4] if medline_date else ""

    publication_year = article_date_year or pubdate_year

    # MeSH: DescriptorName and MajorTopicYN collected in parallel, stored separately
    mesh_descriptors  = []
    mesh_major_topics = []
    for m in medline.get("MeshHeadingList", []):
        desc = m["DescriptorName"]
        mesh_descriptors.append(str(desc))
        mesh_major_topics.append(xml_attribute(desc, "MajorTopicYN"))

    n_authors = len(art.get("AuthorList", []))

    return {
        "pmid":                   pmid,
        "medline_status":         medline_status,
        "doi":                    doi,
        "issn":                   issn,
        "journal_title":          title,
        "journal_iso_abbr":       iso_abbr,
        "article_title":          article_title,
        "language":               language,
        "publication_type":       "; ".join(pub_type_texts),
        "publication_type_ui":    "; ".join(pub_type_uis),
        "article_date":           article_date,
        "publication_year":       publication_year,
        "mesh_descriptor":        "; ".join(mesh_descriptors),
        "mesh_major_topic_yn":    "; ".join(mesh_major_topics),
        "n_authors":              n_authors,
        "url":                    url,
        "pubmed_url":             pubmed_url,
    }


def parse_authors(raw: dict) -> list[dict]:
    """Extract one row per author from a raw PubMed XML record, including affiliation and identifier fields."""
    try:
        medline = raw["MedlineCitation"]
        art     = medline["Article"]
    except KeyError:
        return []

    pmid = str(medline.get("PMID", ""))
    rows = []

    for position, author in enumerate(art.get("AuthorList", []), start=1):
        last_name  = str(author.get("LastName",       ""))
        fore_name  = str(author.get("ForeName",       ""))
        initials   = str(author.get("Initials",       ""))
        collective = str(author.get("CollectiveName", ""))

        # Author identifier
        author_identifiers        = []
        author_identifier_sources = []
        for id_obj in author.get("Identifier", []):
            author_identifiers.append(str(id_obj))
            author_identifier_sources.append(xml_attribute(id_obj, "Source"))

        # AffiliationInfo: one row per author, all affiliations combined
        affil_texts   = []
        affil_ids     = []
        affil_sources = []
        for aff in author.get("AffiliationInfo", []):
            affil_texts.append(str(aff.get("Affiliation", "")))
            for inst_id in aff.get("Identifier", []):
                affil_ids.append(str(inst_id))
                affil_sources.append(xml_attribute(inst_id, "Source"))

        rows.append({
            "pmid":                          pmid,
            "author_position":               position,
            "last_name":                     last_name,
            "fore_name":                     fore_name,
            "initials":                      initials,
            "collective_name":               collective,
            "author_identifier":             " | ".join(author_identifiers),
            "author_identifier_source":      " | ".join(author_identifier_sources),
            "affiliation":                   " | ".join(affil_texts),
            "affiliation_identifier":        " | ".join(affil_ids),
            "affiliation_identifier_source": " | ".join(affil_sources),
        })

    return rows


if __name__ == "__main__":
    t_start = time.time()

    raw_config = load_yaml(YAML_PATH)

    # Respect active_dzg setting – only extract data for the configured DZG
    raw_config.pop("colors", None)
    active_dzg = raw_config.pop("active_dzg", None)
    if active_dzg:
        if active_dzg not in raw_config:
            log.error("active_dzg '%s' not found in YAML, available: %s", active_dzg, list(raw_config))
            raise SystemExit(1)
        config = {active_dzg: raw_config[active_dzg]}
        log.info("Active DZG: %s (set in dzg_search_terms.yaml)", active_dzg)
    else:
        config = raw_config
        log.info("No active_dzg set, extracting all DZGs")

    article_rows         = []
    author_rows          = []
    seen_pmids: set[str] = set()

    max_results = None if FETCH_ALL else MAX_RESULTS

    for dzg in config:
        search_term = build_query(config[dzg])
        log.info("Extracting %s", dzg)

        articles, total = fetch_articles_by_year(search_term, max_results)

        log.info("%s hits in PubMed", f"{total:,}")
        log.info("%s articles loaded", f"{len(articles):,}")

        for raw in articles:
            parsed = parse_article(raw)
            if not parsed:
                continue
            if parsed["pmid"] in seen_pmids:
                continue
            seen_pmids.add(parsed["pmid"])

            article_rows.append(parsed)
            author_rows.extend(parse_authors(raw))

        # Save intermediate state after each DZG
        pd.DataFrame(article_rows).to_csv(CSV_ARTICLES, index=False, encoding="utf-8-sig", sep=";")
        pd.DataFrame(author_rows).to_csv(CSV_AUTHORS,   index=False, encoding="utf-8-sig", sep=";")
        log.info("Intermediate state saved, %s articles total", f"{len(article_rows):,}")

    log.info("Articles: %s rows -> %s", f"{len(article_rows):,}", CSV_ARTICLES.name)
    log.info("Authors:  %s rows -> %s", f"{len(author_rows):,}", CSV_AUTHORS.name)

    runtime = time.time() - t_start
    write_metadata(CSV_METADATA, len(config), len(article_rows), len(author_rows), runtime)
    log.info("Metadata written to %s", CSV_METADATA.name)