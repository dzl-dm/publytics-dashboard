"""
PubMed data extraction for the DZG selected via active_dzg in the yaml file.

Produces two CSV files:
  - pubmed_articles.csv : one row per article (primary key: PMID)
  - pubmed_authors.csv  : one row per author on an article (primary key: PMID + author_position, foreign key: PMID)
Also appends one row to metadata_extraction.csv per run, logging when the data was last extracted.

Requirements: pip install biopython pandas pyyaml
"""

import time
import yaml
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from Bio import Entrez

# Email for notifications in case of failures
Entrez.email = "your.email@example.com"  # Adjust!
# Entrez.api_key = "YOUR_API_KEY"  # Optional, for faster requests


FETCH_ALL       = True  # Set to True to fetch all results, otherwise capped at MAX_RESULTS per DZG (test run)
MAX_RESULTS     = 500
BATCH_SIZE      = 100   # for efetch (article retrieval)
YEAR_FROM       = 2009  # start year for splitting into annual slices
SLEEP_SECONDS   = 0.4   # delay between fetches


# Expected folder layout:   <project>/code/  contains the scripts and the YAML config
#                           <project>/data/  holds all CSV input and output files
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
YAML_PATH     = CODE_DIR / "dzg_search_terms.yaml"
CSV_ARTICLES  = DATA_DIR / "pubmed_articles.csv"
CSV_AUTHORS   = DATA_DIR / "pubmed_authors.csv"
CSV_METADATA  = DATA_DIR / "metadata_extraction.csv"


def load_yaml(path: Path) -> dict:
    """Load and return a YAML file as a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_search_term(config: dict, dzg: str) -> str:
    """Combine all affiliation terms for a DZG into a single query string."""
    terms = [t for nw in config[dzg].values() for t in nw]
    return " OR ".join(f'"{t}"[Affiliation]' for t in terms)


def get_attr(obj, key: str) -> str:
    """Read an XML attribute from a Biopython record object. Returns empty string if missing."""
    return str(getattr(obj, "attributes", {}).get(key, ""))


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


def _efetch(webenv: str, query_key: str, start: int, batch_size: int) -> list:
    """Fetch one batch of PubMed XML records from the server history session."""
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


def fetch_articles(webenv: str, query_key: str, target: int) -> list:
    """Download all articles for a search session in batches, with up to 3 retries per batch."""
    all_articles = []

    for start in range(0, target, BATCH_SIZE):
        page_size = min(BATCH_SIZE, target - start)

        for attempt in range(3):
            try:
                all_articles.extend(_efetch(webenv, query_key, start, page_size))
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"  Error batch {start}: {e} – retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"  Batch {start} skipped: {e}")

        time.sleep(SLEEP_SECONDS)

    return all_articles


def fetch_articles_by_year(search_term: str, max_results: int) -> tuple[list, int]:
    """Run annual sliced searches from YEAR_FROM to today and collect all matching articles."""
    current_year = date.today().year

    all_articles = []
    total_hits   = 0

    for year in range(YEAR_FROM, current_year + 1):
        webenv, query_key, hits_year = start_search(search_term, year)
        if hits_year == 0:
            continue
        time.sleep(SLEEP_SECONDS)

        total_hits += hits_year

        remaining = max_results - len(all_articles)
        if remaining <= 0:
            break
        target = min(hits_year, remaining)

        print(f"    {year}: {hits_year} hits, loading {target} ...")
        all_articles.extend(fetch_articles(webenv, query_key, target))

    return all_articles, total_hits


def parse_article(raw: dict) -> dict | None:
    """Extract article-level metadata from a raw PubMed XML record. Returns None if the record is malformed."""
    try:
        medline = raw["MedlineCitation"]
        art     = medline["Article"]
    except KeyError:
        return None

    pmid           = str(medline.get("PMID", ""))
    medline_status = get_attr(medline, "Status")

    journal  = art.get("Journal", {})
    issn     = str(journal.get("ISSN", ""))
    title    = str(journal.get("Title", ""))
    iso_abbr = str(journal.get("ISOAbbreviation", ""))

    article_title = str(art.get("ArticleTitle", ""))
    language      = "; ".join(str(s) for s in art.get("Language", []))

    # Find DOI in ArticleIdList
    doi = ""
    for id_obj in raw.get("PubmedData", {}).get("ArticleIdList", []):
        if get_attr(id_obj, "IdType") == "doi":
            doi = str(id_obj)
            break
    url        = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    # PublicationType: text and UI collected in parallel, stored separately
    pub_type_texts = []
    pub_type_uis   = []
    for pt in art.get("PublicationTypeList", []):
        pub_type_texts.append(str(pt))
        pub_type_uis.append(get_attr(pt, "UI"))

    # ArticleDate: individual fields plus a combined date string, only the first entry is used
    article_date = article_date_year = article_date_month = article_date_day = ""
    article_dates = art.get("ArticleDate", [])
    if article_dates:
        first = article_dates[0]
        article_date_year  = str(first.get("Year",  ""))
        article_date_month = str(first.get("Month", ""))
        article_date_day   = str(first.get("Day",   ""))
        article_date       = f"{article_date_year}-{article_date_month.zfill(2)}-{article_date_day.zfill(2)}"

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
        mesh_major_topics.append(get_attr(desc, "MajorTopicYN"))

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
        "article_date_year":      article_date_year,
        "article_date_month":     article_date_month,
        "article_date_day":       article_date_day,
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
        suffix     = str(author.get("Suffix",         ""))
        collective = str(author.get("CollectiveName", ""))

        # Author identifier
        author_identifiers        = []
        author_identifier_sources = []
        for id_obj in author.get("Identifier", []):
            author_identifiers.append(str(id_obj))
            author_identifier_sources.append(get_attr(id_obj, "Source"))

        # AffiliationInfo: one row per author, all affiliations combined
        affil_texts   = []
        affil_ids     = []
        affil_sources = []
        for aff in author.get("AffiliationInfo", []):
            affil_texts.append(str(aff.get("Affiliation", "")))
            for inst_id in aff.get("Identifier", []):
                affil_ids.append(str(inst_id))
                affil_sources.append(get_attr(inst_id, "Source"))

        rows.append({
            "pmid":                          pmid,
            "author_position":               position,
            "last_name":                     last_name,
            "fore_name":                     fore_name,
            "initials":                      initials,
            "suffix":                        suffix,
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
            print(f"ERROR: active_dzg '{active_dzg}' not found in YAML. Available: {list(raw_config)}")
            raise SystemExit(1)
        config = {active_dzg: raw_config[active_dzg]}
        print(f"Active DZG: {active_dzg} (set in dzg_search_terms.yaml)")
    else:
        config = raw_config
        print("No active_dzg set – extracting all DZGs")

    article_rows         = []
    author_rows          = []
    seen_pmids: set[str] = set()

    max_results = 10**9 if FETCH_ALL else MAX_RESULTS

    for dzg in config:
        search_term = build_search_term(config, dzg)
        print(f"\n{dzg}")

        articles, total = fetch_articles_by_year(search_term, max_results)

        print(f"  {total:,} hits in PubMed")
        print(f"  {len(articles)} articles loaded")

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
        print(f"  Intermediate state saved ({len(article_rows)} articles total)")

    print(f"\nArticles: {len(article_rows)} rows  ->  {CSV_ARTICLES.name}")
    print(f"Authors:  {len(author_rows)} rows  ->  {CSV_AUTHORS.name}")

    runtime = time.time() - t_start
    write_metadata(CSV_METADATA, len(config), len(article_rows), len(author_rows), runtime)
    print(f"Metadata: {CSV_METADATA.name}")