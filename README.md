# DZG Publication Analysis Dashboard Based on PubMed Data

Extracts the publication record of a German Center for Health Research (DZG) from PubMed, enriches it with citation and journal metrics, and visualises it in an interactive Streamlit dashboard.

The pipeline covers one DZG at a time. Which one is set in a YAML config.

## Table of Contents

| Section | Content |
|---|---|
| [Quick Start](#quick-start) | Installation and the three commands to run |
| [How It Works](#how-it-works) | Pipeline overview and folder structure |
| [Configuration](#configuration) | Search terms, colours, stoplist and script settings |
| [Output Files](#output-files) | Column reference for every generated file |
| [Known Limitations](#known-limitations) | What the data cannot tell you |
| [Further Resources](#further-resources) | External documentation |

## Quick Start

Python 3.10 or higher is required.

```bash
pip install biopython pandas pyyaml requests aiohttp streamlit plotly matplotlib numpy venn

python code/01_load.py
python code/02_preprocessing.py
streamlit run code/03_dashboard.py
```

Set your own email address in `code/01_load.py` before the first run. NCBI requires it for API access. No registration or key is needed.

## How It Works

Python is used as the programming language.
The pipeline consists of three scripts, run in order.

| Script | Task |
|---|---|
| `01_load.py` | Queries PubMed by affiliation and writes raw article and author tables |
| `02_preprocessing.py` | Adds DZG affiliation flags, citation counts from NIH iCite, journal prestige from SCImago, and a MeSH term table |
| `03_dashboard.py` | Renders the dashboard with five tabs covering publications, citations, journal metrics, collaboration and MeSH topics |

Paths resolve relative to the script location, so the project runs from any folder. The `data` folder is created automatically on first run.

```
project/
├── code/
│   ├── 01_load.py
│   ├── 02_preprocessing.py
│   ├── 03_dashboard.py
│   ├── dzg_search_terms.yaml
│   └── mesh_stoplist.yaml
└── data/
    ├── pubmed_articles.csv
    ├── pubmed_authors.csv
    ├── pubmed_articles_processed.csv
    ├── pubmed_authors_processed.csv
    ├── pubmed_mesh.csv
    ├── metadata_extraction.csv
    ├── metadata_processing.csv
    └── sjr_cache/
```

## Configuration

### Search Terms

`dzg_search_terms.yaml` holds everything that varies between centers. The `active_dzg` key decides which DZG is extracted and displayed. Each DZG carries a `general` block plus optional site blocks, and every site block becomes a `<DZG>_<site>` column in the processed files and a site filter in the dashboard.

```yaml
active_dzg: DZL

DZL:
  general:
    - "German Center for Lung Research"
    - "DZL"
  TLRC:
    - "Translational Lung Research Center"
    - "TLRC"
```

### Colours

The `colors` block in the same file assigns one colour per DZG and per site, so a site keeps the same colour in every chart it appears in. Anything left out falls back to a built in palette. Values are hex color codes in the form #RRGGBB. This [colour chart](https://www.computerhope.com/htmcolor.htm) lists common colours with their codes.

```yaml
colors:
  dzg:
    DZL:  "#5dade2"
    DZIF: "#3ecfaa"
  sites:
    DZL:
      TLRC:   "#4f8ef7"
      BREATH: "#f5a623"
```

### MeSH Stoplist

`mesh_stoplist.yaml` lists terms that carry no thematic information, such as `Humans` or `Female`. They are hidden by default in the dashboard and can be shown again with a checkbox.

### Script Settings

| Setting | Script | Purpose | Default |
|---|---|---|---|
| `Entrez.email` | `01_load.py` | Your email address, required by NCBI | none |
| `Entrez.api_key` | `01_load.py` | Optional key that raises the rate limit | disabled |
| `FETCH_ALL` | `01_load.py` | `False` caps the run at `MAX_RESULTS` for testing | `True` |
| `YEAR_FROM` | `01_load.py` | First publication year searched | `2009` |
| `ICITE_BATCH_SIZE` | `02_preprocessing.py` | PMIDs per iCite request, maximum 200 | `200` |
| `REQUEST_CONCURRENCY` | `02_preprocessing.py` | Parallel iCite requests | `10` |

## Output Files

### Article Tables

One row per article, keyed by `pmid`. The raw file carries the PubMed fields such as `doi`, `issn`, `journal_title`, `article_title`, `publication_year`, `mesh_descriptor` and `n_authors`. Preprocessing adds the following.

| Column | Content |
|---|---|
| `DZIF`, `DZNE`, ... | DZG affiliation as `True` or `False` |
| `DZL_TLRC`, `DZL_BREATH`, ... | Site affiliation, present only where sites are defined |
| `n_dzg_authors` | Number of authors on the article affiliated with any DZG |
| `cited_by_count` | Total citations reported by NIH iCite |
| `rcr` | Relative Citation Ratio, where 1.0 equals the NIH field average |
| `sjr` | SCImago Journal Rank of the publishing journal, matched by ISSN |
| `sjr_quartile` | Journal quartile from Q1 to Q4 |

Although only one DZG is extracted, preprocessing evaluates all DZG columns. This is what lets the dashboard show how often the active center publishes together with the other centers.

### Author Tables

One row per author, referencing `pmid` and ordered by `author_position`. Beyond the name and affiliation fields, preprocessing adds the DZG affiliation columns plus `is_first_author` and `is_last_author`. Affiliation always refers to the specific article, so an author moving between centers does not distort earlier publications.

### MeSH Table

`pubmed_mesh.csv` holds one row per article and MeSH term, with `is_stopword` marking terms from the stoplist.

### Metadata

Both pipeline steps append one row per run. `metadata_extraction.csv` records article and author counts, `metadata_processing.csv` records citation coverage together with `total_citations`, `avg_rcr`, `avg_sjr` and `n_q1_articles`, which makes the effect of each update traceable. The dashboard shows both timestamps in the sidebar.

## Known Limitations

**Affiliations are free text.** PubMed stores the affiliation as the publisher submitted it, without a controlled vocabulary. The same institute therefore appears under many spellings, translations and abbreviations, and typographical errors are passed through unchanged. Matching relies on a hand maintained term list, so some publications are missed. At the same time PubMed searches case insensitively, so a short term like `BREATH` also matches unrelated contexts. Preprocessing applies a word boundary and context check for that case, but full precision is not achievable.

**SJR data lags behind.** SCImago publishes its rankings with roughly a year of delay. Where no entry exists for the publication year, the nearest earlier year for the same ISSN is used, so a quartile may reflect an older ranking.

**MeSH indexing is incomplete for recent articles.** Publications from the current year are often not yet indexed, so empty MeSH fields are expected there.

**Citation counts are a snapshot.** iCite updates regularly, so figures are accurate at query time and drift afterwards.

**SCImago can block automated requests.** The script sends a browser like `User-Agent` header. Should this stop working, the yearly rankings can be downloaded manually from SCImago and placed in `data/sjr_cache/` as `sjr_<year>.csv`, which the script reads instead of the network.

## Further Resources

| Resource | Link |
|---|---|
| PubMed XML field descriptions | https://www.nlm.nih.gov/bsd/licensee/elements_descriptions.html |
| NCBI E-utilities documentation | https://www.ncbi.nlm.nih.gov/books/NBK25497/ |
| NCBI API key | https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/ |
| NIH iCite | https://icite.od.nih.gov |
| SCImago Journal Rank | https://www.scimagojr.com |
