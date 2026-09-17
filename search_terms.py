"""
Grammar of the search terms in dzg_search_terms.yaml.

A term is normally matched as written, anywhere in the affiliation and ignoring case.
Two markers at the start of a term change that:

  {noorder}   every word after the marker has to occur somewhere in the affiliation,
              in any order.

  {nosearch}  the term is not sent to PubMed and never proves membership in the DZG.
              Use it for wording that is too general, which would return many unrelated articles. 
              It only fills the site column.
"""

from typing import NamedTuple

import pandas as pd

KNOWN_MARKERS = {"noorder", "nosearch"}


class SearchTerm(NamedTuple):
    """One entry of a search term list, with its markers already read."""

    words: list[str]   # all of these have to occur in the affiliation
    searchable: bool   # False for {nosearch}, which limits the term to the site column


def parse_term(raw: str) -> SearchTerm:
    """Split one YAML entry into its markers and the words to look for.
    Raises ValueError on an unknown marker.
    """
    text = raw.strip()
    markers = set()

    while text.startswith("{"):
        marker, closing, rest = text.partition("}")
        if not closing:
            raise ValueError(f"Search term {raw!r} has an opening brace without a closing one")
        markers.add(marker[1:].strip().lower())
        text = rest.strip()

    unknown = markers - KNOWN_MARKERS
    if unknown:
        raise ValueError(
            f"Search term {raw!r} uses unknown marker(s) {sorted(unknown)}. "
            f"Known markers are {sorted(KNOWN_MARKERS)}."
        )
    if not text:
        raise ValueError(f"Search term {raw!r} has markers but no words to search for")

    words = text.split() if "noorder" in markers else [text]
    return SearchTerm(words=words, searchable="nosearch" not in markers)


def dzg_terms(networks: dict) -> list[str]:
    """Collect the terms of one DZG that identify the DZG itself.
    Runs over the general block and every site block. {nosearch} terms are skipped.
    """
    terms = []
    for network in networks.values():
        for raw in network:
            if parse_term(raw).searchable:
                terms.append(raw)
    return terms


def build_query(networks: dict) -> str:
    """Build the PubMed affiliation query for one DZG.
    Every term becomes one clause and the clauses are joined with OR. A {noorder} term
    carries several words, those become an AND group in brackets so that PubMed only
    returns affiliations containing all of them.
    """
    clauses = []
    for raw in dzg_terms(networks):
        words  = parse_term(raw).words
        fields = [f'"{word}"[Affiliation]' for word in words]
        if len(fields) == 1:
            clauses.append(fields[0])
        else:
            clauses.append("(" + " AND ".join(fields) + ")")
    return " OR ".join(clauses)


def affiliation_matches(affiliations: pd.Series, terms: list[str]) -> pd.Series:
    """Mark the affiliations that match any of the search terms.
    A term matches when all of its words occur in the affiliation, ignoring case.
    """
    word_groups = []
    for raw in terms:
        word_groups.append([word.lower() for word in parse_term(raw).words])

    def matches_any_term(affiliation: str) -> bool:
        text = str(affiliation).lower()
        for words in word_groups:
            if all(word in text for word in words):
                return True
        return False

    return affiliations.apply(matches_any_term)