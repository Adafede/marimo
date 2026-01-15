"""
Wikidata utilities subpackage.

Contains QID utilities and URL constants. No heavy dependencies.
"""

from adafedemarimo.wikidata.qid import (
    extract_qid,
    is_qid,
    normalize_qid,
    entity_url,
)

from adafedemarimo.wikidata.urls import (
    WIKIDATA_SPARQL_ENDPOINT,
    QLEVER_SPARQL_ENDPOINT,
    SCHOLIA_BASE,
    WIKIDATA_ENTITY_PREFIX,
    WIKIDATA_STATEMENT_PREFIX,
    scholia_url,
    wikidata_wiki_url,
)

__all__ = [
    # qid
    "extract_qid",
    "is_qid",
    "normalize_qid",
    "entity_url",
    # urls
    "WIKIDATA_SPARQL_ENDPOINT",
    "QLEVER_SPARQL_ENDPOINT",
    "SCHOLIA_BASE",
    "WIKIDATA_ENTITY_PREFIX",
    "WIKIDATA_STATEMENT_PREFIX",
    "scholia_url",
    "wikidata_wiki_url",
]

