"""Wikidata URL constants."""

__all__ = [
    "ENTITY_PREFIX",
    "QLEVER_ENDPOINT",
    "REFERENCE_PREFIX",
    "SPARQL_ENDPOINT",
    "STATEMENT_PREFIX",
    "WIKIDATA_BASE",
    "WIKIDATA_HTTP_BASE",
    "WIKI_PREFIX",
]

# Base URLs
WIKIDATA_BASE = "https://www.wikidata.org/"
WIKIDATA_HTTP_BASE = "http://www.wikidata.org/"

# Derived prefixes (DRY)
ENTITY_PREFIX = WIKIDATA_HTTP_BASE + "entity/"
STATEMENT_PREFIX = ENTITY_PREFIX + "statement/"
REFERENCE_PREFIX = WIKIDATA_HTTP_BASE + "reference/"
WIKI_PREFIX = WIKIDATA_BASE + "wiki/"

# SPARQL endpoints
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
QLEVER_ENDPOINT = "https://qlever.dev/api/wikidata"
