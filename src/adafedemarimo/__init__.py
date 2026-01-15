"""
adafedemarimo - Minimal, reusable utilities for data applications.

This package is organized into focused subpackages:

- text/     Pure text utilities (no dependencies)
            - strings: pluralize, truncate
            - smiles: validate_smiles, escape_for_sparql
            - formula: parse_formula, count_element, matches_*

- html/     HTML generation (no dependencies)
            - tags: link, image, styled_link
            - urls: build_query_string, structure_image_url, doi_url

- sparql/   SPARQL client (urllib only)
            - client: SPARQLClient, query
            - retry: with_retry, RetryConfig
            - builders: values_clause, optional_block, prefix_block

- df/       DataFrame utilities (requires polars)
            - filters: filter_range, filter_by_values
            - transforms: rename_columns, extract_from_url, coalesce_columns

- rdf/      RDF utilities (requires rdflib)
            - graph: add_literal, add_resource, bind_namespaces
            - namespaces: WIKIDATA_NAMESPACES, COMMON_NAMESPACES

- wikidata/ Wikidata utilities (no heavy dependencies)
            - qid: extract_qid, is_qid, normalize_qid, entity_url
            - urls: WIKIDATA_SPARQL_ENDPOINT, scholia_url

Usage:
    # Import specific functions from subpackages
    from adafedemarimo.text import validate_smiles, parse_formula
    from adafedemarimo.html import styled_link, structure_image_url
    from adafedemarimo.sparql import SPARQLClient, with_retry
    from adafedemarimo.df import filter_range, rename_columns
    from adafedemarimo.rdf import add_literal, WIKIDATA_NAMESPACES
    from adafedemarimo.wikidata import extract_qid, scholia_url
"""

__version__ = "0.0.1"

# Convenience imports from text (no dependencies)
from adafedemarimo.text import (
    pluralize,
    truncate,
    validate_smiles,
    escape_for_sparql,
    parse_formula,
    count_element,
    normalize_subscripts,
    matches_element_range,
    matches_halogen_constraint,
)

# Convenience imports from html (no dependencies)
from adafedemarimo.html import (
    link,
    image,
    styled_link,
    build_query_string,
    structure_image_url,
    doi_url,
    scholia_url,
)

# Convenience imports from wikidata (no heavy dependencies)
from adafedemarimo.wikidata import (
    extract_qid,
    is_qid,
    normalize_qid,
    entity_url,
    WIKIDATA_SPARQL_ENDPOINT,
    QLEVER_SPARQL_ENDPOINT,
    WIKIDATA_ENTITY_PREFIX,
    WIKIDATA_STATEMENT_PREFIX,
)

__all__ = [
    "__version__",
    # text.strings
    "pluralize",
    "truncate",
    # text.smiles
    "validate_smiles",
    "escape_for_sparql",
    # text.formula
    "parse_formula",
    "count_element",
    "normalize_subscripts",
    "matches_element_range",
    "matches_halogen_constraint",
    # html.tags
    "link",
    "image",
    "styled_link",
    # html.urls
    "build_query_string",
    "structure_image_url",
    "doi_url",
    "scholia_url",
    # wikidata.qid
    "extract_qid",
    "is_qid",
    "normalize_qid",
    "entity_url",
    # wikidata.urls
    "WIKIDATA_SPARQL_ENDPOINT",
    "QLEVER_SPARQL_ENDPOINT",
    "WIKIDATA_ENTITY_PREFIX",
    "WIKIDATA_STATEMENT_PREFIX",
]

