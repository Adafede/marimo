"""
Core utilities, configuration, and constants for LOTUS Wikidata Explorer.

This module provides the foundational infrastructure used throughout the application.
"""

from .config import CONFIG, ELEMENT_CONFIGS, HALOGEN_CONFIGS, EXPORT_FORMATS
from .constants import (
    # RDF Namespaces
    WD,
    WDREF,
    WDS,
    WDT,
    P,
    PS,
    PR,
    PROV,
    SCHEMA,
    # URLs
    SCHOLIA_URL,
    WIKIDATA_URL,
    WIKIDATA_HTTP_URL,
    WIKIDATA_ENTITY_URL,
    WIKIDATA_WIKI_URL,
    # SPARQL fragments
    SPARQL_PREFIXES,
    SACHEM_PREFIXES,
    COMPOUND_SELECT_VARS,
    COMPOUND_MINIMAL_VARS,
    COMPOUND_INTERIM_VARS,
    COMPOUND_IDENTIFIERS,
    TAXON_REFERENCE_ASSOCIATION,
    COMPOUND_PROPERTIES_OPTIONAL,
    TAXONOMIC_REFERENCE_OPTIONAL,
    REFERENCE_METADATA_OPTIONAL,
    # Translation maps
    SUBSCRIPT_MAP,
    FORMULA_PATTERN,
    PLURAL_MAP,
)
from .models import ElementRange, FormulaFilters
from .utils import (
    extract_qid,
    extract_qids_from_dataframe,
    get_binding_value,
)

__all__ = [
    # Config
    "CONFIG",
    "ELEMENT_CONFIGS",
    "HALOGEN_CONFIGS",
    "EXPORT_FORMATS",
    # Constants
    "WD",
    "WDREF",
    "WDS",
    "WDT",
    "P",
    "PS",
    "PR",
    "PROV",
    "SCHEMA",
    "SCHOLIA_URL",
    "WIKIDATA_URL",
    "WIKIDATA_HTTP_URL",
    "WIKIDATA_ENTITY_URL",
    "WIKIDATA_WIKI_URL",
    "SPARQL_PREFIXES",
    "SACHEM_PREFIXES",
    "COMPOUND_SELECT_VARS",
    "COMPOUND_MINIMAL_VARS",
    "COMPOUND_INTERIM_VARS",
    "COMPOUND_IDENTIFIERS",
    "TAXON_REFERENCE_ASSOCIATION",
    "COMPOUND_PROPERTIES_OPTIONAL",
    "TAXONOMIC_REFERENCE_OPTIONAL",
    "REFERENCE_METADATA_OPTIONAL",
    "SUBSCRIPT_MAP",
    "FORMULA_PATTERN",
    "PLURAL_MAP",
    # Models
    "ElementRange",
    "FormulaFilters",
    # Utils
    "extract_qid",
    "extract_qids_from_dataframe",
    "get_binding_value",
]
