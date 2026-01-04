"""
Query module for LOTUS Wikidata Explorer.

Provides SPARQL query building, execution, and validation capabilities.
"""

from .validation import validate_smiles, escape_smiles_for_sparql
from .builders import (
    build_taxon_search_query,
    build_base_sachem_query,
    build_smiles_substructure_query,
    build_smiles_similarity_query,
    build_smiles_taxon_query,
    build_compounds_query,
    build_all_compounds_query,
    build_sparql_values_clause,
    build_taxon_details_query,
    build_taxon_connectivity_query,
)
from .executor import execute_sparql, get_sparql_wrapper

__all__ = [
    # Validation
    "validate_smiles",
    "escape_smiles_for_sparql",
    # Query builders
    "build_taxon_search_query",
    "build_base_sachem_query",
    "build_smiles_substructure_query",
    "build_smiles_similarity_query",
    "build_smiles_taxon_query",
    "build_compounds_query",
    "build_all_compounds_query",
    "build_sparql_values_clause",
    "build_taxon_details_query",
    "build_taxon_connectivity_query",
    # Execution
    "execute_sparql",
    "get_sparql_wrapper",
]
