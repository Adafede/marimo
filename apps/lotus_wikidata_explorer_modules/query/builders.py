"""
SPARQL query builders for LOTUS Wikidata Explorer.

Contains all functions that construct SPARQL queries for various search scenarios.
"""

from typing import Optional, List

from ..core.constants import (
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
)
from .validation import escape_smiles_for_sparql

__all__ = [
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
]


def build_taxon_search_query(taxon_name: str) -> str:
    """Build SPARQL query to find taxa by scientific name. Returns up to 10 results."""
    return f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?taxon ?taxon_name WHERE {{
        VALUES ?taxon_name {{ "{taxon_name}" }}
        ?taxon wdt:P225 ?taxon_name .
    }}
    """


def build_base_sachem_query(
    smiles: str,
    search_type: str = "substructure",
    threshold: float = 0.8,
    include_taxon_filter: bool = False,
    taxon_qid: Optional[str] = None,
) -> str:
    """
    Build base SACHEM chemical search query.

    Args:
        smiles: SMILES string for chemical structure
        search_type: 'substructure' or 'similarity'
        threshold: Similarity threshold (0.0-1.0) for similarity searches
        include_taxon_filter: Whether to filter by taxon
        taxon_qid: QID of taxon to filter by (if include_taxon_filter=True)

    Returns:
        Complete SPARQL query string
    """
    escaped_smiles = escape_smiles_for_sparql(smiles)

    # Build SACHEM service clause based on search type
    if search_type == "similarity":
        sachem_clause = f"""
        SERVICE idsm:wikidata {{
            VALUES ?QUERY_SMILES {{ "{escaped_smiles}" }}
            VALUES ?CUTOFF {{ "{threshold}"^^xsd:double }}
            ?compound sachem:similarCompoundSearch[
            sachem:query ?QUERY_SMILES;
            sachem:cutoff ?CUTOFF
            ].
        }}
        """
    else:
        # substructure
        sachem_clause = f"""
        SERVICE idsm:wikidata {{
            VALUES ?SUBSTRUCTURE {{ "{escaped_smiles}" }}
            ?compound sachem:substructureSearch [
                sachem:query ?SUBSTRUCTURE
            ].
        }}
        """

    # Build taxon filter if requested
    taxon_filter = ""
    if include_taxon_filter and taxon_qid:
        taxon_filter = f"""
        {TAXON_REFERENCE_ASSOCIATION}
        ?taxon (wdt:P171*) wd:{taxon_qid}
        {REFERENCE_METADATA_OPTIONAL}
        """
    else:
        taxon_filter = f"""
        {TAXONOMIC_REFERENCE_OPTIONAL}
        """

    # Construct full query
    return f"""
    {SPARQL_PREFIXES}{SACHEM_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
        {{
            SELECT {COMPOUND_MINIMAL_VARS} WHERE {{
                {sachem_clause}
                {COMPOUND_IDENTIFIERS}
            }}
        }}
        {taxon_filter}
        {COMPOUND_PROPERTIES_OPTIONAL}
    }}
    """


def build_smiles_substructure_query(smiles: str) -> str:
    """Build SPARQL query for chemical substructure search using SACHEM."""
    return build_base_sachem_query(smiles, search_type="substructure")


def build_smiles_similarity_query(smiles: str, threshold: float = 0.8) -> str:
    """Build SPARQL query for chemical similarity search using SACHEM."""
    return build_base_sachem_query(
        smiles, search_type="similarity", threshold=threshold
    )


def build_smiles_taxon_query(
    smiles: str, qid: str, search_type: str = "substructure", threshold: float = 0.8
) -> str:
    """Build SPARQL query to find compounds by SMILES within a specific taxon."""
    return build_base_sachem_query(
        smiles,
        search_type=search_type,
        threshold=threshold,
        include_taxon_filter=True,
        taxon_qid=qid,
    )


def build_compounds_query(qid: str) -> str:
    """Build SPARQL query to find compounds in a specific taxon and its descendants."""
    return f"""
    {SPARQL_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
      {{
        SELECT {COMPOUND_INTERIM_VARS} WHERE {{
          {COMPOUND_IDENTIFIERS}
          {TAXON_REFERENCE_ASSOCIATION}
        }}
      }}
      ?taxon (wdt:P171*) wd:{qid}.
      {REFERENCE_METADATA_OPTIONAL}      
      {COMPOUND_PROPERTIES_OPTIONAL}
    }}
    """


def build_all_compounds_query() -> str:
    """Build SPARQL query to retrieve all compounds."""
    return f"""
    {SPARQL_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
        {{
            SELECT {COMPOUND_INTERIM_VARS} WHERE {{
                    {COMPOUND_IDENTIFIERS}
                    {TAXON_REFERENCE_ASSOCIATION}
            }}
        }}
      {REFERENCE_METADATA_OPTIONAL}
      {COMPOUND_PROPERTIES_OPTIONAL}
    }}
    """


def build_sparql_values_clause(
    variable: str, values: List[str], use_wd_prefix: bool = True
) -> str:
    """
    Build a SPARQL VALUES clause for a list of values.

    Args:
        variable: Variable name (without ?)
        values: List of values
        use_wd_prefix: Whether to prefix values with 'wd:' (for QIDs)

    Returns:
        SPARQL VALUES clause
    """
    if use_wd_prefix:
        values_str = " ".join(f"wd:{v}" for v in values)
    else:
        values_str = " ".join(f"<{v}>" for v in values)

    return f"VALUES ?{variable} {{ {values_str} }}"


def build_taxon_details_query(qids: List[str]) -> str:
    """Build query to retrieve detailed information for taxa."""
    values_clause = build_sparql_values_clause("taxon", qids)
    return f"""
    {SPARQL_PREFIXES}

    SELECT ?taxon ?taxonLabel ?taxonDescription ?taxon_parent ?taxon_parentLabel 
    WHERE {{
      {values_clause}

      # Parent taxon
      OPTIONAL {{ ?taxon wdt:P171 ?taxon_parent }}

      # Taxon labels (English preferred)
      OPTIONAL {{
        ?taxon rdfs:label ?taxonLabel .
        FILTER(LANG(?taxonLabel) = "en")
      }}
      OPTIONAL {{
        ?taxon rdfs:label ?taxonLabel .
        FILTER(LANG(?taxonLabel) = "mul")
      }}

      # Taxon descriptions
      OPTIONAL {{
        ?taxon schema:description ?taxonDescription .
        FILTER(LANG(?taxonDescription) = "en")
      }}
      OPTIONAL {{
        ?taxon schema:description ?taxonDescription .
        FILTER(LANG(?taxonDescription) = "mul")
      }}

      # Parent labels
      OPTIONAL {{
        ?taxon_parent rdfs:label ?taxon_parentLabel .
        FILTER(LANG(?taxon_parentLabel) = "en")
      }}
      OPTIONAL {{
        ?taxon_parent rdfs:label ?taxon_parentLabel .
        FILTER(LANG(?taxon_parentLabel) = "mul")
      }}
    }}
    """


def build_taxon_connectivity_query(qids: List[str]) -> str:
    """Build query to count compound connections for each taxon."""
    values_clause = build_sparql_values_clause("taxon", qids)
    return f"""
    {SPARQL_PREFIXES}

    SELECT ?taxon (COUNT(DISTINCT ?compound) AS ?compound_count) WHERE {{
      {values_clause}

      # Count compounds directly linked to this taxon or its descendants
      {{
        SELECT ?taxon ?compound WHERE {{
          {values_clause}
          ?descendant (wdt:P171*) ?taxon .
          ?compound wdt:P235 ?inchikey ;
                    p:P703/ps:P703 ?descendant .
        }}
      }}
    }}
    GROUP BY ?taxon
    ORDER BY DESC(?compound_count)
    """
