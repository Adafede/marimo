"""Build SACHEM chemical search queries."""

__all__ = ["query_sachem"]

from .prefixes import PREFIXES
from .prefixes_sachem import PREFIXES as SACHEM_PREFIXES
from .patterns_compound import (
    SELECT_VARS_FULL,
    PROPERTIES_OPTIONAL,
    REFERENCE_METADATA_OPTIONAL,
)


def _build_sachem_service(
    escaped_smiles: str,
    search_type: str,
    threshold: float,
) -> str:
    """Build the SACHEM SERVICE clause."""
    if search_type == "similarity":
        return f"""
    SERVICE idsm:wikidata {{
        VALUES ?QUERY_SMILES {{ "{escaped_smiles}" }}
        VALUES ?CUTOFF {{ "{threshold}"^^xsd:double }}
        ?compound sachem:similarCompoundSearch [
            sachem:query ?QUERY_SMILES;
            sachem:cutoff ?CUTOFF
        ].
    }}"""
    else:
        return f"""
    SERVICE idsm:wikidata {{
        ?compound sachem:substructureSearch [
            sachem:query "{escaped_smiles}"
        ].
    }}"""


def query_sachem(
    escaped_smiles: str,
    search_type: str = "substructure",
    threshold: float = 0.8,
    taxon_qid: str | None = None,
) -> str:
    """
    Build SACHEM chemical search query.

    OPTIMIZATION: When taxon_qid is provided, we filter by taxonomic data FIRST
    (uses Wikidata's indexes, creates a much smaller set), then apply SACHEM
    SERVICE to the pre-filtered compounds. This is dramatically faster.

    Args:
        escaped_smiles: SMILES string (already escaped for SPARQL)
        search_type: Either "substructure" or "similarity"
        threshold: Tanimoto similarity threshold (0.0-1.0, for similarity search)
        taxon_qid: Optional QID to filter by taxon (e.g., "Q12345")

    Returns:
        Complete SPARQL query string
    """
    sachem_clause = _build_sachem_service(escaped_smiles, search_type, threshold)

    if taxon_qid:
        # OPTIMIZED: Filter by taxonomic data FIRST (uses indexes, much smaller set)
        # Then apply SACHEM to pre-filtered compounds
        return f"""
{PREFIXES}
{SACHEM_PREFIXES}
SELECT {SELECT_VARS_FULL} WHERE {{
    # Filter compounds with taxonomic data FIRST (much smaller set)
    ?compound p:P703 ?statement .
    ?statement wikibase:rank wikibase:NormalRank ;
               ps:P703 ?taxon ;
               prov:wasDerivedFrom ?ref .
    ?ref pr:P248 ?ref_qid .
    ?taxon wdt:P225 ?taxon_name .

    # Filter by taxon hierarchy
    ?taxon (wdt:P171*) wd:{taxon_qid} .

    # Then check structural match (filters pre-filtered compounds)
    {sachem_clause}
    
    # Get compound identifiers
    ?compound wdt:P235 ?compound_inchikey ;
              wdt:P233 ?compound_smiles_conn .
    
    {REFERENCE_METADATA_OPTIONAL}
    {PROPERTIES_OPTIONAL}
}}
"""
    else:
        # No taxon filter - standard SACHEM search with optional taxonomic data
        return f"""
{PREFIXES}
{SACHEM_PREFIXES}
SELECT {SELECT_VARS_FULL} WHERE {{
    {sachem_clause}

    # Get compound identifiers
    ?compound wdt:P235 ?compound_inchikey ;
              wdt:P233 ?compound_smiles_conn .

    # Get taxonomic associations with provenance (optional)
    OPTIONAL {{
        ?compound p:P703 ?statement .
        ?statement ps:P703 ?taxon ;
                   prov:wasDerivedFrom ?ref .
        ?ref pr:P248 ?ref_qid .
        ?taxon wdt:P225 ?taxon_name .
        {REFERENCE_METADATA_OPTIONAL}
    }}

    {PROPERTIES_OPTIONAL}
}}
"""
