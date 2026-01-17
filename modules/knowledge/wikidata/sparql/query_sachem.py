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
            ?compound sachem:similarCompoundSearch[
                sachem:query ?QUERY_SMILES;
                sachem:cutoff ?CUTOFF
            ].
        }}
        """
    else:
        return f"""
        SERVICE idsm:wikidata {{
            VALUES ?SUBSTRUCTURE {{ "{escaped_smiles}" }}
            ?compound sachem:substructureSearch [
                sachem:query ?SUBSTRUCTURE
            ].
        }}
        """


def query_sachem(
    escaped_smiles: str,
    search_type: str = "substructure",
    threshold: float = 0.8,
    taxon_qid: str | None = None,
) -> str:
    """
    Build SACHEM chemical search query.

    When taxon_qid is provided, the query is optimized to:
    1. First find compounds in the taxon (fast - uses Wikidata index)
    2. Then filter by structure using SACHEM (applied to smaller set)

    This is MUCH faster than searching SACHEM first, then filtering by taxon.

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
        # OPTIMIZED: Filter by taxon FIRST, then apply SACHEM to the subset
        # This dramatically improves performance for taxon-scoped searches
        return f"""
        {PREFIXES}{SACHEM_PREFIXES}
        SELECT {SELECT_VARS_FULL} WHERE {{
            # First: Find compounds in this taxon (fast - uses index)
            ?taxon (wdt:P171*) wd:{taxon_qid} .
            ?taxon wdt:P225 ?taxon_name .
            ?compound p:P703 ?statement .
            ?statement ps:P703 ?taxon ;
                       prov:wasDerivedFrom ?ref .
            ?ref pr:P248 ?ref_qid .
            
            # Then: Filter by structure using SACHEM (applied to smaller set)
            {sachem_clause}
            
            # Get compound identifiers
            ?compound wdt:P235 ?compound_inchikey ;
                      wdt:P233 ?compound_smiles_conn .
            
            {REFERENCE_METADATA_OPTIONAL}
            {PROPERTIES_OPTIONAL}
        }}
        """
    else:
        # No taxon filter - standard SACHEM search
        return f"""
        {PREFIXES}{SACHEM_PREFIXES}
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
