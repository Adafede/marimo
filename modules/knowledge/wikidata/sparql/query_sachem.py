"""Build SACHEM chemical search queries."""

__all__ = ["query_sachem"]

from .prefixes import PREFIXES
from .prefixes_sachem import PREFIXES as SACHEM_PREFIXES
from .patterns_compound import (
    SELECT_VARS_FULL,
    PROPERTIES_OPTIONAL,
    REFERENCE_METADATA_OPTIONAL,
)


def query_sachem(
    escaped_smiles: str,
    search_type: str = "substructure",
    threshold: float = 0.8,
    taxon_qid: str | None = None,
) -> str:
    """
    Build SACHEM chemical search query.

    Args:
        escaped_smiles: SMILES string (already escaped for SPARQL)
        search_type: Either "substructure" or "similarity"
        threshold: Tanimoto similarity threshold (0.0-1.0, for similarity search)
        taxon_qid: Optional QID to filter by taxon (e.g., "Q12345")

    Returns:
        Complete SPARQL query string
    """
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
        # substructure search (default)
        sachem_clause = f"""
        SERVICE idsm:wikidata {{
            VALUES ?SUBSTRUCTURE {{ "{escaped_smiles}" }}
            ?compound sachem:substructureSearch [
                sachem:query ?SUBSTRUCTURE
            ].
        }}
        """

    # Build taxon filter if requested
    if taxon_qid:
        taxon_filter = f"?taxon (wdt:P171*) wd:{taxon_qid} ."
    else:
        taxon_filter = ""

    # Construct full query - simpler structure that works with SACHEM
    return f"""
    {PREFIXES}{SACHEM_PREFIXES}
    SELECT {SELECT_VARS_FULL} WHERE {{
        {sachem_clause}
        
        # Get compound identifiers
        ?compound wdt:P235 ?compound_inchikey ;
                  wdt:P233 ?compound_smiles_conn .
        
        # Get taxonomic associations with provenance
        OPTIONAL {{
            ?compound p:P703 ?statement .
            ?statement ps:P703 ?taxon ;
                       prov:wasDerivedFrom ?ref .
            ?ref pr:P248 ?ref_qid .
            ?taxon wdt:P225 ?taxon_name .
            {taxon_filter}
            {REFERENCE_METADATA_OPTIONAL}
        }}
        
        {PROPERTIES_OPTIONAL}
    }}
    """
