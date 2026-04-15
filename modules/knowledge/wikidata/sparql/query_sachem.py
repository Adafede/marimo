"""Build SACHEM chemical search queries."""

__all__ = ["query_sachem"]

from .prefixes import PREFIXES
from .prefixes_sachem import PREFIXES_SACHEM
from .patterns_compound import (
    SELECT_VARS_FULL,
    PROPERTIES_OPTIONAL,
    REFERENCE_METADATA_OPTIONAL,
)


def build_sachem_service(
    structure_literal: str,
    search_type: str,
    threshold: float,
) -> str:
    """Build the SACHEM SERVICE clause.

Parameters
----------
structure_literal : str
    Structure literal.
search_type : str
    Search .
threshold : float
    Threshold.

Returns
-------
str
    Computed result.
    """
    is_multiline_literal = structure_literal.startswith(
        "'''",
    ) or structure_literal.startswith(
        '"""',
    )

    if search_type == "similarity":
        return f"""
    SERVICE idsm:wikidata {{
        ?c sachem:similarCompoundSearch [
            sachem:query {structure_literal};
            sachem:cutoff "{threshold}"^^xsd:double
        ].
    }}"""
    elif is_multiline_literal:
        # Query Molfiles are handled robustly via scoredSubstructureSearch + explicit options.
        return f"""
    SERVICE idsm:wikidata {{
        [ sachem:compound ?c; sachem:score ?_sachem_score ]
            sachem:scoredSubstructureSearch [
                sachem:query {structure_literal};
                sachem:searchMode sachem:substructureSearch;
                sachem:chargeMode sachem:defaultChargeAsAny;
                sachem:isotopeMode sachem:ignoreIsotopes;
                sachem:aromaticityMode sachem:aromaticityDetectIfMissing;
                sachem:stereoMode sachem:ignoreStereo;
                sachem:tautomerMode sachem:ignoreTautomers;
                sachem:radicalMode sachem:ignoreSpinMultiplicity;
                sachem:topn "-1"^^xsd:integer;
                sachem:internalMatchingLimit "1000000"^^xsd:integer
            ]
        .
    }}"""
    else:
        return f"""
    SERVICE idsm:wikidata {{
        ?c sachem:substructureSearch [
            sachem:query {structure_literal}
        ].
    }}"""


def query_sachem(
    escaped_smiles: str,
    search_type: str = "substructure",
    threshold: float = 0.8,
    taxon_qid: str | None = None,
) -> str:
    """Build SACHEM chemical search query.

    OPTIMIZATION: When taxon_qid is provided, we filter by taxonomic data FIRST
    (uses Wikidata's indexes, creates a much smaller set), then apply SACHEM
    SERVICE to the pre-filtered compounds. This is dramatically faster.

Parameters
----------
escaped_smiles : str
    Escaped smiles.
search_type : str
    Default is 'substructure'.
threshold : float
    Default is 0.8.
taxon_qid : str | None
    None. Default is None.

Returns
-------
str
    Computed result.
    """
    sachem_clause = build_sachem_service(
        structure_literal=escaped_smiles,
        search_type=search_type,
        threshold=threshold,
    )
    is_multiline_literal = escaped_smiles.startswith(
        "'''",
    ) or escaped_smiles.startswith(
        '"""',
    )

    if taxon_qid and is_multiline_literal:
        # For multiline query molfiles, keep SERVICE first; pre-binding ?c can trigger
        # unstable behavior in some federated SACHEM executions.
        return f"""
{PREFIXES}
{PREFIXES_SACHEM}
SELECT
{SELECT_VARS_FULL}
WHERE {{
    {sachem_clause}

    # Get compound identifiers
    ?c wdt:P235 ?compound_inchikey ;
              wdt:P233 ?compound_smiles_conn .

    # Require taxonomic association and filter by hierarchy
    ?c p:P703 ?statement .
    ?statement wikibase:rank wikibase:NormalRank ;
               ps:P703 ?t ;
               prov:wasDerivedFrom ?ref .
    ?ref pr:P248 ?r .
    ?t wdt:P225 ?taxon_name .
    ?t (wdt:P171*) wd:{taxon_qid} .

    {REFERENCE_METADATA_OPTIONAL}
    {PROPERTIES_OPTIONAL}
}}
"""
    elif taxon_qid:
        # OPTIMIZED: Filter by taxonomic data FIRST (uses indexes, much smaller set)
        # Then apply SACHEM to pre-filtered compounds
        return f"""
{PREFIXES}
{PREFIXES_SACHEM}
SELECT
{SELECT_VARS_FULL}
WHERE {{
    # Filter compounds with taxonomic data FIRST (much smaller set)
    ?c p:P703 ?statement .
    ?statement wikibase:rank wikibase:NormalRank ;
               ps:P703 ?t ;
               prov:wasDerivedFrom ?ref .
    ?ref pr:P248 ?r .
    ?t wdt:P225 ?taxon_name .

    # Filter by taxon hierarchy
    ?t (wdt:P171*) wd:{taxon_qid} .

    # Then check structural match (filters pre-filtered compounds)
    {sachem_clause}

    # Get compound identifiers
    ?c wdt:P235 ?compound_inchikey ;
              wdt:P233 ?compound_smiles_conn .

    {REFERENCE_METADATA_OPTIONAL}
    {PROPERTIES_OPTIONAL}
}}
"""
    else:
        # No taxon filter - standard SACHEM search with optional taxonomic data
        return f"""
{PREFIXES}
{PREFIXES_SACHEM}
SELECT
{SELECT_VARS_FULL}
WHERE {{
    {sachem_clause}

    # Get compound identifiers
    ?c wdt:P235 ?compound_inchikey ;
              wdt:P233 ?compound_smiles_conn .

    # Get taxonomic associations with provenance (optional)
    OPTIONAL {{
        ?c p:P703 ?statement .
        ?statement ps:P703 ?t ;
                   prov:wasDerivedFrom ?ref .
        ?ref pr:P248 ?r .
        ?t wdt:P225 ?taxon_name .
        {REFERENCE_METADATA_OPTIONAL}
    }}
    {PROPERTIES_OPTIONAL}
}}
"""
