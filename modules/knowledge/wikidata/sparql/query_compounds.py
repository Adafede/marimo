"""Build compound queries for Wikidata."""

__all__ = ["query_compounds_by_taxon", "query_all_compounds"]

from .prefixes import PREFIXES
from .patterns_compound import (
    SELECT_VARS_FULL,
    SELECT_VARS_INTERIM,
    COMPOUND_IDENTIFIERS,
    TAXON_REFERENCE_ASSOCIATION,
    PROPERTIES_OPTIONAL,
    REFERENCE_METADATA_OPTIONAL,
)


def query_compounds_by_taxon(qid: str) -> str:
    """Build SPARQL query to find compounds in a specific taxon and its descendants.

    Parameters
    ----------
    qid : str
        Qid.

    Returns
    -------
    str
        String representation of query compounds by taxon.
    """
    return f"""
    {PREFIXES}
    SELECT
    {SELECT_VARS_FULL}
    WHERE {{
      {{
        SELECT
        {SELECT_VARS_INTERIM}
        WHERE {{
          {COMPOUND_IDENTIFIERS}
          {TAXON_REFERENCE_ASSOCIATION}
        }}
      }}
      ?t (wdt:P171*) wd:{qid}.
      {REFERENCE_METADATA_OPTIONAL}
      {PROPERTIES_OPTIONAL}
    }}
    """


def query_all_compounds() -> str:
    """Build SPARQL query to retrieve all compounds.

    Returns
    -------
    str
        String representation of query all compounds.
    """
    return f"""
    {PREFIXES}
    SELECT
    {SELECT_VARS_FULL}
    WHERE {{
        {{
            SELECT
            {SELECT_VARS_INTERIM}
            WHERE {{
                {COMPOUND_IDENTIFIERS}
                {TAXON_REFERENCE_ASSOCIATION}
            }}
        }}
      {REFERENCE_METADATA_OPTIONAL}
      {PROPERTIES_OPTIONAL}
    }}
    """
