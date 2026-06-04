"""Build taxon connectivity query."""

__all__ = ["query_taxon_connectivity"]

from .prefixes import PREFIXES


def query_taxon_connectivity(values_clause_str: str) -> str:
    """Build query to count compound connections for each taxon.

    Parameters
    ----------
    values_clause_str : str
        Values clause str.

    Returns
    -------
    str
        String representation of query taxon connectivity.

    """
    return f"""
    {PREFIXES}
    SELECT
    ?taxon
    (COUNT(?taxon) AS ?count)
    WHERE {{
      {values_clause_str}
      ?s ?p ?taxon.
    }}
    GROUP BY ?taxon
    ORDER BY DESC(?count)
    """
