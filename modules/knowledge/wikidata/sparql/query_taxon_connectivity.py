"""Build taxon connectivity query."""

__all__ = ["query_taxon_connectivity"]

from .prefixes import PREFIXES


def query_taxon_connectivity(values_clause_str: str) -> str:
    """Build query to count compound connections for each taxon."""
    return f"""
    {PREFIXES}
    SELECT ?taxon (COUNT(DISTINCT ?compound) AS ?compound_count) WHERE {{
      {values_clause_str}
      {{
        SELECT ?taxon ?compound WHERE {{
          {values_clause_str}
          ?descendant (wdt:P171*) ?taxon .
          ?compound wdt:P235 ?inchikey ;
                    p:P703/ps:P703 ?descendant .
        }}
      }}
    }}
    GROUP BY ?taxon
    ORDER BY DESC(?compound_count)
    """
