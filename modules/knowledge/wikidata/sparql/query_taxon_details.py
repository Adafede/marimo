"""Build taxon details query."""

__all__ = ["query_taxon_details"]

from .prefixes import PREFIXES


def query_taxon_details(values_clause_str: str) -> str:
    """Build SPARQL query for taxon details."""
    return f"""
    {PREFIXES}
    SELECT
    ?taxon
    ?taxonLabel
    ?taxonDescription
    ?taxon_parent
    ?taxon_parentLabel
    WHERE {{
      {values_clause_str}
      OPTIONAL {{ ?taxon wdt:P171 ?taxon_parent }}
      OPTIONAL {{ ?taxon rdfs:label ?taxonLabel . FILTER(LANG(?taxonLabel) = "en") }}
      OPTIONAL {{ ?taxon rdfs:label ?taxonLabel . FILTER(LANG(?taxonLabel) = "mul") }}
      OPTIONAL {{ ?taxon schema:description ?taxonDescription . FILTER(LANG(?taxonDescription) = "en") }}
      OPTIONAL {{ ?taxon schema:description ?taxonDescription . FILTER(LANG(?taxonDescription) = "mul") }}
      OPTIONAL {{ ?taxon_parent rdfs:label ?taxon_parentLabel . FILTER(LANG(?taxon_parentLabel) = "en") }}
      OPTIONAL {{ ?taxon_parent rdfs:label ?taxon_parentLabel . FILTER(LANG(?taxon_parentLabel) = "mul") }}
    }}
    """
