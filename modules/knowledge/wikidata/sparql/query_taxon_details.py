"""Build taxon details query."""

__all__ = ["query_taxon_details"]

from .prefixes import PREFIXES


def query_taxon_details(values_clause_str: str) -> str:
    """Build SPARQL query for taxon details."""
    return f"""
    {PREFIXES}
    SELECT
    (xsd:integer(STRAFTER(STR(?t), "Q")) AS ?taxon)
    ?taxonLabel
    ?taxonDescription
    (xsd:integer(STRAFTER(STR(?tp), "Q")) AS ?taxon_parent)
    ?taxon_parentLabel
    WHERE {{
      {values_clause_str}
      OPTIONAL {{ ?t wdt:P171 ?tp }}
      OPTIONAL {{ ?t rdfs:label ?taxonLabel . FILTER(LANG(?taxonLabel) = "en") }}
      OPTIONAL {{ ?t rdfs:label ?taxonLabel . FILTER(LANG(?taxonLabel) = "mul") }}
      OPTIONAL {{ ?t schema:description ?taxonDescription . FILTER(LANG(?taxonDescription) = "en") }}
      OPTIONAL {{ ?t schema:description ?taxonDescription . FILTER(LANG(?taxonDescription) = "mul") }}
      OPTIONAL {{ ?tp rdfs:label ?taxon_parentLabel . FILTER(LANG(?taxon_parentLabel) = "en") }}
      OPTIONAL {{ ?tp rdfs:label ?taxon_parentLabel . FILTER(LANG(?taxon_parentLabel) = "mul") }}
    }}
    """
