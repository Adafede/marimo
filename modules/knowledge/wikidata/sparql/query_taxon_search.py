"""Build taxon search query."""

__all__ = ["query_taxon_search"]


def query_taxon_search(taxon_name: str) -> str:
    """Build SPARQL query to find taxa by scientific name."""
    return f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?taxon ?taxon_name WHERE {{
        VALUES ?taxon_name {{ "{taxon_name}" }}
        ?taxon wdt:P225 ?taxon_name .
    }}
    """
