"""Build taxon search query."""

__all__ = ["query_taxon_search"]


def query_taxon_search(taxon_name: str) -> str:
    """Build SPARQL query to find taxa by scientific name.

    Parameters
    ----------
    taxon_name : str
        Taxon name.

    Returns
    -------
    str
        String representation of query taxon search.

    """
    return f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT
    ?taxon
    ?taxon_name WHERE {{
        VALUES ?taxon_name {{ "{taxon_name}" }}
        ?taxon wdt:P225 ?taxon_name .
    }}
    """
