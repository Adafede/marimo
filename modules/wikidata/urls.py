"""
Wikidata URL constants and builders - no external dependencies.

URLs and utility functions for Wikidata services.
"""

__all__ = [
    "WIKIDATA_SPARQL_ENDPOINT",
    "QLEVER_SPARQL_ENDPOINT", 
    "SCHOLIA_BASE",
    "WIKIDATA_ENTITY_PREFIX",
    "WIKIDATA_STATEMENT_PREFIX",
    "scholia_url",
    "wikidata_wiki_url",
]


# SPARQL endpoints
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
QLEVER_SPARQL_ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"

# URL prefixes
WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
WIKIDATA_STATEMENT_PREFIX = "http://www.wikidata.org/entity/statement/"

# Service URLs
SCHOLIA_BASE = "https://scholia.toolforge.org/"


def scholia_url(qid: str) -> str:
    """
    Build Scholia URL for a Wikidata QID.
    
    Args:
        qid: Wikidata QID (e.g., "Q12345")
    
    Returns:
        Scholia URL
    
    Example:
        >>> scholia_url("Q12345")
        'https://scholia.toolforge.org/Q12345'
    """
    return f"{SCHOLIA_BASE}{qid}"


def wikidata_wiki_url(qid: str) -> str:
    """
    Build Wikidata wiki URL for a QID.
    
    Args:
        qid: Wikidata QID
    
    Returns:
        Wikidata wiki URL
    
    Example:
        >>> wikidata_wiki_url("Q12345")
        'https://www.wikidata.org/wiki/Q12345'
    """
    return f"https://www.wikidata.org/wiki/{qid}"

