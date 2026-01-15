"""
HTTP client for SPARQL endpoints - minimal dependencies (urllib only).

A simple, reusable SPARQL client that works in both native Python
and browser (Pyodide/WASM) environments.
"""

__all__ = [
    "SPARQLClient",
    "query",
]

import sys
import urllib.parse
import urllib.request
from typing import Optional

# Patch urllib for Pyodide/WASM (browser) compatibility
if "pyodide" in sys.modules:
    import pyodide_http
    pyodide_http.patch_all()


class SPARQLClient:
    """
    Simple SPARQL client using urllib.
    
    Works in both native Python and Pyodide/WASM environments.
    
    Example:
        >>> client = SPARQLClient("https://query.wikidata.org/sparql")
        >>> csv_bytes = client.query_csv("SELECT ?item WHERE { ?item wdt:P31 wd:Q5 } LIMIT 10")
    """
    
    def __init__(
        self,
        endpoint: str,
        timeout: int = 120,
        user_agent: str = "adafedemarimo/1.0",
    ):
        """
        Initialize SPARQL client.
        
        Args:
            endpoint: SPARQL endpoint URL
            timeout: Query timeout in seconds
            user_agent: User agent string for requests
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.user_agent = user_agent
    
    def query_csv(self, query: str) -> bytes:
        """
        Execute a SPARQL query, returning CSV bytes.
        
        Args:
            query: SPARQL query string
        
        Returns:
            Raw CSV bytes
        
        Raises:
            urllib.error.URLError: On network errors
            urllib.error.HTTPError: On HTTP errors
        """
        headers = {
            "Accept": "text/csv",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": self.user_agent,
        }
        data = urllib.parse.urlencode({"query": query}).encode("utf-8")
        
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return response.read()
    
    def query_json(self, query: str) -> bytes:
        """
        Execute a SPARQL query, returning JSON bytes.
        
        Args:
            query: SPARQL query string
        
        Returns:
            Raw JSON bytes
        """
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": self.user_agent,
        }
        data = urllib.parse.urlencode({"query": query}).encode("utf-8")
        
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return response.read()


def query(
    sparql_query: str,
    endpoint: str,
    timeout: int = 120,
    format: str = "csv",
) -> bytes:
    """
    Execute a SPARQL query (convenience function).
    
    Args:
        sparql_query: SPARQL query string
        endpoint: SPARQL endpoint URL
        timeout: Query timeout in seconds
        format: Response format ("csv" or "json")
    
    Returns:
        Raw response bytes
    
    Example:
        >>> result = query(
        ...     "SELECT ?item WHERE { ?item wdt:P31 wd:Q5 } LIMIT 10",
        ...     "https://query.wikidata.org/sparql"
        ... )
    """
    client = SPARQLClient(endpoint, timeout)
    if format == "json":
        return client.query_json(sparql_query)
    return client.query_csv(sparql_query)

