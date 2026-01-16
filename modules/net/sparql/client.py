"""SPARQL HTTP client class."""

__all__ = ["Client"]

import sys
import urllib.parse
import urllib.request

if "pyodide" in sys.modules:
    import pyodide_http
    pyodide_http.patch_all()


class Client:
    """Simple SPARQL client using urllib."""

    def __init__(
        self,
        endpoint: str,
        timeout: int = 120,
        user_agent: str = "adafedemarimo/1.0",
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.user_agent = user_agent

    def _request(self, query: str, accept: str) -> bytes:
        """Execute HTTP request to SPARQL endpoint."""
        headers = {
            "Accept": accept,
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": self.user_agent,
        }
        data = urllib.parse.urlencode({"query": query}).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return response.read()

    def query_csv(self, query: str) -> bytes:
        """Execute query returning CSV bytes."""
        return self._request(query, "text/csv")

    def query_json(self, query: str) -> bytes:
        """Execute query returning JSON bytes."""
        return self._request(query, "application/sparql-results+json")
