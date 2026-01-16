"""Build VALUES clause for SPARQL."""

__all__ = ["values_clause"]

from typing import List


def values_clause(variable: str, values: List[str], prefix: str = "") -> str:
    """Build a VALUES clause for SPARQL query."""
    if not values:
        return ""
    formatted = " ".join(f"{prefix}{v}" for v in values)
    return f"VALUES ?{variable} {{ {formatted} }}"
