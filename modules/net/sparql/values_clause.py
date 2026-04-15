"""Build VALUES clause for SPARQL."""

__all__ = ["values_clause"]

from collections.abc import Sequence


def values_clause(variable: str, values: Sequence[str], prefix: str = "") -> str:
    """Build a VALUES clause for SPARQL query.

    Parameters
    ----------
    variable : str
        Variable.
    values : Sequence[str]
        Values.
    prefix : str
        Default is ''.

    Returns
    -------
    str
        String representation of values clause.
    """
    if not values:
        return ""
    formatted = " ".join(f"{prefix}{v}" for v in values)
    return f"VALUES ?{variable} {{ {formatted} }}"
