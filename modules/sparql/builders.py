"""
SPARQL query building utilities - no external dependencies.

Pure functions for constructing SPARQL query fragments.
"""

__all__ = [
    "values_clause",
    "optional_block",
    "filter_lang",
    "prefix_block",
]

from typing import List, Optional


def values_clause(
    variable: str,
    values: List[str],
    prefix: Optional[str] = None,
) -> str:
    """
    Build a SPARQL VALUES clause.

    Args:
        variable: Variable name (without ?)
        values: List of values
        prefix: Optional prefix to prepend to each value (e.g., "wd:")

    Returns:
        SPARQL VALUES clause string

    Example:
        >>> values_clause("item", ["Q1", "Q2", "Q3"], prefix="wd:")
        'VALUES ?item { wd:Q1 wd:Q2 wd:Q3 }'
        >>> values_clause("name", ["Alice", "Bob"])
        'VALUES ?name { "Alice" "Bob" }'
    """
    if prefix:
        values_str = " ".join(f"{prefix}{v}" for v in values)
    else:
        values_str = " ".join(f'"{v}"' for v in values)

    return f"VALUES ?{variable} {{ {values_str} }}"


def optional_block(content: str) -> str:
    """
    Wrap content in OPTIONAL block.

    Args:
        content: SPARQL pattern to make optional

    Returns:
        OPTIONAL { content } string

    Example:
        >>> optional_block("?item wdt:P31 ?type .")
        'OPTIONAL { ?item wdt:P31 ?type . }'
    """
    return f"OPTIONAL {{ {content} }}"


def filter_lang(variable: str, lang: str = "en") -> str:
    """
    Generate language filter for a variable.

    Args:
        variable: Variable name (without ?)
        lang: Language code

    Returns:
        FILTER clause string

    Example:
        >>> filter_lang("label", "en")
        'FILTER(LANG(?label) = "en")'
    """
    return f'FILTER(LANG(?{variable}) = "{lang}")'


def prefix_block(prefixes: dict[str, str]) -> str:
    """
    Generate PREFIX declarations.

    Args:
        prefixes: Dict mapping prefix names to URIs

    Returns:
        PREFIX declarations string

    Example:
        >>> prefix_block({"wd": "http://www.wikidata.org/entity/"})
        'PREFIX wd: <http://www.wikidata.org/entity/>'
    """
    lines = [f"PREFIX {name}: <{uri}>" for name, uri in prefixes.items()]
    return "\n".join(lines)
