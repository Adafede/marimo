"""
Wikidata QID utilities - no external dependencies.

Pure functions for extracting and validating Wikidata QIDs.
"""

__all__ = [
    "extract_qid",
    "is_qid",
    "normalize_qid",
    "entity_url",
]

from typing import Optional

# Default Wikidata entity URL prefix
_DEFAULT_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def extract_qid(
    url: Optional[str],
    prefix: str = _DEFAULT_ENTITY_PREFIX,
) -> Optional[str]:
    """
    Extract QID from Wikidata entity URL.
    
    Args:
        url: Wikidata entity URL (e.g., "http://www.wikidata.org/entity/Q12345")
        prefix: URL prefix to remove
    
    Returns:
        QID string (e.g., "Q12345") or None if url is None
    
    Example:
        >>> extract_qid("http://www.wikidata.org/entity/Q12345")
        'Q12345'
        >>> extract_qid(None)
        None
    """
    if url is None:
        return None
    return url.replace(prefix, "")


def is_qid(value: str) -> bool:
    """
    Check if a string is a valid Wikidata QID format.
    
    Args:
        value: String to check
    
    Returns:
        True if string matches QID format (Q followed by digits)
    
    Example:
        >>> is_qid("Q12345")
        True
        >>> is_qid("Aspirin")
        False
    """
    if not value:
        return False
    value = value.strip().upper()
    return value.startswith("Q") and value[1:].isdigit()


def normalize_qid(value: str) -> str:
    """
    Normalize a QID to uppercase.
    
    Args:
        value: QID string (possibly lowercase)
    
    Returns:
        Uppercase QID
    
    Example:
        >>> normalize_qid("q12345")
        'Q12345'
    """
    return value.strip().upper()


def entity_url(
    qid: str,
    prefix: str = _DEFAULT_ENTITY_PREFIX,
) -> str:
    """
    Build Wikidata entity URL from QID.
    
    Args:
        qid: Wikidata QID
        prefix: URL prefix
    
    Returns:
        Full entity URL
    
    Example:
        >>> entity_url("Q12345")
        'http://www.wikidata.org/entity/Q12345'
    """
    return f"{prefix}{qid}"

