"""Build entity URL from QID."""

__all__ = ["to_url"]

from .prefix import ENTITY_PREFIX


def to_url(qid: str, prefix: str = ENTITY_PREFIX) -> str:
    """Build Wikidata entity URL from QID."""
    return f"{prefix}{qid}"
