"""Extract QID from Wikidata entity URL."""

__all__ = ["extract_from_url"]

from .prefix import ENTITY_PREFIX


def extract_from_url(url: str | None, prefix: str = ENTITY_PREFIX) -> str | None:
    """Extract QID from Wikidata entity URL."""
    if url is None:
        return None
    return url.replace(prefix, "")
