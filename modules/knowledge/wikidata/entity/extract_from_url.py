"""Extract QID from Wikidata entity URL."""

__all__ = ["extract_from_url"]

from typing import Optional

from .prefix import ENTITY_PREFIX


def extract_from_url(url: Optional[str], prefix: str = ENTITY_PREFIX) -> Optional[str]:
    """Extract QID from Wikidata entity URL."""
    if url is None:
        return None
    return url.replace(prefix, "")
