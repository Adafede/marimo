"""Extract QID from Wikidata entity URL."""

__all__ = ["extract_from_url"]

from .prefix import ENTITY_PREFIX


def extract_from_url(url: str | None, prefix: str = ENTITY_PREFIX) -> str | None:
    """Extract QID from Wikidata entity URL.

    Parameters
    ----------
    url : str | None
        Url.
    prefix : str
        ENTITY_PREFIX. Default is ENTITY_PREFIX.

    Returns
    -------
    str | None
        Extracted from url.

    """
    if url is None:
        return None
    return url.replace(prefix, "")
