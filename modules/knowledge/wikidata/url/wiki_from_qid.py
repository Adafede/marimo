"""Build Wikidata wiki URL from QID."""

__all__ = ["wiki_from_qid"]

from .constants import WIKI_PREFIX


def wiki_from_qid(qid: str) -> str:
    """Build Wikidata wiki URL for a QID."""
    return f"{WIKI_PREFIX}{qid}"
