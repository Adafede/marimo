"""QID link to Scholia."""

__all__ = ["link_from_qid"]

from .styled_anchor import styled_anchor
from .scholia import SCHOLIA_BASE, scholia_url

_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def _extract_qid(url: str) -> str:
    """Extract QID from URL or return as-is if already QID."""
    if not url:
        return ""
    if url.startswith(_ENTITY_PREFIX):
        return url[len(_ENTITY_PREFIX):]
    if url.upper().startswith("Q") and url[1:].isdigit():
        return url.upper()
    return url


def link_from_qid(url_or_qid: str, color: str = "#3377c4") -> str:
    """Create styled HTML link to Scholia for a Wikidata QID or entity URL."""
    if not url_or_qid:
        return ""
    qid = _extract_qid(url_or_qid)
    if not qid:
        return ""
    return styled_anchor(scholia_url(qid), qid, color)
