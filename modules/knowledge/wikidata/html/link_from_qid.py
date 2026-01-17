"""QID link to Scholia."""

__all__ = ["link_from_qid"]

from .scholia import scholia_url
from .styled_anchor import styled_anchor

_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def _extract_qid_from_url_or_qid(value: str) -> str:
    """Extract QID from URL or return as-is if already QID."""
    if not value:
        return ""
    if value.startswith(_ENTITY_PREFIX):
        return value[len(_ENTITY_PREFIX) :]
    if value.upper().startswith("Q") and value[1:].isdigit():
        return value.upper()
    return value


def link_from_qid(url_or_qid: str, color: str = "#3377c4") -> str:
    """Create styled HTML link to Scholia for a Wikidata QID or entity URL."""
    if not url_or_qid:
        return ""
    qid = _extract_qid_from_url_or_qid(value=url_or_qid)
    if not qid:
        return ""
    return styled_anchor(url=scholia_url(qid=qid), text=qid, color=color)
