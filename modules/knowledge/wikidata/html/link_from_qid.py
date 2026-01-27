"""QID link to Scholia."""

__all__ = ["link_from_qid"]

from .scholia import scholia_url
from .styled_anchor import styled_anchor

ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def extract_qid_from_url_or_qid(value: str | int) -> str:
    """Extract QID from URL, int, or return as proper QID string."""
    if value is None or value == "":
        return ""

    # Handle integers directly
    if isinstance(value, int):
        return f"Q{value}"

    # Convert to string for further checks
    val_str = str(value).strip()

    # Handle URLs
    if val_str.startswith(ENTITY_PREFIX):
        return "Q" + val_str[len(ENTITY_PREFIX) :]

    # Handle strings already like Q42
    if val_str.upper().startswith("Q") and val_str[1:].isdigit():
        return val_str.upper()

    # Anything else, just return as string
    return val_str


def link_from_qid(url_or_qid: str, color: str = "#3377c4") -> str:
    """Create styled HTML link to Scholia for a Wikidata QID or entity URL."""
    if not url_or_qid:
        return ""
    qid = extract_qid_from_url_or_qid(value=url_or_qid)
    if not qid:
        return ""
    return styled_anchor(url=scholia_url(qid=qid), text=qid, color=color)
