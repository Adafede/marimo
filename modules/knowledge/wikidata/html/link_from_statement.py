"""Statement link generation."""

__all__ = ["link_from_statement"]

from .styled_anchor import styled_anchor


def link_from_statement(url: str, color: str = "#3377c4") -> str:
    """Create styled HTML link for a Wikidata statement URL."""
    if not url:
        return ""
    statement_id = url.split("/")[-1]
    return styled_anchor(url, statement_id, color)
