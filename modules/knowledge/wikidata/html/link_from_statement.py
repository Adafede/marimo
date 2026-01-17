"""Statement link generation."""

__all__ = ["link_from_statement"]

from .styled_anchor import DEFAULT_LINK_COLOR, styled_anchor


def extract_statement_id(url: str) -> str:
    """Extract statement ID from Wikidata statement URL."""
    return url.split("/")[-1]


def link_from_statement(url: str, color: str = DEFAULT_LINK_COLOR) -> str:
    """Create styled HTML link for a Wikidata statement URL."""
    if not url:
        return ""
    statement_id = extract_statement_id(url=url)
    return styled_anchor(url=url, text=statement_id, color=color)
