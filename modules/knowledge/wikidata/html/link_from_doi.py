"""DOI constants and link generation."""

__all__ = ["DOI_BASE", "link_from_doi"]

from .styled_anchor import DEFAULT_LINK_COLOR, styled_anchor

DOI_BASE = "https://doi.org/"


def extract_doi(doi: str) -> str:
    """Extract DOI identifier from URL or raw DOI."""
    return doi.split("doi.org/")[-1] if "doi.org/" in doi else doi


def link_from_doi(doi: str, color: str = DEFAULT_LINK_COLOR) -> str:
    """Create styled HTML link for a DOI."""
    if not doi:
        return ""
    clean_doi = extract_doi(doi=doi)
    return styled_anchor(url=f"{DOI_BASE}{clean_doi}", text=clean_doi, color=color)
