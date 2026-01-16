"""DOI constants and link generation."""

__all__ = ["DOI_BASE", "link_from_doi"]

from .styled_anchor import styled_anchor

DOI_BASE = "https://doi.org/"


def link_from_doi(doi: str, color: str = "#3377c4") -> str:
    """Create styled HTML link for a DOI."""
    if not doi:
        return ""
    clean_doi = doi.split("doi.org/")[-1] if "doi.org/" in doi else doi
    return styled_anchor(f"{DOI_BASE}{clean_doi}", clean_doi, color)
