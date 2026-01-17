"""Cached CDK Depict URL generation."""

__all__ = ["url_from_smiles_cached"]

from functools import lru_cache

from .url import CDK_DEPICT_URL
from .url_from_smiles import url_from_smiles

CACHE_SIZE: int = 256


@lru_cache(maxsize=CACHE_SIZE)
def url_from_smiles_cached(
    smiles: str | None,
    base_url: str = CDK_DEPICT_URL,
    layout: str = "cow",
    img_format: str = "svg",
    annotate: str | None = "cip",
) -> str:
    """
    Generate URL for chemical structure depiction from SMILES (cached).

    Handles None/empty SMILES gracefully by returning empty string.
    Results are cached for performance.
    """
    if not smiles:
        return ""
    return url_from_smiles(
        smiles=smiles,
        base_url=base_url,
        layout=layout,
        img_format=img_format,
        annotate=annotate,
    )
