"""Render molecule with highlighted substructures (deprecated, use depict.with_highlights)."""

__all__ = ["render_with_highlights"]

from collections import defaultdict

from .depict.collect_highlights import SmartsEntry
from .depict.with_highlights import with_highlights


def render_with_highlights(
    name: str,
    smi: str,
    smarts_mols: list[SmartsEntry],
    match_counter: defaultdict | None = None,
    width: int = 200,
    height: int = 200,
) -> str:
    """
    Render a molecule as SVG with highlighted substructures.

    Deprecated: Use depict.with_highlights instead.

    Args:
        name: Display name for the molecule
        smi: SMILES string
        smarts_mols: List of (name, smarts, smarts_mol, rgb_color) tuples
        match_counter: Optional defaultdict to count matches per SMARTS
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        HTML string with SVG and labels
    """
    return with_highlights(
        name=name,
        smiles=smi,
        smarts_entries=smarts_mols,
        match_counter=match_counter,
        width=width,
        height=height,
    )
