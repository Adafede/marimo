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
    """Render a molecule as SVG with highlighted substructures.

                Deprecated: Use depict.with_highlights instead.

    Parameters
    ----------
    name : str
        Name.
    smi : str
        Smi.
    smarts_mols : list[SmartsEntry]
        Smarts mols.
    match_counter : defaultdict | None
        None. Default is None.
    width : int
        Default is 200.
    height : int
        Default is 200.

    Returns
    -------
    str
        String representation of render with highlights.

    """
    return with_highlights(
        name=name,
        smiles=smi,
        smarts_entries=smarts_mols,
        match_counter=match_counter,
        width=width,
        height=height,
    )
