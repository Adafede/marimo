"""Collect atom highlight data from SMARTS pattern matches."""

__all__ = ["collect_highlights", "RGBColor", "SmartsEntry"]

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from rdkit.Chem import Mol

from ..smarts.find_matches import find_matches, iter_match_atoms

RGBColor = tuple[float, float, float]
SmartsEntry = tuple[str, str, Any, RGBColor]  # (name, smarts_str, smarts_mol, color)


def _iter_pattern_highlights(
    mol: Mol,
    smarts_entries: list[SmartsEntry],
    match_counter: defaultdict | None,
) -> Iterator[tuple[list[int], RGBColor, str]]:
    """
    Iterate over pattern matches yielding highlight data.

    Yields:
        Tuple of (atom_indices, color, tooltip_string)
    """
    for pattern_name, _, smarts_mol, color in smarts_entries:
        matches = find_matches(mol=mol, pattern=smarts_mol)
        if not matches:
            continue

        atom_indices = list(iter_match_atoms(matches=matches))
        tooltip = f"{pattern_name}: {len(matches)} match(es)"

        if match_counter is not None:
            match_counter[pattern_name] += 1

        yield atom_indices, color, tooltip


def collect_highlights(
    mol: Mol,
    smarts_entries: list[SmartsEntry],
    match_counter: defaultdict | None = None,
) -> tuple[list[int], dict[int, RGBColor], list[str]]:
    """
    Collect atom highlight data from SMARTS pattern matches.

    Args:
        mol: RDKit Mol object
        smarts_entries: List of (name, smarts_str, smarts_mol, rgb_color) tuples
        match_counter: Optional defaultdict to count matches per pattern name

    Returns:
        Tuple of (atom_indices, atom_colors, tooltip_strings)
    """
    atom_ids: list[int] = []
    colors: dict[int, RGBColor] = {}
    tooltips: list[str] = []

    for indices, color, tooltip in _iter_pattern_highlights(
        mol=mol,
        smarts_entries=smarts_entries,
        match_counter=match_counter,
    ):
        atom_ids.extend(indices)
        colors.update(dict.fromkeys(indices, color))
        tooltips.append(tooltip)

    return atom_ids, colors, tooltips
