"""Find substructure matches in molecule."""

__all__ = ["find_matches"]

from collections.abc import Iterator

from rdkit.Chem import Mol


def find_matches(mol: Mol, pattern: Mol) -> tuple[tuple[int, ...], ...]:
    """Find all substructure matches for a SMARTS pattern in a molecule.

Parameters
----------
mol : Mol
    Mol.
pattern : Mol
    Pattern.

Returns
-------
tuple[tuple[int, ...], ...]
    Computed result.
    """
    return mol.GetSubstructMatches(pattern)


def iter_match_atoms(matches: tuple[tuple[int, ...], ...]) -> Iterator[int]:
    """Iterate over all atom indices from substructure matches.

Parameters
----------
matches : tuple[tuple[int, ...], ...]
    Matches.

Yields
------
int
    Generated values.
    """
    for match in matches:
        yield from match
