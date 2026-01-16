"""Find substructure matches in molecule."""

__all__ = ["find_matches"]

from collections.abc import Iterator

from rdkit.Chem import Mol


def find_matches(mol: Mol, pattern: Mol) -> tuple[tuple[int, ...], ...]:
    """
    Find all substructure matches for a SMARTS pattern in a molecule.
    
    Args:
        mol: Target molecule to search
        pattern: SMARTS pattern as Mol object
    
    Returns:
        Tuple of tuples containing atom indices for each match
    """
    return mol.GetSubstructMatches(pattern)


def iter_match_atoms(matches: tuple[tuple[int, ...], ...]) -> Iterator[int]:
    """
    Iterate over all atom indices from substructure matches.
    
    Args:
        matches: Tuple of match tuples from GetSubstructMatches
    
    Yields:
        Individual atom indices
    """
    for match in matches:
        yield from match
