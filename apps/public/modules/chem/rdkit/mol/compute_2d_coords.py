"""Compute 2D coordinates for RDKit Mol."""

__all__ = ["compute_2d_coords"]

from rdkit.Chem import Mol
from rdkit.Chem.rdDepictor import Compute2DCoords


def compute_2d_coords(mol: Mol) -> None:
    """Compute 2D coordinates for molecule depiction (modifies mol in place)."""
    Compute2DCoords(mol)
