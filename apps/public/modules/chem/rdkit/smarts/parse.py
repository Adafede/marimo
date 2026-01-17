"""Parse SMARTS string to RDKit Mol object."""

__all__ = ["parse"]

from rdkit.Chem import Mol, MolFromSmarts


def parse(smarts: str) -> Mol | None:
    """Parse SMARTS string to RDKit Mol pattern object."""
    if not smarts:
        return None
    return MolFromSmarts(smarts)
