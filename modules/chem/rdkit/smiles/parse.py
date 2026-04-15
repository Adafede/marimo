"""Parse SMILES string to RDKit Mol object."""

__all__ = ["parse"]

from rdkit.Chem import Mol, MolFromSmiles


def parse(smiles: str) -> Mol | None:
    """Parse SMILES string to RDKit Mol object.

    Parameters
    ----------
    smiles : str
        Smiles.

    Returns
    -------
    Mol | None
        Return value produced by parse.
    """
    if not smiles:
        return None
    return MolFromSmiles(smiles)
