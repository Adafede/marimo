"""Load Indigo molecule from SMILES string."""

__all__ = ["mol_from_smiles"]

from indigo import IndigoObject
from ..indigo_instance import get_indigo


def mol_from_smiles(
    smiles: str | None,
    aromatize: bool = True,
) -> IndigoObject | None:
    """
    Load molecule from SMILES string using the shared Indigo instance.

    Args:
        smiles: SMILES string to parse
        aromatize: Whether to aromatize the molecule

    Returns:
        IndigoObject molecule or None if invalid/empty
    """
    if not smiles:
        return None

    indigo = get_indigo()
    try:
        mol = indigo.loadMolecule(smiles)
        if aromatize:
            mol.aromatize()
        return mol
    except Exception:
        return None
