"""Load Indigo molecule from SMILES string."""

__all__ = ["mol_from_smiles"]

from indigo import IndigoObject
from ..indigo_instance import get_indigo


def mol_from_smiles(
    smiles: str | None,
    aromatize: bool = True,
) -> IndigoObject | None:
    """Load molecule from SMILES string using the shared Indigo instance.

    Parameters
    ----------
    smiles : str | None
        Smiles.
    aromatize : bool
        True. Default is True.

    Returns
    -------
    IndigoObject | None
        Result mol from smiles.
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
