"""Find Maximum Common Substructure SMARTS from SMILES list."""

__all__ = ["find_mcs_smarts"]

from typing import Tuple, Optional


def find_mcs_smarts(smiles_list: list[tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Find Maximum Common Substructure SMARTS from a list of SMILES.

    Args:
        smiles_list: List of (name, smiles) tuples

    Returns:
        Tuple of (smarts_string, error_message)
    """
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdFMCS import FindMCS

    mols = [MolFromSmiles(smi) for _, smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    if len(mols) < 2:
        return None, "⚠️ Need at least two valid SMILES to find MCS."

    mcs_result = FindMCS(mols)
    if mcs_result.canceled or not mcs_result.smartsString:
        return None, "⚠️ Could not determine MCS."

    return mcs_result.smartsString, None
