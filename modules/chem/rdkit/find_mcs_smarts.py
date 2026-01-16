"""Find Maximum Common Substructure SMARTS from SMILES list."""

__all__ = ["find_mcs_smarts"]

# RDKit imports are inside the function for lazy loading (optional dependency)

_MIN_MOLECULES_FOR_MCS = 2


def find_mcs_smarts(
    smiles_list: list[tuple[str, str]],
) -> tuple[str | None, str | None]:
    """
    Find Maximum Common Substructure SMARTS from a list of SMILES.

    Args:
        smiles_list: List of (name, smiles) tuples

    Returns:
        Tuple of (smarts_string, error_message)
    """
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdFMCS import FindMCS

    valid_mols = [
        mol for _, smi in smiles_list if (mol := MolFromSmiles(smi)) is not None
    ]

    if len(valid_mols) < _MIN_MOLECULES_FOR_MCS:
        return (
            None,
            f"⚠️ Need at least {_MIN_MOLECULES_FOR_MCS} valid SMILES to find MCS.",
        )

    mcs_result = FindMCS(valid_mols)
    if mcs_result.canceled or not mcs_result.smartsString:
        return None, "⚠️ Could not determine MCS."

    return mcs_result.smartsString, None
